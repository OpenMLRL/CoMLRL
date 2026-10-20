"""Preference-specific state and portable assets for safe-boundary checkpoints."""

import errno
import hashlib
import os
import json
from pathlib import Path
import shutil
import stat
import tempfile
import threading

import torch
from transformers import AutoTokenizer

from comlrl.utils.checkpoint import capture_trainer_state
from comlrl.utils.distributed import unwrap_model
from .output_paths import resolve_output_path


_EXTRA_FIELDS = {
    "_reward_model_active",
    "_evaluating_with_task_reward",
    "_initial_policy_checkpoint_saved",
    "target_seed",
    "comparator_seed",
    "_comparator_rng_states",
    "_comparator_rng_streams",
    "actor_critic",
}


def validate_preference_progress(trainer, state, kind):
    """Check saved cursors before the generic loader changes live state."""
    progress = state.get("progress")
    if not isinstance(progress, dict):
        raise ValueError("Preference checkpoint progress must be a dictionary.")
    if not progress:
        return
    if progress.get("kind") != kind:
        raise ValueError(f"The checkpoint does not contain {kind} training progress.")

    if kind == "marlhf":
        phase = progress.get("phase")
        if phase not in ("online_rl", "complete") or (
            phase == "online_rl" and "rl" not in progress
        ):
            raise ValueError("Invalid MARLHF checkpoint phase.")
        rl = progress.get("rl", {})
        expected_kind = (
            "actor_critic" if trainer.args.rl_algorithm in ("maac", "iac") else "magrpo"
        )
        if not isinstance(rl, dict) or (
            rl
            and (
                rl.get("kind") != expected_kind
                or type(rl.get("epoch")) is not int
                or not 0 <= rl["epoch"] <= int(trainer.args.num_train_epochs)
            )
        ):
            raise ValueError("Invalid MARLHF online-RL checkpoint cursor.")
        if (phase == "online_rl" or rl) and "reward_model" not in state.get(
            "models", {}
        ):
            raise ValueError("MARLHF online-RL checkpoint is missing its reward model.")
        if rl and expected_kind == "actor_critic" and rl["epoch"] > 0:
            extra = state.get("extra")
            child = extra.get("actor_critic") if isinstance(extra, dict) else None
            if not isinstance(child, dict) or child.get("progress") != rl:
                raise ValueError(
                    "MARLHF checkpoint is missing matching actor-critic progress."
                )
            if any(child.get("runtime", {}).get("rollout_buffers", [])):
                raise ValueError(
                    "MARLHF actor-critic checkpoint has unfinished rollouts."
                )
        elif rl and any(state.get("runtime", {}).get("rollout_buffers", [])):
            raise ValueError("MARLHF checkpoint has unfinished rollouts.")
        return

    cursor = "epoch" if kind == "madpo" else "iteration"
    total = int(
        trainer.args.num_train_epochs
        if cursor == "epoch"
        else trainer.args.num_iterations
    )
    required = {"kind", cursor, "updates_seen", "complete"}
    required.add("preference_pairs" if kind == "madpo" else "total_pairs")
    if not required.issubset(progress):
        raise ValueError(f"Incomplete {kind} checkpoint progress.")
    if type(progress[cursor]) is not int or not 0 <= progress[cursor] <= total:
        raise ValueError(f"Invalid {kind} checkpoint {cursor}.")
    if type(progress["complete"]) is not bool:
        raise ValueError("Invalid preference checkpoint completion flag.")
    counters = ("updates_seen",) if kind == "madpo" else ("updates_seen", "total_pairs")
    for name in counters:
        if type(progress[name]) is not int or progress[name] < 0:
            raise ValueError(f"Invalid preference checkpoint counter: {name}")
    if kind == "madpo":
        from .madpo import PreferencePair  # madpo imports this module

        pairs = progress["preference_pairs"]
        if not isinstance(pairs, list) or not all(
            isinstance(pair, PreferencePair) for pair in pairs
        ):
            raise ValueError("Invalid MADPO checkpoint preference data.")


def validate_preference_extra_state(state):
    if not isinstance(state, dict) or not set(state).issubset(_EXTRA_FIELDS):
        raise ValueError("Invalid preference checkpoint extra-state fields.")
    for name in (
        "_reward_model_active",
        "_evaluating_with_task_reward",
        "_initial_policy_checkpoint_saved",
    ):
        if name in state and type(state[name]) is not bool:
            raise ValueError(f"Invalid preference checkpoint flag: {name}")
    for name in ("target_seed", "comparator_seed"):
        if name in state and type(state[name]) is not int:
            raise ValueError(f"Invalid preference checkpoint seed: {name}")
    if "actor_critic" in state and not isinstance(state["actor_critic"], dict):
        raise ValueError("Invalid nested actor-critic checkpoint state.")
    if "_comparator_rng_states" in state:
        rng = state["_comparator_rng_states"]
        if not isinstance(rng, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            for key, value in rng.items()
        ):
            raise ValueError("Invalid comparator RNG checkpoint state.")


def preference_extra_state(trainer):
    state = {}
    for name in (
        "_reward_model_active",
        "_evaluating_with_task_reward",
        "_initial_policy_checkpoint_saved",
        "target_seed",
        "comparator_seed",
        "_comparator_rng_states",
        "_comparator_rng_streams",
    ):
        if hasattr(trainer, name):
            state[name] = getattr(trainer, name)
    child = getattr(trainer, "_checkpoint_ac_trainer", None)
    if child is not None:
        state["actor_critic"] = capture_trainer_state(child)
    elif getattr(trainer, "_checkpoint_pending_ac_state", None) is not None:
        state["actor_critic"] = trainer._checkpoint_pending_ac_state
    return state


def restore_preference_extra_state(trainer, state, checkpoint_dir):
    validate_preference_extra_state(state)
    for name, value in state.items():
        if name != "actor_critic":
            if name == "_comparator_rng_states":
                value = {key: tensor.detach().clone() for key, tensor in value.items()}
            setattr(trainer, name, value)
    if "_comparator_rng_states" in state:
        trainer._comparator_rng_locks = {}
        trainer._comparator_rng_registry_lock = threading.Lock()
    # A reused parent may still own the child from its previous training run.
    # Only the loaded snapshot is authoritative until a new child is constructed.
    trainer._checkpoint_ac_trainer = None
    trainer._checkpoint_pending_ac_state = state.get("actor_critic")
    trainer._checkpoint_loaded_from = str(checkpoint_dir)
    if getattr(trainer, "reward_model", None) is not None:
        # Every supported preference boundary is after reward-model training.
        trainer.reward_model.eval()


def _asset_path(checkpoint_dir, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Checkpoint assets require relative paths.")
    root = Path(checkpoint_dir).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Checkpoint asset path escapes the checkpoint directory.")
    return path


def validate_preference_assets(trainer, checkpoint_dir, manifest):
    allowed = {
        "iterative",
        "replay",
        "policy_history",
        "reward_tokenizer",
        "comparator_agents",
        "centralized_comparator",
    }
    if not isinstance(manifest, dict) or not set(manifest).issubset(allowed):
        raise ValueError("Invalid preference checkpoint asset manifest.")
    if "iterative" in manifest and manifest["iterative"] is not True:
        raise ValueError("Invalid iterative checkpoint asset marker.")
    if manifest.get("iterative"):
        _preference_resume_destinations(trainer, checkpoint_dir)
    records = manifest.get("replay", [])
    if not isinstance(records, list):
        raise ValueError("Invalid replay checkpoint manifest.")
    previous = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "iteration",
            "num_pairs",
            "path",
        }:
            raise ValueError("Invalid replay checkpoint record.")
        iteration, count = record["iteration"], record["num_pairs"]
        if (
            type(iteration) is not int
            or iteration <= previous
            or type(count) is not int
            or count < 1
        ):
            raise ValueError("Invalid replay checkpoint order or count.")
        previous = iteration
        path = _asset_path(checkpoint_dir, record["path"])
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if (
            not isinstance(payload, dict)
            or payload.get("iteration") != iteration
            or payload.get("num_pairs") != count
            or not isinstance(payload.get("pairs"), list)
            or len(payload["pairs"]) != count
        ):
            raise ValueError("Replay checkpoint contents do not match their manifest.")
    directories = [
        manifest[name]
        for name in ("policy_history", "reward_tokenizer", "centralized_comparator")
        if name in manifest
    ]
    if "comparator_agents" in manifest:
        comparators = manifest["comparator_agents"]
        if not isinstance(comparators, list) or len(comparators) != len(trainer.agents):
            raise ValueError("Invalid comparator checkpoint actor count.")
        directories.extend(comparators)
    for relative in directories:
        directory = _asset_path(checkpoint_dir, relative)
        if not directory.is_dir():
            raise ValueError(
                f"Missing preference checkpoint asset directory: {relative}"
            )
        for path in directory.rglob("*"):
            _asset_path(
                checkpoint_dir, str(path.relative_to(Path(checkpoint_dir).resolve()))
            )
    if "policy_history" in manifest:
        history = _asset_path(checkpoint_dir, manifest["policy_history"])
        for policy in history.iterdir():
            if not policy.is_dir() or not all(
                (policy / f"agent_{index}").is_dir()
                for index in range(len(trainer.agents))
            ):
                raise ValueError("Invalid history-policy checkpoint assets.")


_ASSET_POOL_NAME = ".comlrl-preference-assets"


def _asset_stat_key(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _require_unchanged_asset(path, signature):
    try:
        unchanged = _asset_stat_key(Path(path).stat()) == signature
    except FileNotFoundError:
        unchanged = False
    if not unchanged:
        raise RuntimeError("A preference asset changed during checkpoint saving.")


def _asset_digest(path, *, cache=None):
    """Hash new or changed source files; cache only stable stat signatures."""
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Checkpoint assets must be regular files.")
    signature = _asset_stat_key(info)
    key = str(path.absolute())
    cached = cache.get(key) if cache is not None else None
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if _asset_stat_key(os.fstat(stream.fileno())) != signature:
            raise RuntimeError("A preference asset changed during checkpoint saving.")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if _asset_stat_key(os.fstat(stream.fileno())) != signature:
            raise RuntimeError("A preference asset changed during checkpoint saving.")
    _require_unchanged_asset(path, signature)
    result = digest.hexdigest()
    if cache is not None:
        cache[key] = (signature, result)
    return result


def _copy_checkpoint_asset(source, destination, pool, *, digest_cache=None):
    """Link immutable snapshots to a copied blob, never to a live training file."""
    source, destination, pool = Path(source), Path(destination), Path(pool)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if pool.is_symlink():
        raise ValueError("The checkpoint asset pool must not be a symbolic link.")
    pool.mkdir(parents=True, exist_ok=True)
    source_signature = _asset_stat_key(source.stat())
    digest = _asset_digest(source, cache=digest_cache)
    _require_unchanged_asset(source, source_signature)
    blob = pool / digest
    if blob.is_symlink():
        raise ValueError("Checkpoint asset blobs must not be symbolic links.")
    if not blob.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=".asset-", dir=pool)
        os.close(descriptor)
        temporary = Path(temporary)
        try:
            shutil.copy2(source, temporary)
            if _asset_digest(temporary) != digest:
                raise RuntimeError(
                    "A preference asset changed during checkpoint saving."
                )
            _require_unchanged_asset(source, source_signature)
            temporary.chmod((stat.S_IMODE(temporary.stat().st_mode) & 0o555) | 0o400)
            try:
                os.link(temporary, blob)
            except FileExistsError:
                pass
            except OSError as error:
                if error.errno not in (
                    errno.EXDEV,
                    errno.EPERM,
                    errno.EOPNOTSUPP,
                    errno.EMLINK,
                ):
                    raise
                os.replace(temporary, blob)
        finally:
            temporary.unlink(missing_ok=True)
    blob_info = blob.stat()
    # Blobs were fully verified when created and are never opened for writing.
    # Re-reading historical model bytes here would make periodic saves quadratic.
    if (
        not stat.S_ISREG(blob_info.st_mode)
        or blob_info.st_size != source_signature[2]
        or blob_info.st_mode & 0o222
    ):
        raise ValueError("The immutable checkpoint asset pool contains a damaged blob.")
    _require_unchanged_asset(source, source_signature)
    # Generated tokenizer/comparator files are interned in place after serialization.
    if destination == source:
        destination.unlink()
    try:
        os.link(blob, destination)
    except OSError as error:
        if error.errno not in (
            errno.EXDEV,
            errno.EPERM,
            errno.EOPNOTSUPP,
            errno.EMLINK,
        ):
            raise
        shutil.copy2(blob, destination)
    if destination != source:
        try:
            _require_unchanged_asset(source, source_signature)
        except RuntimeError:
            destination.unlink()
            raise
    return str(destination)


def prune_preference_asset_pool(checkpoint_root):
    """Drop unreferenced blobs after checkpoint publication and retention pruning.

    Published snapshots retain their own hardlinks and are independently portable.
    A staging directory signals that another save may still be linking its assets.
    """
    root = Path(checkpoint_root)
    pool = root / _ASSET_POOL_NAME
    if not pool.is_dir() or pool.is_symlink():
        return
    if any(
        path.name.startswith(".") and path != pool and (path / "assets").is_dir()
        for path in root.iterdir()
    ):
        return
    for path in pool.iterdir():
        if (
            len(path.name) == 64
            and all(char in "0123456789abcdef" for char in path.name)
            and not path.is_symlink()
            and path.is_file()
            and path.stat().st_nlink == 1
        ):
            path.unlink()


def save_preference_assets(trainer, checkpoint_dir):
    """Snapshot portable assets without changing live paths or training RNG.

    Content-addressed, read-only blobs share storage across checkpoints. Live
    files are copied into the pool first, and restoration always makes new copies.
    """
    root = Path(checkpoint_dir)
    pool = root.parent / _ASSET_POOL_NAME
    manifest = {}
    digest_cache = getattr(trainer, "_checkpoint_asset_digest_cache", None)
    if digest_cache is None:
        digest_cache = {}
        trainer._checkpoint_asset_digest_cache = digest_cache

    def copy_asset(source, destination):
        return _copy_checkpoint_asset(
            source,
            destination,
            pool,
            digest_cache=(None if Path(source) == Path(destination) else digest_cache),
        )

    if hasattr(trainer, "_preference_replay_shards"):
        manifest["iterative"] = True
        manifest["replay"] = []
        for index, shard in enumerate(
            getattr(trainer, "_preference_replay_shards_state", [])
        ):
            relative = f"assets/preference_replay/shard_{index:06d}.json"
            copy_asset(shard.path, root / relative)
            manifest["replay"].append(
                {
                    "iteration": shard.iteration,
                    "num_pairs": shard.num_pairs,
                    "path": relative,
                }
            )
        history = getattr(trainer, "_policy_checkpoint_dir_path", None)
        if history and Path(history).is_dir():
            relative = "assets/policy_history"
            (root / relative).mkdir(parents=True)
            # A caller may use the same parent for full-state checkpoints.
            # Copy policy directories only, never the current staging tree.
            for path in Path(history).iterdir():
                if path.is_dir() and (
                    path.name == "initial"
                    or (path.name.startswith("iteration_") and path.name[10:].isdigit())
                ):
                    shutil.copytree(
                        path, root / relative / path.name, copy_function=copy_asset
                    )
            manifest["policy_history"] = relative

    generated = []
    reward_tokenizer = getattr(trainer, "reward_tokenizer", None)
    if reward_tokenizer is not None:
        relative = "assets/reward_tokenizer"
        reward_tokenizer.save_pretrained(str(root / relative))
        generated.append(root / relative)
        manifest["reward_tokenizer"] = relative

    comparator_agents = getattr(trainer, "_comparator_agents", None)
    if comparator_agents:
        manifest["comparator_agents"] = []
        for index, agent in enumerate(comparator_agents):
            relative = f"assets/comparator/agent_{index}"
            unwrap_model(agent).save_pretrained(str(root / relative))
            generated.append(root / relative)
            manifest["comparator_agents"].append(relative)
    central = getattr(trainer, "_centralized_comparator_agent", None)
    if central is not None:
        relative = "assets/centralized_comparator"
        unwrap_model(central).save_pretrained(str(root / relative))
        generated.append(root / relative)
        manifest["centralized_comparator"] = relative
    for directory in generated:
        for path in directory.rglob("*"):
            if path.is_file():
                copy_asset(path, path)
    return manifest


def _resume_output_path(trainer, directory, cache_attribute, config_attribute=None):
    """Resolve the iterative output convention without creating directories yet."""
    cached = getattr(trainer, cache_attribute, None)
    path = resolve_output_path(
        directory,
        cached=cached,
        configured=(
            getattr(trainer.args, config_attribute, None)
            if config_attribute and not cached
            else None
        ),
        wandb_config=getattr(trainer, "wandb_config", None),
    )
    if path is None:
        base = getattr(trainer, "_checkpoint_directory", None)
        if base:
            # Checkpoint saving expands this API argument; configured replay,
            # policy, and W&B output paths retain their literal '~' convention.
            base = Path(base).expanduser().absolute()
        else:
            configured = [
                Path(os.path.abspath(str(value))).parent
                for name in ("preference_replay_dir", "policy_checkpoint_dir")
                if (value := getattr(trainer.args, name, None))
            ]
            if configured and len(set(configured)) == 1:
                base = configured[0]
        path = resolve_output_path(directory, default_base=base or os.getcwd())
    return Path(path).absolute().resolve()


def _preference_resume_destinations(trainer, checkpoint_dir):
    """Preflight output paths without changing files or live trainer attributes."""
    source_root = Path(checkpoint_dir).resolve()
    destinations = {
        "_preference_replay_dir_path": _resume_output_path(
            trainer,
            "preference_replay",
            "_preference_replay_dir_path",
            "preference_replay_dir",
        ),
        "_policy_checkpoint_dir_path": _resume_output_path(
            trainer,
            "policy_checkpoints",
            "_policy_checkpoint_dir_path",
            "policy_checkpoint_dir",
        ),
        "_reward_distribution_dir_path": _resume_output_path(
            trainer,
            "reward_distributions",
            "_reward_distribution_dir_path",
        ),
    }
    for path in destinations.values():
        if path.is_relative_to(source_root):
            raise ValueError("Resumed outputs must be outside the source checkpoint.")
        if path.exists() and not path.is_dir():
            raise ValueError(f"Resumed output path is not a directory: {path}")
    return destinations


def _copy_restored_asset(source, destination):
    """Restored runs own writable copies, independent of every saved snapshot."""
    result = shutil.copy2(source, destination)
    path = Path(destination)
    path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
    return result


def load_preference_assets(trainer, checkpoint_dir, manifest):
    if manifest.get("iterative"):
        from .iterative import PreferenceReplayShard  # iterative imports this module

        destinations = _preference_resume_destinations(trainer, checkpoint_dir)
        allocated = set()
        for name, path in destinations.items():
            path.mkdir(parents=True, exist_ok=True)
            # Preserve an existing run instead of silently overwriting its files.
            if path in allocated or any(path.iterdir()):
                path = Path(tempfile.mkdtemp(prefix="resume-", dir=path))
            destinations[name] = path
            allocated.add(path)
        replay_dir = destinations["_preference_replay_dir_path"]
        shards = []
        for record in manifest.get("replay", []):
            path = replay_dir / f"iteration_{int(record['iteration']):04d}.json"
            _copy_restored_asset(_asset_path(checkpoint_dir, record["path"]), path)
            shards.append(
                PreferenceReplayShard(
                    iteration=int(record["iteration"]),
                    path=str(path),
                    num_pairs=int(record["num_pairs"]),
                )
            )
        history_dir = destinations["_policy_checkpoint_dir_path"]
        if manifest.get("policy_history"):
            shutil.copytree(
                _asset_path(checkpoint_dir, manifest["policy_history"]),
                history_dir,
                copy_function=_copy_restored_asset,
                dirs_exist_ok=True,
            )
        trainer._preference_replay_shards_state = shards
        for name, path in destinations.items():
            setattr(trainer, name, str(path))

    if manifest.get("reward_tokenizer"):
        trainer.reward_tokenizer = AutoTokenizer.from_pretrained(
            str(_asset_path(checkpoint_dir, manifest["reward_tokenizer"]))
        )
    if manifest.get("comparator_agents"):
        trainer._comparator_agents = trainer._load_frozen_policy_agents(
            [
                str(_asset_path(checkpoint_dir, relative))
                for relative in manifest["comparator_agents"]
            ]
        )
    if manifest.get("centralized_comparator"):
        trainer._centralized_comparator_agent = (
            trainer._load_single_frozen_policy_agent(
                str(_asset_path(checkpoint_dir, manifest["centralized_comparator"])),
                trainer._centralized_comparator_agent_index(),
            )
        )


def train_iterations_checkpointed(trainer, *, reward_model, **kwargs):
    """Checkpoint complete iterations; replay generation stays indivisible."""
    kind = "marlhf_iter" if reward_model else "madpo_iter"
    name = "MARLHFIter" if reward_model else "MADPOIter"
    if int(trainer.args.num_turns) != 1:
        raise ValueError(f"{name} currently supports num_turns=1 only.")
    saved = getattr(trainer, "_checkpoint_progress", {})
    if not isinstance(saved, dict):
        raise ValueError("Iterative checkpoint progress must be a dictionary.")
    progress = dict(saved)
    required = {"kind", "iteration", "updates_seen", "total_pairs", "complete"}
    if progress and (progress.get("kind") != kind or not required.issubset(progress)):
        raise ValueError(f"The checkpoint does not contain {name} progress.")
    start = progress.get("iteration", 0)
    total_iterations = int(trainer.args.num_iterations)
    if type(start) is not int or not 0 <= start <= total_iterations:
        raise ValueError("Invalid iterative checkpoint iteration.")
    if progress and type(progress["complete"]) is not bool:
        raise ValueError("Invalid iterative checkpoint completion flag.")
    for counter_name in ("updates_seen", "total_pairs"):
        value = progress.get(counter_name, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid iterative checkpoint counter: {counter_name}")
    if progress.get("complete") and start == total_iterations:
        return
    trainer._checkpoint_init_wandb()
    if not reward_model:
        for index, agent in enumerate(trainer.agents):
            agent.to(trainer.agent_devices[index])
            agent.train()
    trainer._maybe_save_initial_policy_checkpoint()
    updates_seen = progress.get("updates_seen", 0)
    total_pairs = progress.get("total_pairs", 0)

    def commit(next_iteration):
        complete = next_iteration == total_iterations
        trainer._checkpoint_commit(
            {
                "kind": kind,
                "iteration": next_iteration,
                "updates_seen": updates_seen,
                "total_pairs": total_pairs,
                "complete": complete,
            },
            force=complete,
        )

    for iteration in range(start, total_iterations):
        if reward_model:
            trainer._reward_model_active = False
            trainer._evaluating_with_task_reward = False
            if trainer.args.preference_scoring_reward != "reward_model":
                trainer._clear_reward_model()
        trainer._reset_iteration_reward_distribution()
        trainer._prepare_iteration_current_copy_comparator()
        try:
            pairs = trainer._build_preference_dataset(iteration_idx=iteration, **kwargs)
        finally:
            trainer._clear_iteration_current_copy_comparator()
            if reward_model:
                trainer._clear_iteration_model_comparator()
        current_pair_count = len(pairs)
        train_pairs = trainer._select_iteration_preference_pairs(
            pairs, iteration_idx=iteration
        )
        total_pairs += len(train_pairs)
        trainer._log_iteration_replay(
            iteration,
            train_pairs=train_pairs,
            current_pair_count=current_pair_count,
            train_pair_count=len(train_pairs),
        )
        pairs.clear()
        if not train_pairs:
            if trainer.verbose:
                print(
                    f"{name}: no replay preference pairs were available "
                    f"for iteration {iteration + 1} "
                    f"(current generated {current_pair_count})."
                )
            trainer._maybe_save_iteration_policy_checkpoint(iteration)
            commit(iteration + 1)
            continue
        if reward_model:
            trainer._clear_reward_model()
            trainer._init_reward_model()
            trainer._train_reward_model(train_pairs)
            trainer._reward_model_active = True
            trainer._train_online_rl(**kwargs)
        else:
            updates_seen = trainer._train_preference_pairs(
                train_pairs,
                updates_seen=updates_seen,
                desc=f"{name} iteration {iteration + 1}/{total_iterations}",
            )
        trainer._maybe_save_iteration_policy_checkpoint(iteration)
        if reward_model and trainer.args.preference_scoring_reward != "reward_model":
            trainer._clear_reward_model()
        commit(iteration + 1)

    if start == total_iterations:
        commit(start)
    if total_pairs == 0 and trainer.verbose:
        print(f"{name}: no non-tied preference pairs were generated.")
