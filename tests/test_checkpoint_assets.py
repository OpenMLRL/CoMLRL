"""Portable preference assets: replay and policy history, digests, output paths."""

from collections import Counter
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from comlrl.trainers.pbrl import (
    MADPOIterTrainer,
    MARLHFIterTrainer,
    checkpoint_extras,
)
from comlrl.trainers.pbrl.checkpoint_extras import (
    _preference_resume_destinations,
    _resume_output_path,
    load_preference_assets,
    prune_preference_asset_pool,
    save_preference_assets,
    validate_preference_assets,
)
from comlrl.trainers.pbrl.iterative import PreferenceReplayShard
from comlrl.trainers.pbrl.output_paths import resolve_output_path
from comlrl.utils.checkpoint import STATE_FILENAME

from checkpoint_helpers import (
    Interrupted,
    TinyTrainer,
    assert_same_training,
    iterative_trainer,
    pair,
)
from helpers import assert_nested_equal, rng_state, seed_all


def asset_trainer(directory):
    return SimpleNamespace(
        args=SimpleNamespace(
            preference_replay_dir=str(directory / "replay"),
            policy_checkpoint_dir=str(directory / "policies"),
        ),
        wandb_config={"output_dir": str(directory)},
        agents=[object()],
        _preference_replay_shards=lambda: [],
        _preference_replay_shards_state=[],
    )


def source_assets(directory):
    trainer = asset_trainer(directory)
    replay = directory / "replay" / "iteration_0001.json"
    replay.parent.mkdir(parents=True)
    replay.write_text(json.dumps({"iteration": 1, "num_pairs": 1, "pairs": [{}]}))
    trainer._preference_replay_shards_state = [PreferenceReplayShard(1, str(replay), 1)]
    history = directory / "policies"
    weights = history / "initial" / "agent_0" / "weights.bin"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"unchanged historical policy")
    trainer._policy_checkpoint_dir_path = str(history)
    return trainer, replay, weights


class ObservedReader:
    def __init__(self, stream, path, counts, after_read=None):
        self.stream, self.path, self.counts = stream, path, counts
        self.after_read = after_read

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def fileno(self):
        return self.stream.fileno()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.counts[self.path] += len(data)
        if self.after_read is not None and data:
            callback, self.after_read = self.after_read, None
            callback()
        return data


def observe_hash_reads(monkeypatch, *, changed_path=None, after_read=None):
    counts = Counter()
    original = Path.open

    def open_file(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if mode == "rb":
            callback = after_read if path == changed_path else None
            return ObservedReader(stream, path, counts, callback)
        return stream

    monkeypatch.setattr(Path, "open", open_file)
    return counts


def trainer_for_paths():
    trainer = object.__new__(MADPOIterTrainer)
    trainer.args = SimpleNamespace()
    trainer.wandb_config = None
    return trainer


def test_assets_restore_to_new_run_with_independent_writable_files(tmp_path):
    source, _, _ = source_assets(tmp_path / "original")
    checkpoint = tmp_path / "checkpoints" / "checkpoint-1"
    manifest = save_preference_assets(source, checkpoint)
    portable = tmp_path / "portable"
    shutil.copytree(checkpoint, portable)
    shutil.rmtree(tmp_path / "original")
    shutil.rmtree(tmp_path / "checkpoints")

    target = asset_trainer(tmp_path / "new-run")
    validate_preference_assets(target, portable, manifest)
    load_preference_assets(target, portable, manifest)
    assert Path(target._preference_replay_dir_path) == tmp_path / "new-run" / "replay"
    assert Path(target._policy_checkpoint_dir_path) == tmp_path / "new-run" / "policies"
    assert (
        Path(target._reward_distribution_dir_path)
        == tmp_path / "new-run" / "reward_distributions"
    )
    restored = Path(target._policy_checkpoint_dir_path) / "initial/agent_0/weights.bin"
    saved = portable / "assets/policy_history/initial/agent_0/weights.bin"
    assert restored.stat().st_ino != saved.stat().st_ino
    assert restored.stat().st_mode & 0o200
    restored.write_bytes(b"new run changed its copy")
    assert saved.read_bytes() == b"unchanged historical policy"
    restored_replay = Path(target._preference_replay_shards_state[0].path)
    restored_replay.write_text("new run changed its replay")
    assert (
        json.loads((portable / manifest["replay"][0]["path"]).read_text())["num_pairs"]
        == 1
    )


def test_existing_run_outputs_are_preserved_in_persistent_resume_subdirectories(
    tmp_path,
):
    source, _, _ = source_assets(tmp_path / "original")
    checkpoint = tmp_path / "checkpoint"
    manifest = save_preference_assets(source, checkpoint)
    target = asset_trainer(tmp_path / "new-run")
    for name in ("replay", "policies", "reward_distributions"):
        directory = tmp_path / "new-run" / name
        directory.mkdir(parents=True)
        (directory / "keep.txt").write_text("existing output")
    load_preference_assets(target, checkpoint, manifest)
    for attribute, name in (
        ("_preference_replay_dir_path", "replay"),
        ("_policy_checkpoint_dir_path", "policies"),
        ("_reward_distribution_dir_path", "reward_distributions"),
    ):
        destination = Path(getattr(target, attribute))
        assert destination.parent == tmp_path / "new-run" / name
        assert destination.name.startswith("resume-")
        assert (destination.parent / "keep.txt").read_text() == "existing output"


def test_resume_rejects_output_inside_source_checkpoint_before_creating_it(tmp_path):
    source, _, _ = source_assets(tmp_path / "original")
    checkpoint = tmp_path / "checkpoint"
    manifest = save_preference_assets(source, checkpoint)
    target = asset_trainer(tmp_path / "new-run")
    target.args.preference_replay_dir = str(checkpoint / "forbidden-output")
    with pytest.raises(ValueError, match="outside the source checkpoint"):
        load_preference_assets(target, checkpoint, manifest)
    assert not (checkpoint / "forbidden-output").exists()
    assert not (tmp_path / "new-run").exists()


def test_unsafe_restore_destination_rejected_before_model_mutation(tmp_path):
    seed_all(10)
    source = iterative_trainer(tmp_path / "original")
    source._update_from_preference_batch([pair(4)])
    source.env_step = 99
    checkpoint = tmp_path / "checkpoint"
    source.save_checkpoint(checkpoint)
    seed_all(21)
    target = iterative_trainer(tmp_path / "new-run")
    target.args.policy_checkpoint_dir = str(checkpoint / "forbidden-output")
    before = target.agents[0].weight.detach().clone()
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="outside the source checkpoint"):
        target.load_checkpoint(checkpoint)
    assert torch.equal(target.agents[0].weight, before)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert not target.optimizers[0].state
    assert target.env_step == 0
    assert not hasattr(target, "_checkpoint_progress")
    assert not (checkpoint / "forbidden-output").exists()
    assert not (tmp_path / "new-run").exists()


@pytest.mark.parametrize("config", ["checkpoint", "wandb", "explicit"])
def test_resumed_output_paths_match_source_tilde_conventions(
    tmp_path, monkeypatch, config
):
    source, _, _ = source_assets(tmp_path / "original")
    checkpoint = tmp_path / "source-checkpoint"
    manifest = save_preference_assets(source, checkpoint)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    target = asset_trainer(tmp_path / "unused")
    target.args.preference_replay_dir = None
    target.args.policy_checkpoint_dir = None
    target.wandb_config = None
    if config == "checkpoint":
        target._checkpoint_directory = "~/runs"
    elif config == "wandb":
        target.wandb_config = {"output_dir": "~/runs"}
    else:
        target.args.preference_replay_dir = "~/runs/replay"
        target.args.policy_checkpoint_dir = "~/runs/policies"
    load_preference_assets(target, checkpoint, manifest)
    expected = tmp_path / ("home/runs" if config == "checkpoint" else "~/runs")
    for attribute in (
        "_preference_replay_dir_path",
        "_policy_checkpoint_dir_path",
        "_reward_distribution_dir_path",
    ):
        assert Path(getattr(target, attribute)).is_relative_to(expected)
    assert (tmp_path / "~").exists() == (config != "checkpoint")


def test_snapshots_share_immutable_assets_without_linking_live_history(tmp_path):
    source, replay, weights = source_assets(tmp_path / "run")
    root = tmp_path / "checkpoints"
    first, second, third = [root / f"checkpoint-{index}" for index in (1, 2, 3)]
    first_manifest = save_preference_assets(source, first)
    save_preference_assets(source, second)
    relative = "assets/policy_history/initial/agent_0/weights.bin"
    saved = first / relative
    assert saved.stat().st_ino == (second / relative).stat().st_ino
    assert saved.stat().st_ino != weights.stat().st_ino
    assert not saved.stat().st_mode & 0o222
    assert weights.stat().st_mode & 0o200
    assert (
        first / first_manifest["replay"][0]["path"]
    ).stat().st_ino != replay.stat().st_ino
    weights.write_bytes(b"a later policy snapshot")
    save_preference_assets(source, third)
    assert saved.read_bytes() == b"unchanged historical policy"
    assert (third / relative).read_bytes() == b"a later policy snapshot"
    assert (third / relative).stat().st_ino != saved.stat().st_ino
    # Every directory is portable even when the shared pool is removed.
    shutil.rmtree(root / ".comlrl-preference-assets")
    validate_preference_assets(source, first, first_manifest)


def test_asset_pool_pruning_keeps_referenced_blobs_and_skips_active_staging(tmp_path):
    source, _, _ = source_assets(tmp_path / "run")
    root = tmp_path / "checkpoints"
    checkpoint = root / "checkpoint-1"
    save_preference_assets(source, checkpoint)
    pool = root / ".comlrl-preference-assets"
    original = set(pool.iterdir())
    prune_preference_asset_pool(root)
    assert set(pool.iterdir()) == original
    shutil.rmtree(checkpoint)
    staging = root / ".checkpoint-2-pending" / "assets"
    staging.mkdir(parents=True)
    prune_preference_asset_pool(root)
    assert set(pool.iterdir()) == original
    shutil.rmtree(staging.parent)
    prune_preference_asset_pool(root)
    assert not list(pool.iterdir())


def test_snapshot_copy_fallback_when_filesystem_has_no_hardlinks(tmp_path, monkeypatch):
    source, _, weights = source_assets(tmp_path / "run")

    def unsupported_link(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hardlinks unsupported")

    monkeypatch.setattr(os, "link", unsupported_link)
    checkpoint = tmp_path / "checkpoint"
    manifest = save_preference_assets(source, checkpoint)
    validate_preference_assets(source, checkpoint, manifest)
    saved = checkpoint / "assets/policy_history/initial/agent_0/weights.bin"
    assert saved.read_bytes() == weights.read_bytes()
    assert saved.stat().st_ino != weights.stat().st_ino


@pytest.mark.parametrize("cls", [MADPOIterTrainer, MARLHFIterTrainer])
def test_iterative_retention_prunes_snapshots_and_resumes_shared_assets(tmp_path, cls):
    seed_all(51)
    baseline = iterative_trainer(tmp_path / "baseline", cls)
    baseline.train()
    expected_rng = rng_state()

    seed_all(51)
    interrupted = iterative_trainer(tmp_path / "original", cls)
    checkpoints = tmp_path / "checkpoints"
    pool = checkpoints / ".comlrl-preference-assets"
    orphan_bytes = b"asset left by an interrupted, unpublished snapshot"
    orphan = pool / hashlib.sha256(orphan_bytes).hexdigest()
    commit = interrupted._checkpoint_commit

    def commit_and_interrupt(progress, force=False):
        commit(progress, force=force)
        if progress["iteration"] == 2:
            orphan.write_bytes(orphan_bytes)
            orphan.chmod(0o444)
            assert orphan.stat().st_nlink == 1
        if progress["iteration"] == 3:
            raise Interrupted

    interrupted._checkpoint_commit = commit_and_interrupt
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=checkpoints, checkpoint_total_limit=2)
    assert sorted(path.name for path in checkpoints.glob("checkpoint-*")) == [
        "checkpoint-2",
        "checkpoint-3",
    ]
    assert not orphan.exists()
    relative = "assets/policy_history/initial/agent_0/weights.pt"
    previous = checkpoints / "checkpoint-2" / relative
    latest = checkpoints / "checkpoint-3" / relative
    assert latest.stat().st_ino == previous.stat().st_ino
    assert not latest.stat().st_mode & 0o222
    links = [
        path
        for path in checkpoints.rglob("*")
        if path.is_file() and path.stat().st_ino == latest.stat().st_ino
    ]
    assert latest.stat().st_nlink == len(links)
    assert latest.stat().st_nlink >= 3  # Two retained snapshots plus the pool.
    assert all(path.stat().st_nlink > 1 for path in pool.iterdir())

    shutil.rmtree(tmp_path / "original")
    seed_all(92)
    resumed = iterative_trainer(tmp_path / "resumed", cls)
    resumed.train(
        resume_from_checkpoint=checkpoints / "checkpoint-3",
        checkpoint_dir=checkpoints,
        checkpoint_total_limit=2,
    )
    assert_same_training(baseline, resumed)
    assert baseline.events == interrupted.events + resumed.events
    assert_nested_equal(rng_state(), expected_rng)
    assert sorted(path.name for path in checkpoints.glob("checkpoint-*")) == [
        "checkpoint-3",
        "checkpoint-4",
    ]
    assert Path(resumed._policy_checkpoint_dir_path).is_relative_to(
        tmp_path / "resumed"
    )
    assert Path(resumed._preference_replay_dir_path).is_relative_to(
        tmp_path / "resumed"
    )
    restored = Path(resumed._policy_checkpoint_dir_path) / "initial/agent_0/weights.pt"
    assert restored.stat().st_ino != latest.stat().st_ino
    assert restored.stat().st_mode & 0o200
    assert all(path.stat().st_nlink > 1 for path in pool.iterdir())


def test_history_hash_and_copy_bytes_grow_only_with_new_assets(tmp_path, monkeypatch):
    trainer, replay, initial = source_assets(tmp_path / "run")
    counts = observe_hash_reads(monkeypatch)
    copy_bytes = 0
    copy2 = checkpoint_extras.shutil.copy2

    def counted_copy(source, destination, *args, **kwargs):
        nonlocal copy_bytes
        copy_bytes += Path(source).stat().st_size
        return copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(checkpoint_extras.shutil, "copy2", counted_copy)
    unique_bytes = replay.stat().st_size + initial.stat().st_size
    for index in range(6):
        policy = initial.parents[2] / f"iteration_{index:04d}/agent_0/weights.bin"
        policy.parent.mkdir(parents=True)
        payload = bytes([index]) * (128 * 1024)
        policy.write_bytes(payload)
        unique_bytes += len(payload)
        checkpoint_extras.save_preference_assets(
            trainer, tmp_path / "checkpoints" / f"checkpoint-{index}"
        )
        # One source hash and one verification hash per newly copied file.
        assert sum(counts.values()) == 2 * unique_bytes
        assert copy_bytes == unique_bytes
    before_reads, before_copies = sum(counts.values()), copy_bytes
    checkpoint_extras.save_preference_assets(
        trainer, tmp_path / "checkpoints" / "checkpoint-unchanged"
    )
    assert sum(counts.values()) == before_reads
    assert copy_bytes == before_copies
    assert len(trainer._checkpoint_asset_digest_cache) == 8
    assert not any(".asset-" in name for name in trainer._checkpoint_asset_digest_cache)


@pytest.mark.parametrize("change", ["mtime", "same_size_restored_mtime", "replace"])
def test_digest_cache_invalidates_modified_or_replaced_source(
    tmp_path, monkeypatch, change
):
    path = tmp_path / "weights.bin"
    original, modified = b"original weights", b"modified weights"
    path.write_bytes(original)
    cache = {}
    counts = observe_hash_reads(monkeypatch)
    assert (
        checkpoint_extras._asset_digest(path, cache=cache)
        == hashlib.sha256(original).hexdigest()
    )
    assert (
        checkpoint_extras._asset_digest(path, cache=cache)
        == hashlib.sha256(original).hexdigest()
    )
    assert sum(counts.values()) == len(original)
    previous = path.stat()
    if change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(modified)
        os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        os.replace(replacement, path)
    else:
        path.write_bytes(modified)
        if change == "same_size_restored_mtime":
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert checkpoint_extras._asset_stat_key(
        path.stat()
    ) != checkpoint_extras._asset_stat_key(previous)
    assert (
        checkpoint_extras._asset_digest(path, cache=cache)
        == hashlib.sha256(modified).hexdigest()
    )
    assert sum(counts.values()) == len(original) + len(modified)
    assert len(cache) == 1


def test_source_change_during_hash_is_rejected_without_caching(tmp_path, monkeypatch):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"before")
    cache = {}
    observe_hash_reads(
        monkeypatch,
        changed_path=source,
        after_read=lambda: source.write_bytes(b"after!"),
    )
    with pytest.raises(RuntimeError, match="changed during checkpoint saving"):
        checkpoint_extras._asset_digest(source, cache=cache)
    assert not cache


@pytest.mark.parametrize("changed", ["source", "copy"])
def test_initial_copy_rejects_changed_source_or_corrupted_copy(
    tmp_path, monkeypatch, changed
):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"original")
    copy2 = checkpoint_extras.shutil.copy2

    def change_after_copy(src, dst, *args, **kwargs):
        result = copy2(src, dst, *args, **kwargs)
        if changed == "copy":
            Path(dst).write_bytes(b"corrupt!")
        elif Path(src) == source:
            source.write_bytes(b"modified")
        return result

    monkeypatch.setattr(checkpoint_extras.shutil, "copy2", change_after_copy)
    destination, pool = tmp_path / "snapshot/weights.bin", tmp_path / "pool"
    with pytest.raises(RuntimeError, match="changed during checkpoint saving"):
        checkpoint_extras._copy_checkpoint_asset(
            source, destination, pool, digest_cache={}
        )
    assert not destination.exists()
    assert not list(pool.iterdir())


def test_cached_source_change_during_link_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"original")
    cache, pool = {}, tmp_path / "pool"
    checkpoint_extras._copy_checkpoint_asset(
        source, tmp_path / "first/weights.bin", pool, digest_cache=cache
    )
    destination = tmp_path / "second/weights.bin"
    link = os.link

    def change_during_link(src, dst, *args, **kwargs):
        result = link(src, dst, *args, **kwargs)
        if Path(dst) == destination:
            source.write_bytes(b"modified")
        return result

    monkeypatch.setattr(os, "link", change_during_link)
    with pytest.raises(RuntimeError, match="changed during checkpoint saving"):
        checkpoint_extras._copy_checkpoint_asset(
            source, destination, pool, digest_cache=cache
        )
    assert not destination.exists()


@pytest.mark.parametrize("damage", ["size", "writable"])
def test_existing_blob_size_and_readonly_guards_remain(tmp_path, damage):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"original")
    cache, pool = {}, tmp_path / "pool"
    checkpoint_extras._copy_checkpoint_asset(
        source, tmp_path / "first/weights.bin", pool, digest_cache=cache
    )
    blob = pool / hashlib.sha256(b"original").hexdigest()
    blob.chmod(0o644)
    if damage == "size":
        blob.write_bytes(b"short")
        blob.chmod(0o444)
    with pytest.raises(ValueError, match="damaged blob"):
        checkpoint_extras._copy_checkpoint_asset(
            source, tmp_path / "second/weights.bin", pool, digest_cache=cache
        )


@pytest.mark.parametrize("prefix", ["relative/../outputs", "~/outputs"])
@pytest.mark.parametrize("source", ["explicit", "output_dir", "section", "wandb_dir"])
def test_training_and_resume_select_the_same_literal_path(
    tmp_path, monkeypatch, prefix, source
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "different-home"))
    trainer = trainer_for_paths()
    if source == "explicit":
        trainer.args.preference_replay_dir = prefix
        trainer.wandb_config = {"output_dir": "ignored"}
        expected = (tmp_path / prefix).resolve()
    else:
        trainer.wandb_config = {
            "output_dir": {"output_dir": prefix},
            "section": {"config_sections": {"output": {"base_dir": prefix}}},
            "wandb_dir": {"dir": prefix},
        }[source]
        expected = (tmp_path / prefix / "preference_replay").resolve()
    before = dict(vars(trainer))
    resumed = _resume_output_path(
        trainer,
        "preference_replay",
        "_preference_replay_dir_path",
        "preference_replay_dir",
    )
    assert resumed == expected
    assert vars(trainer) == before
    assert list(tmp_path.iterdir()) == []
    assert trainer._preference_replay_dir() == str(expected)
    assert expected.is_dir()
    assert not (tmp_path / "different-home").exists()


@pytest.mark.parametrize("cached", ["~/cached", Path("relative-cached")])
def test_cached_path_is_verbatim_and_wins_over_every_other_source(
    tmp_path, monkeypatch, cached
):
    monkeypatch.chdir(tmp_path)
    trainer = trainer_for_paths()
    trainer._preference_replay_dir_path = cached
    trainer.args.preference_replay_dir = "ignored-explicit"
    trainer.args.policy_checkpoint_dir = "ignored-policy"
    trainer.wandb_config = {"output_dir": "ignored-wandb"}
    trainer._checkpoint_directory = "ignored-checkpoint"
    assert (
        resolve_output_path(
            "preference_replay",
            cached=cached,
            configured="ignored",
            wandb_config={"output_dir": "ignored"},
            default_base="ignored",
        )
        is cached
    )
    assert trainer._preference_replay_dir() is cached
    assert (
        _resume_output_path(
            trainer,
            "preference_replay",
            "_preference_replay_dir_path",
            "preference_replay_dir",
        )
        == (tmp_path / cached).resolve()
    )
    del trainer.args
    assert trainer._preference_replay_dir() is cached
    assert (
        _resume_output_path(
            trainer,
            "preference_replay",
            "_preference_replay_dir_path",
            "preference_replay_dir",
        )
        == (tmp_path / cached).resolve()
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "checkpoint,replay,policy,expected",
    [
        ("~/checkpoints", "other/replay", "other/policy", "home/checkpoints"),
        (None, "~/run/replay", "~/run/policy", "~/run"),
        (None, "single/replay", None, "single"),
        (None, "first/replay", "second/policy", "."),
        (None, None, None, "."),
    ],
)
def test_resume_additional_fallbacks_preserve_their_source_conventions(
    tmp_path, monkeypatch, checkpoint, replay, policy, expected
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    trainer = trainer_for_paths()
    trainer._checkpoint_directory = checkpoint
    trainer.args.preference_replay_dir = replay
    trainer.args.policy_checkpoint_dir = policy
    before = dict(vars(trainer))
    path = _resume_output_path(
        trainer, "reward_distributions", "_reward_distribution_dir_path"
    )
    assert path == (tmp_path / expected / "reward_distributions").resolve()
    assert vars(trainer) == before
    assert list(tmp_path.iterdir()) == []
    # Checkpoint-specific fallback does not affect ordinary training.
    assert trainer._reward_distribution_dir() == str(tmp_path / "reward_distributions")


def test_checkpoint_tilde_fallback_matches_actual_checkpoint_save(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    trainer = TinyTrainer()
    with trainer._checkpoint_context(checkpoint_dir="~/checkpoints"):
        destination = _resume_output_path(
            trainer, "reward_distributions", "_reward_distribution_dir_path"
        )
        assert destination == tmp_path / "home/checkpoints/reward_distributions"
        assert not destination.parent.exists()
        trainer._checkpoint_commit({"epoch": 1})
        checkpoint = destination.parent / "checkpoint-1" / STATE_FILENAME
        assert checkpoint.is_file()
        assert not destination.exists()
    assert not (tmp_path / "~").exists()


@pytest.mark.parametrize(
    "wandb_config,expected",
    [
        (
            {
                "output_dir": "first",
                "config_sections": {"output": {"base_dir": "second"}},
                "dir": "third",
            },
            "first",
        ),
        (
            {"config_sections": {"output": {"base_dir": "second"}}, "dir": "third"},
            "second",
        ),
        ({"config_sections": ["ignored"], "dir": "third"}, "third"),
        ({"config_sections": {"output": []}, "dir": "third"}, "third"),
    ],
)
def test_wandb_source_priority_precedes_resume_fallback(
    tmp_path, monkeypatch, wandb_config, expected
):
    monkeypatch.chdir(tmp_path)
    trainer = trainer_for_paths()
    trainer.wandb_config = wandb_config
    trainer._checkpoint_directory = "ignored-checkpoint"
    trainer.args.policy_checkpoint_dir = "ignored-parent/policy"
    assert (
        _resume_output_path(
            trainer, "reward_distributions", "_reward_distribution_dir_path"
        )
        == tmp_path / expected / "reward_distributions"
    )
    assert list(tmp_path.iterdir()) == []


def test_resume_preflight_keeps_symlink_boundary_check_without_creating_outputs(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "source-checkpoint"
    checkpoint.mkdir()
    (tmp_path / "alias").symlink_to(checkpoint, target_is_directory=True)
    trainer = trainer_for_paths()
    trainer.args.preference_replay_dir = "alias/nested"
    before = dict(vars(trainer))
    with pytest.raises(ValueError, match="outside the source checkpoint"):
        _preference_resume_destinations(trainer, checkpoint)
    assert vars(trainer) == before
    assert list(checkpoint.iterdir()) == []
    assert not (tmp_path / "policy_checkpoints").exists()
