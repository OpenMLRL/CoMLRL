import copy
import gc
import inspect
import json
import numbers
import os
import random
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM

from comlrl.schedulers import DeviceScheduler
from comlrl.utils.distributed import unwrap_model
from comlrl.utils.model_loading import (
    freeze_model,
    resolve_model_kwargs,
    resolve_model_sources,
)
from comlrl.utils.reward_utils import call_reward_function, resolve_reward_range
from comlrl.utils.tokenizer_utils import apply_tokenizer_specials

from .madpo import (
    AgentPreferenceTensors,
    MADPOConfig,
    MADPOTrainer,
    PreferencePair,
)
from .centralized import CentralizedComparatorAdapter
from .marlhf import MARLHFConfig, MARLHFTrainer
from .output_paths import resolve_output_path


_REWARD_DISTRIBUTION_BINS = 16


def _normalize_comparator_policy(policy: Optional[str]) -> str:
    mode = str(policy or "current").strip().lower()
    if mode in {"current", "self", "online"}:
        return "current"
    if mode in {
        "current_copy",
        "copy",
        "online_copy",
        "current-offload",
        "current_offload",
    }:
        return "current_copy"
    if mode in {"model", "external", "comparator", "reference", "ref"}:
        return "model"
    if mode in {"history", "checkpoint", "previous", "previous_iteration"}:
        return "history"
    if mode in {"api", "http", "endpoint"}:
        return "api"
    raise ValueError(
        "comparator_policy must be one of: current, current_copy, model, history, api."
    )


def _normalize_comparator_generation_mode(mode: Optional[str]) -> str:
    value = str(mode or "decentralized").strip().lower()
    if value in {"decentralized", "decentralised", "multi_agent", "multi-agent"}:
        return "decentralized"
    if value in {"centralized", "centralised", "single_agent", "single-agent"}:
        return "centralized"
    raise ValueError(
        "comparator_generation_mode must be one of: decentralized, centralized."
    )


def _normalize_preference_replay_mode(mode: Optional[str]) -> str:
    value = str(mode or "current").strip().lower()
    if value in {"current", "latest", "new"}:
        return "current"
    if value in {"nearest_k", "recent_k", "last_k", "k"}:
        return "nearest_k"
    if value == "all_history":
        return "all_history"
    if value in {"lambda", "lambda_decay", "td_lambda", "td-lambda"}:
        return "lambda_decay"
    raise ValueError(
        "preference_replay_mode must be one of: current, nearest_k, all_history, lambda_decay."
    )


def _normalize_preference_scoring_reward(mode: Optional[str]) -> str:
    value = str(mode or "task").strip().lower()
    if value in {"task", "oracle", "environment", "env"}:
        return "task"
    if value in {"reward_model", "model", "learned"}:
        return "reward_model"
    raise ValueError("preference_scoring_reward must be one of: task, reward_model.")


def _validate_iterative_config(args: Any) -> None:
    if int(args.num_iterations) < 1:
        raise ValueError("num_iterations must be >= 1.")
    args.preference_replay_mode = _normalize_preference_replay_mode(
        args.preference_replay_mode
    )
    _validate_preference_replay_args(args)
    args.comparator_policy = _normalize_comparator_policy(args.comparator_policy)
    args.comparator_generation_mode = _normalize_comparator_generation_mode(
        getattr(args, "comparator_generation_mode", "decentralized")
    )
    if args.comparator_generation_mode == "centralized":
        if int(args.comparator_centralized_agent_index) < 0:
            raise ValueError("comparator_centralized_agent_index must be >= 0.")
        if int(args.comparator_centralized_agent_index) >= int(args.num_agents):
            raise ValueError(
                "comparator_centralized_agent_index must be smaller than num_agents."
            )
        args.comparator_centralized_agent_index = int(
            args.comparator_centralized_agent_index
        )
    if args.comparator_num_candidates is not None:
        if int(args.comparator_num_candidates) < 1:
            raise ValueError("comparator_num_candidates must be >= 1 or null.")
        args.comparator_num_candidates = int(args.comparator_num_candidates)
    if getattr(args, "comparator_api_max_n_per_request", None) is not None:
        if int(args.comparator_api_max_n_per_request) < 1:
            raise ValueError("comparator_api_max_n_per_request must be >= 1 or null.")
        args.comparator_api_max_n_per_request = int(
            args.comparator_api_max_n_per_request
        )
    if args.comparator_policy == "model":
        if args.comparator_model_name is None and args.comparator_agents is None:
            raise ValueError(
                "comparator_model_name or comparator_agents is required when "
                "comparator_policy='model'."
            )
        if args.comparator_history_k is not None:
            raise ValueError(
                "comparator_history_k is only valid when "
                "comparator_policy='history'."
            )
    elif args.comparator_policy == "history":
        if args.comparator_history_k is None:
            args.comparator_history_k = 1
        if int(args.comparator_history_k) < 1:
            raise ValueError("comparator_history_k must be >= 1.")
        args.comparator_history_k = int(args.comparator_history_k)
        if args.comparator_model_name is not None or args.comparator_agents is not None:
            raise ValueError(
                "comparator_model_name and comparator_agents are only valid when "
                "comparator_policy='model'."
            )
    elif args.comparator_policy in {"current", "current_copy"}:
        if args.comparator_history_k is not None:
            raise ValueError(
                "comparator_history_k is only valid when "
                "comparator_policy='history'."
            )
        if args.comparator_model_name is not None or args.comparator_agents is not None:
            raise ValueError(
                "comparator_model_name and comparator_agents are only valid when "
                "comparator_policy='model'."
            )
    else:
        if args.comparator_history_k is not None:
            raise ValueError(
                "comparator_history_k is only valid when "
                "comparator_policy='history'."
            )
    if args.comparator_policy == "api" and not args.comparator_api_url:
        raise ValueError("comparator_api_url is required when comparator_policy='api'.")


def _validate_preference_replay_args(args: Any) -> None:
    mode = args.preference_replay_mode
    replay_k = args.preference_replay_k
    replay_lambda = args.preference_replay_lambda
    sample_size = args.preference_replay_sample_size

    if mode == "current":
        if replay_k not in (None, 1):
            raise ValueError(
                "preference_replay_k must be null or 1 when "
                "preference_replay_mode='current'."
            )
        if replay_lambda is not None:
            raise ValueError(
                "preference_replay_lambda is only valid when "
                "preference_replay_mode='lambda_decay'."
            )
        if sample_size is not None:
            raise ValueError(
                "preference_replay_sample_size is only valid when "
                "preference_replay_mode is 'nearest_k', 'all_history', "
                "or 'lambda_decay'."
            )
        return

    if mode == "nearest_k":
        if replay_k is None:
            raise ValueError(
                "preference_replay_k is required when "
                "preference_replay_mode='nearest_k'."
            )
        if int(replay_k) < 1:
            raise ValueError("preference_replay_k must be >= 1.")
        args.preference_replay_k = int(replay_k)
        if replay_lambda is not None:
            raise ValueError(
                "preference_replay_lambda is only valid when "
                "preference_replay_mode='lambda_decay'."
            )
        if sample_size is not None:
            if int(sample_size) < 1:
                raise ValueError("preference_replay_sample_size must be >= 1 or null.")
            args.preference_replay_sample_size = int(sample_size)
        return

    if mode == "all_history":
        if replay_k is not None:
            raise ValueError(
                "preference_replay_k is only valid when "
                "preference_replay_mode='nearest_k'."
            )
        if replay_lambda is not None:
            raise ValueError(
                "preference_replay_lambda is only valid when "
                "preference_replay_mode='lambda_decay'."
            )
        if sample_size is not None:
            if int(sample_size) < 1:
                raise ValueError("preference_replay_sample_size must be >= 1 or null.")
            args.preference_replay_sample_size = int(sample_size)
        return

    if mode == "lambda_decay":
        if replay_k is not None:
            raise ValueError(
                "preference_replay_k is only valid when "
                "preference_replay_mode='nearest_k'."
            )
        if replay_lambda is None:
            raise ValueError(
                "preference_replay_lambda is required when "
                "preference_replay_mode='lambda_decay'."
            )
        replay_lambda = float(replay_lambda)
        if replay_lambda < 0.0 or replay_lambda > 1.0:
            raise ValueError("preference_replay_lambda must be in [0, 1].")
        args.preference_replay_lambda = replay_lambda
        if sample_size is not None:
            if int(sample_size) < 1:
                raise ValueError("preference_replay_sample_size must be >= 1 or null.")
            args.preference_replay_sample_size = int(sample_size)
        return

    raise ValueError(f"Unsupported preference_replay_mode: {mode}")


@dataclass
class _IterativePreferenceConfig:
    """Preference-refresh, replay and comparator settings shared by iterative trainers."""

    num_iterations: int = 6
    preference_replay_mode: str = "current"
    preference_replay_k: Optional[int] = None
    preference_replay_lambda: Optional[float] = None
    preference_replay_sample_size: Optional[int] = None
    preference_replay_dir: Optional[str] = None
    log_reward_distribution: bool = False
    policy_checkpoint_dir: Optional[str] = None
    comparator_policy: str = "current"
    comparator_generation_mode: str = "decentralized"
    comparator_centralized_agent_index: int = 0
    comparator_model_name: Optional[str] = None
    comparator_agents: Optional[Sequence[str]] = None
    comparator_devices: Optional[Union[str, Sequence[str]]] = None
    comparator_num_candidates: Optional[int] = None
    comparator_history_k: Optional[int] = None
    comparator_api_url: Optional[str] = None
    comparator_api_format: str = "generic"
    comparator_api_model: Optional[str] = None
    comparator_api_max_n_per_request: Optional[int] = None
    comparator_api_timeout: float = 120.0
    comparator_api_headers: Optional[Dict[str, str]] = None
    comparator_api_key: Optional[str] = None
    comparator_api_key_env: Optional[str] = None
    comparator_api_key_header: str = "Authorization"
    comparator_api_key_prefix: str = "Bearer"
    comparator_api_response_field: str = "completions"
    comparator_api_extra_body: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_iterative_config(self)

    def _allowed_pair_selection_modes(self) -> Tuple[str, ...]:
        return ("reward_gap", "all", "random", "comparator_reward")


@dataclass
class MADPOIterConfig(_IterativePreferenceConfig, MADPOConfig):
    """Configuration for iterative MADPO preference refresh."""

    num_train_epochs: int = 1
    preference_num_candidates: int = 20
    preference_pairs_per_sample: Optional[int] = 4
    pair_selection: str = "comparator_reward"


@dataclass
class MARLHFIterConfig(_IterativePreferenceConfig, MARLHFConfig):
    """Configuration for iterative MARLHF preference and reward refresh."""

    num_train_epochs: int = 2
    preference_num_candidates: int = 20
    preference_pairs_per_sample: Optional[int] = 4
    pair_selection: str = "comparator_reward"
    reward_num_train_epochs: int = 2
    preference_scoring_reward: str = "task"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.preference_scoring_reward = _normalize_preference_scoring_reward(
            self.preference_scoring_reward
        )


@dataclass
class PreferenceReplayShard:
    iteration: int
    path: str
    num_pairs: int


class MADPOIterTrainer(MADPOTrainer):
    """
    Iterative MADPO.

    Each iteration refreshes the preference dataset by comparing completions from
    the current policy against a comparator policy, then runs DPO updates on the
    newly generated pairs.
    """

    default_config_cls = MADPOIterConfig
    algorithm_name = "MADPOIter"

    def _checkpoint_validate_progress(self, state):
        from .checkpointing import validate_preference_progress

        validate_preference_progress(self, state, "madpo_iter")

    def __init__(
        self,
        *args,
        centralized_comparator_adapter: Optional[CentralizedComparatorAdapter] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            centralized_comparator_adapter=centralized_comparator_adapter,
            **kwargs,
        )
        self._reward_distribution_range: Optional[Tuple[float, float]] = None
        if self._log_reward_distribution_enabled():
            self._reward_distribution_range = self._resolve_reward_distribution_range()
        self._initialize_comparator_rng()

    def _initialize_comparator_rng(self) -> None:
        target_seed = int(torch.initial_seed())
        if target_seed >= 1 << 63:
            target_seed -= 1 << 64
        self.target_seed = target_seed
        self.comparator_seed = -target_seed if target_seed != 0 else 1
        self._comparator_rng_states: Dict[str, torch.Tensor] = {}
        self._comparator_rng_streams: Optional[Dict[str, Dict[str, str]]] = None
        self._comparator_rng_locks: Dict[str, Any] = {}
        self._comparator_rng_registry_lock = threading.Lock()

    def _checkpoint_comparator_devices(self) -> Dict[str, torch.device]:
        """Resolve comparator slots without loading transient policy models."""
        if getattr(self.args, "comparator_policy", "current") == "api":
            return {}
        indices = range(self.num_agents)
        if (
            getattr(self.args, "comparator_generation_mode", "decentralized")
            == "centralized"
        ):
            indices = [self._centralized_comparator_agent_index()]
        if getattr(self.args, "comparator_policy", "current") == "current":
            devices = [
                next(unwrap_model(agent).parameters()).device for agent in self.agents
            ]
        else:
            devices = self._resolve_comparator_devices()
        result = {}
        for index in indices:
            device = torch.device(devices[index])
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            elif device.type == "cpu":
                device = torch.device("cpu")
            result[str(index)] = device
        return result

    def _checkpoint_validate_comparator_rng(self, states, streams) -> None:
        """Reject incompatible streams before checkpoint restoration mutates state."""
        devices = self._checkpoint_comparator_devices()
        if any(device.type not in {"cpu", "cuda"} for device in devices.values()):
            raise ValueError("Comparator RNG checkpoints require CPU or CUDA.")
        if not isinstance(states, dict):
            raise ValueError("Invalid comparator RNG states.")
        if streams is None:
            expected = {str(device) for device in devices.values()}
            if states and set(states) != expected:
                raise ValueError(
                    "Legacy comparator RNG device groups differ from this run. "
                    "Resume on the original devices and save a new checkpoint "
                    "before changing comparator devices."
                )
            state_devices = {str(device): device for device in devices.values()}
        else:
            if not isinstance(streams, dict) or set(streams) != set(devices):
                raise ValueError("Comparator RNG stream slots do not match this run.")
            state_devices = {}
            for index, record in streams.items():
                if (
                    not isinstance(record, dict)
                    or set(record) != {"id", "backend"}
                    or not isinstance(record["id"], str)
                    or not record["id"].startswith("comparator-")
                    or not record["id"][len("comparator-") :].isdigit()
                    or not isinstance(record["backend"], str)
                    or record["backend"] not in {"cpu", "cuda"}
                ):
                    raise ValueError("Invalid comparator RNG stream metadata.")
                device = devices[index]
                if device.type != record["backend"]:
                    raise ValueError(
                        "Comparator RNG backend changed; CPU and CUDA states "
                        "cannot be resumed across backends."
                    )
                previous = state_devices.setdefault(record["id"], device)
                if previous.type != device.type:
                    raise ValueError(
                        "A comparator RNG stream has conflicting backends."
                    )
            if set(states) != set(state_devices):
                raise ValueError("Comparator RNG states do not match logical streams.")
        for key, state in states.items():
            device = state_devices[key]
            if device.type not in {"cpu", "cuda"}:
                raise ValueError("Comparator RNG checkpoints require CPU or CUDA.")
            if (
                not isinstance(state, torch.Tensor)
                or state.dtype != torch.uint8
                or state.ndim != 1
            ):
                raise ValueError("Invalid comparator RNG state tensor.")
            try:
                torch.Generator(device=device).set_state(state.cpu())
            except (RuntimeError, TypeError) as error:
                raise ValueError(
                    f"Comparator RNG state is incompatible with backend {device.type}."
                ) from error

    def _checkpoint_prepare_training(self) -> None:
        """Give checkpointed comparator streams stable IDs, preserving sharing."""
        states = self._comparator_rng_states
        streams = getattr(self, "_comparator_rng_streams", None)
        self._checkpoint_validate_comparator_rng(states, streams)
        if streams is not None:
            return
        groups = {}
        streams = {}
        portable_states = {}
        for index, device in self._checkpoint_comparator_devices().items():
            device_key = str(device)
            stream = groups.setdefault(device_key, f"comparator-{index}")
            streams[index] = {"id": stream, "backend": device.type}
            if stream not in portable_states:
                state = states.get(device_key)
                if state is None:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(self.comparator_seed)
                    state = generator.get_state()
                portable_states[stream] = state.clone()
        self._comparator_rng_states = portable_states
        self._comparator_rng_streams = streams
        self._comparator_rng_locks = {}

    def _comparator_stream_index(self, agent: Any, device: torch.device) -> str:
        """Keep the one-argument private RNG API usable for identifiable policies."""
        streams = self._comparator_rng_streams
        matches = []
        agent_module = unwrap_model(agent)
        for name in ("agents", "_comparator_agents", "_current_copy_comparator_agents"):
            for index, candidate in enumerate(getattr(self, name, None) or []):
                if (
                    candidate is not None
                    and unwrap_model(candidate) is agent_module
                    and str(index) in streams
                ):
                    matches.append(str(index))
        if (
            not matches
            and getattr(self.args, "comparator_generation_mode", "decentralized")
            == "centralized"
        ):
            matches = [str(self._centralized_comparator_agent_index())]
        if not matches:
            matches = [
                index
                for index, target in self._checkpoint_comparator_devices().items()
                if target == device
            ]
        if matches and len({streams[index]["id"] for index in matches}) == 1:
            return matches[0]
        raise ValueError(
            "Comparator RNG slot is ambiguous; pass agent_idx when using a "
            "transient policy with multiple logical streams on the same device."
        )

    @contextmanager
    def _comparator_rng(
        self, agent: Any, agent_idx: Optional[int] = None
    ) -> Iterator[None]:
        """Use a persistent comparator-only RNG stream on the agent device."""
        agent_module = unwrap_model(agent)
        try:
            device = next(agent_module.parameters()).device
        except StopIteration:
            yield
            return

        device = torch.device(device)
        device_key = str(device)
        streams = getattr(self, "_comparator_rng_streams", None)
        state_key = device_key
        lock_keys = [device_key]
        if streams is not None:
            index = (
                str(agent_idx)
                if agent_idx is not None
                else self._comparator_stream_index(agent, device)
            )
            if index not in streams:
                raise ValueError(f"Comparator RNG slot {index} is not configured.")
            record = streams[index]
            if record["backend"] != device.type:
                raise ValueError(
                    "Comparator RNG backend changed; CPU/CUDA streams are incompatible."
                )
            state_key = record["id"]
            lock_keys = ["stream:" + state_key, "device:" + device_key]
        with self._comparator_rng_registry_lock:
            locks = [
                self._comparator_rng_locks.setdefault(key, threading.RLock())
                for key in lock_keys
            ]

        with ExitStack() as stack:
            for lock in locks:
                stack.enter_context(lock)
            comparator_state = self._comparator_rng_states.get(state_key)
            if comparator_state is None:
                if streams is not None:
                    raise ValueError(
                        "Missing logical comparator RNG state; refusing to reseed."
                    )
                generator = torch.Generator(device=device)
                generator.manual_seed(self.comparator_seed)
                comparator_state = generator.get_state()

            if device.type == "cuda":
                original_state = torch.cuda.get_rng_state(device)
                torch.cuda.set_rng_state(comparator_state, device)
                try:
                    yield
                finally:
                    self._comparator_rng_states[state_key] = torch.cuda.get_rng_state(
                        device
                    )
                    torch.cuda.set_rng_state(original_state, device)
                return

            if device.type == "cpu":
                original_state = torch.random.get_rng_state()
                torch.random.set_rng_state(comparator_state)
                try:
                    yield
                finally:
                    self._comparator_rng_states[state_key] = (
                        torch.random.get_rng_state()
                    )
                    torch.random.set_rng_state(original_state)
                return

            raise ValueError(
                "Independent comparator RNG currently supports CPU and CUDA devices; "
                f"got {device}."
            )

    def train(
        self,
        resume_from_checkpoint=None,
        checkpoint_dir=None,
        checkpoint_interval=1,
        checkpoint_total_limit=None,
        **kwargs,
    ):
        if (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, numbers.Integral)
            or checkpoint_interval < 1
        ):
            raise ValueError("checkpoint_interval must be a positive integer.")
        if (
            resume_from_checkpoint is not None
            or checkpoint_dir is not None
            or checkpoint_interval != 1
            or checkpoint_total_limit is not None
        ):
            from .checkpointing import train_iterations_checkpointed

            with self._checkpoint_context(
                resume_from_checkpoint=resume_from_checkpoint,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                checkpoint_total_limit=checkpoint_total_limit,
            ):
                return train_iterations_checkpointed(self, reward_model=False, **kwargs)

        self._checkpoint_untracked_preference = True
        if int(self.args.num_turns) != 1:
            raise ValueError("MADPOIter currently supports num_turns=1 only.")

        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        for agent_idx, agent in enumerate(self.agents):
            agent.to(self.agent_devices[agent_idx])
            agent.train()

        self._maybe_save_initial_policy_checkpoint()
        updates_seen = 0
        total_pairs = 0
        for iteration_idx in range(int(self.args.num_iterations)):
            self._reset_iteration_reward_distribution()
            self._prepare_iteration_current_copy_comparator()
            try:
                preference_pairs = self._build_preference_dataset(
                    iteration_idx=iteration_idx,
                    **kwargs,
                )
            finally:
                self._clear_iteration_current_copy_comparator()
            current_pair_count = len(preference_pairs)
            train_pairs = self._select_iteration_preference_pairs(
                preference_pairs,
                iteration_idx=iteration_idx,
            )
            total_pairs += len(train_pairs)
            self._log_iteration_replay(
                iteration_idx,
                train_pairs=train_pairs,
                current_pair_count=current_pair_count,
                train_pair_count=len(train_pairs),
            )
            preference_pairs.clear()
            if not train_pairs:
                if self.verbose:
                    print(
                        "MADPOIter: no replay preference pairs were available "
                        f"for iteration {iteration_idx + 1} "
                        f"(current generated {current_pair_count})."
                    )
                self._maybe_save_iteration_policy_checkpoint(iteration_idx)
                continue

            updates_seen = self._train_preference_pairs(
                train_pairs,
                updates_seen=updates_seen,
                desc=(
                    f"MADPOIter iteration {iteration_idx + 1}/"
                    f"{int(self.args.num_iterations)}"
                ),
            )
            self._maybe_save_iteration_policy_checkpoint(iteration_idx)

        if total_pairs == 0 and self.verbose:
            print("MADPOIter: no non-tied preference pairs were generated.")

    def _select_iteration_preference_pairs(
        self,
        current_pairs: List[PreferencePair],
        *,
        iteration_idx: int,
    ) -> List[PreferencePair]:
        shard = self._write_iteration_preference_pairs(iteration_idx, current_pairs)
        shards = self._preference_replay_shards()
        if shard is not None:
            shards.append(shard)

        mode = self.args.preference_replay_mode
        if mode == "current":
            return list(current_pairs)
        if mode == "nearest_k":
            k = int(self.args.preference_replay_k)
            sample_size = self._replay_sample_size(len(current_pairs), shards)
            return self._sample_replay_preferences(shards[-k:], sample_size)
        if mode == "all_history":
            sample_size = self._replay_sample_size(len(current_pairs), shards)
            return self._sample_replay_preferences(shards, sample_size)
        if mode == "lambda_decay":
            sample_size = self._replay_sample_size(len(current_pairs), shards)
            weights = self._lambda_decay_weights(len(shards))
            return self._sample_replay_preferences(shards, sample_size, weights=weights)
        raise ValueError(f"Unsupported preference_replay_mode: {mode}")

    def _replay_sample_size(
        self,
        current_pair_count: int,
        shards: Sequence[PreferenceReplayShard],
    ) -> int:
        if self.args.preference_replay_sample_size is not None:
            return int(self.args.preference_replay_sample_size)
        if current_pair_count > 0:
            return int(current_pair_count)
        for shard in reversed(shards):
            if shard.num_pairs > 0:
                return int(shard.num_pairs)
        return 0

    def _log_iteration_replay(
        self,
        iteration_idx: int,
        *,
        train_pairs: Sequence[PreferencePair],
        current_pair_count: int,
        train_pair_count: int,
    ) -> None:
        log_distribution = self._log_reward_distribution_enabled()
        if log_distribution:
            self._write_iteration_reward_distribution(iteration_idx, train_pairs)
        if not (self.wandb_initialized and wandb.run is not None):
            return
        shards = getattr(self, "_preference_replay_shards_state", [])
        metrics: Dict[str, Any] = {
            "iter/current_iteration": float(iteration_idx + 1),
            "iter/current_preference_pairs": float(current_pair_count),
            "iter/total_preference_pairs": float(
                sum(int(shard.num_pairs) for shard in shards)
            ),
            "iter/train_preference_pairs": float(train_pair_count),
        }
        if log_distribution:
            metrics.update(self._iteration_reward_distribution_metrics(iteration_idx))
            metrics.update(
                self._selected_reward_distribution_metrics(
                    iteration_idx,
                    train_pairs,
                )
            )
        wandb.log(metrics, step=int(self.env_step))

    def _log_reward_distribution_enabled(self) -> bool:
        return bool(getattr(self.args, "log_reward_distribution", False))

    def _resolve_reward_distribution_range(self) -> Tuple[float, float]:
        reward_range = resolve_reward_range(self.reward_func)
        if reward_range is None:
            raise ValueError(
                "log_reward_distribution=True requires reward_func.reward_range "
                "to declare the raw reward scale as (minimum, maximum)."
            )
        return reward_range

    def _reward_distribution_bin_edges(self) -> np.ndarray:
        reward_range = getattr(self, "_reward_distribution_range", None)
        if reward_range is None:
            reward_range = self._resolve_reward_distribution_range()
            self._reward_distribution_range = reward_range
        return np.linspace(
            reward_range[0],
            reward_range[1],
            _REWARD_DISTRIBUTION_BINS + 1,
        )

    def _reset_iteration_reward_distribution(self) -> None:
        self._iteration_reward_distribution = {
            "target": [],
            "comparator": [],
        }

    def _record_iteration_reward_distribution(
        self,
        *,
        target_rewards: Sequence[float],
        comparator_rewards: Sequence[float],
    ) -> None:
        if not self._log_reward_distribution_enabled():
            return
        distribution = getattr(self, "_iteration_reward_distribution", None)
        if distribution is None:
            self._reset_iteration_reward_distribution()
            distribution = self._iteration_reward_distribution
        distribution["target"].extend(self._finite_floats(target_rewards))
        distribution["comparator"].extend(self._finite_floats(comparator_rewards))

    @staticmethod
    def _finite_floats(values: Sequence[float]) -> List[float]:
        result: List[float] = []
        for value in values:
            float_value = float(value)
            if np.isfinite(float_value):
                result.append(float_value)
        return result

    def _iteration_reward_distribution_metrics(
        self,
        iteration_idx: int,
    ) -> Dict[str, Any]:
        distribution = getattr(self, "_iteration_reward_distribution", None)
        if not distribution:
            return {}
        return self._reward_distribution_metrics(
            "iter/reward_distribution",
            "Candidate Reward Distribution",
            iteration_idx,
            distribution.get("target", []),
            distribution.get("comparator", []),
        )

    def _selected_reward_distribution_metrics(
        self,
        iteration_idx: int,
        selected_pairs: Sequence[PreferencePair],
    ) -> Dict[str, Any]:
        if not selected_pairs:
            return {}
        target_rewards, comparator_rewards = self._selected_pair_reward_values(
            selected_pairs
        )
        return self._reward_distribution_metrics(
            "iter/selected_reward_distribution",
            "Selected Preference Reward Distribution",
            iteration_idx,
            target_rewards,
            comparator_rewards,
            pair_count=len(selected_pairs),
        )

    def _reward_distribution_metrics(
        self,
        namespace: str,
        title: str,
        iteration_idx: int,
        target_rewards: Sequence[float],
        comparator_rewards: Sequence[float],
        pair_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not target_rewards and not comparator_rewards:
            return {}

        bin_edges = self._reward_distribution_bin_edges()
        target_counts, edges = self._reward_distribution_counts(
            target_rewards,
            bin_edges,
        )
        comparator_counts, _ = self._reward_distribution_counts(
            comparator_rewards,
            bin_edges,
        )
        iteration = int(iteration_idx) + 1
        metrics: Dict[str, Any] = {f"{namespace}/current_iteration": float(iteration)}
        if pair_count is not None:
            metrics[f"{namespace}/pair_count"] = float(pair_count)
        metrics[f"{namespace}/target_sample_count"] = float(len(target_rewards))
        metrics[f"{namespace}/comparator_sample_count"] = float(len(comparator_rewards))
        series = [("target", target_counts), ("comparator", comparator_counts)]
        for kind in ("line", "bar"):
            image = self._reward_distribution_image(
                kind=kind, title=title, edges=edges, series=series
            )
            if image is not None:
                metrics[f"{namespace}/iteration_{iteration:04d}/{kind}_plot"] = image
        if target_rewards:
            metrics[f"{namespace}/target_mean"] = float(np.mean(target_rewards))
        if comparator_rewards:
            metrics[f"{namespace}/comparator_mean"] = float(np.mean(comparator_rewards))
        return metrics

    def _selected_pair_reward_values(
        self,
        selected_pairs: Sequence[PreferencePair],
    ) -> Tuple[List[float], List[float]]:
        target_rewards: List[float] = []
        comparator_rewards: List[float] = []
        for pair in selected_pairs:
            if pair.target_raw_reward is None or pair.comparator_raw_reward is None:
                continue
            target_rewards.extend(self._finite_floats([pair.target_raw_reward]))
            comparator_rewards.extend(self._finite_floats([pair.comparator_raw_reward]))
        return target_rewards, comparator_rewards

    def _write_iteration_reward_distribution(
        self,
        iteration_idx: int,
        selected_pairs: Sequence[PreferencePair],
    ) -> None:
        distribution = getattr(self, "_iteration_reward_distribution", None) or {}
        candidate_target = distribution.get("target", [])
        candidate_comparator = distribution.get("comparator", [])
        selected_target, selected_comparator = self._selected_pair_reward_values(
            selected_pairs
        )
        if (
            not candidate_target
            and not candidate_comparator
            and not selected_target
            and not selected_comparator
        ):
            return

        bin_edges = self._reward_distribution_bin_edges()
        reward_min = float(bin_edges[0])
        reward_max = float(bin_edges[-1])
        iteration = int(iteration_idx) + 1
        payload = {
            "iteration": iteration,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "num_bins": int(_REWARD_DISTRIBUTION_BINS),
            "bin_edges": [float(value) for value in bin_edges.tolist()],
            "bin_centers": [
                float(value)
                for value in ((bin_edges[:-1] + bin_edges[1:]) / 2.0).tolist()
            ],
            "candidate": self._reward_distribution_json_section(
                candidate_target,
                candidate_comparator,
                bin_edges,
            ),
            "selected": self._reward_distribution_json_section(
                selected_target,
                selected_comparator,
                bin_edges,
            ),
        }

        output_dir = self._reward_distribution_dir()
        path = os.path.join(output_dir, f"iteration_{iteration:04d}.json")
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _reward_distribution_json_section(
        self,
        target_rewards: Sequence[float],
        comparator_rewards: Sequence[float],
        bin_edges: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "target": self._reward_distribution_json_series(
                target_rewards,
                bin_edges,
            ),
            "comparator": self._reward_distribution_json_series(
                comparator_rewards,
                bin_edges,
            ),
        }

    def _reward_distribution_json_series(
        self,
        rewards: Sequence[float],
        bin_edges: np.ndarray,
    ) -> Dict[str, Any]:
        finite_rewards = self._finite_floats(rewards)
        counts, _ = self._reward_distribution_counts(finite_rewards, bin_edges)
        return {
            "sample_count": int(len(finite_rewards)),
            "mean": float(np.mean(finite_rewards)) if finite_rewards else None,
            "counts": [int(value) for value in counts.tolist()],
            "raw_rewards": [float(value) for value in finite_rewards],
        }

    @staticmethod
    def _reward_distribution_image(
        *,
        title: str,
        edges: np.ndarray,
        series: Sequence[Tuple[str, np.ndarray]],
        kind: str,
    ) -> Optional[Any]:
        if not series:
            return None
        if kind == "bar":
            series = [(label, counts) for label, counts in series if len(counts) > 0]
            if not series:
                return None

        if kind == "line":
            xs = ((edges[:-1] + edges[1:]) / 2.0).tolist()
        max_count = max(
            [float(np.max(counts)) for _, counts in series if len(counts) > 0] or [1.0]
        )
        max_count = max(max_count, 1.0)

        width, height = 240, 80
        left, right = 29, 10
        top, bottom = 14, 18
        plot_left = left
        plot_right = width - right
        plot_top = top
        plot_bottom = height - bottom

        palette = {
            "target": "#e87030",
            "comparator": "#6e6e6e",
        }

        elements: List[str] = [
            '<div style="max-width:240px;width:100%;overflow:hidden;">',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'style="display:block;max-width:100%;height:auto;">',
            '<rect width="100%" height="100%" fill="white"/>',
            (
                f'<text x="{plot_left}" y="9" font-size="7" '
                f'font-family="Arial, sans-serif" font-weight="700">'
                f"{MADPOIterTrainer._xml_escape(title)}</text>"
            ),
        ]

        x_min = float(edges[0])
        x_range = max(float(edges[-1] - edges[0]), 1e-12)

        def x_position(value: float) -> float:
            return plot_left + ((float(value) - x_min) / x_range) * (
                plot_right - plot_left
            )

        def y_position(value: float) -> float:
            return plot_bottom - (float(value) / max_count) * (plot_bottom - plot_top)

        for x_tick in np.linspace(float(edges[0]), float(edges[-1]), 5):
            x_pos = int(round(x_position(float(x_tick))))
            elements.append(
                f'<line x1="{x_pos}" y1="{plot_top}" x2="{x_pos}" '
                f'y2="{plot_bottom}" stroke="#dddddd" stroke-width="1"/>'
            )
            elements.append(
                f'<text x="{x_pos}" y="{plot_bottom + 8}" font-size="5.5" '
                'font-family="Arial, sans-serif" text-anchor="middle" '
                'fill="#4d4d4d">'
                f"{MADPOIterTrainer._format_axis_tick(float(x_tick))}</text>"
            )

        for y_tick in np.linspace(0.0, max_count, 5):
            y_pos = int(round(y_position(float(y_tick))))
            elements.append(
                f'<line x1="{plot_left}" y1="{y_pos}" x2="{plot_right}" '
                f'y2="{y_pos}" stroke="#dddddd" stroke-width="1"/>'
            )
            elements.append(
                f'<text x="{plot_left - 4}" y="{y_pos + 2}" font-size="6" '
                'font-family="Arial, sans-serif" text-anchor="end" '
                'fill="#4d4d4d">'
                f"{MADPOIterTrainer._format_axis_tick(float(y_tick))}</text>"
            )
        axes = [
            (
                f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" '
                f'y2="{plot_bottom}" stroke="#505050" stroke-width="1"/>'
            ),
            (
                f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" '
                f'y2="{plot_bottom}" stroke="#505050" stroke-width="1"/>'
            ),
            (
                f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 3}" '
                'font-size="5.5" font-family="Arial, sans-serif" '
                'text-anchor="middle">reward</text>'
            ),
            (
                f'<text x="7" y="{(plot_top + plot_bottom) / 2:.1f}" '
                'font-size="5.5" font-family="Arial, sans-serif" '
                'text-anchor="middle" transform="rotate(-90 7 '
                f'{(plot_top + plot_bottom) / 2:.1f})">count</text>'
            ),
        ]

        if kind == "line":
            elements.extend(axes)
            legend_y = plot_top + 7
            for label, counts in series:
                if len(counts) == 0:
                    continue
                color = palette.get(label, "#2878d6")
                points: List[Tuple[int, int]] = []
                for reward_value, count in zip(xs, counts.tolist()):
                    x_pos = x_position(float(reward_value))
                    y_pos = y_position(float(count))
                    points.append((int(round(x_pos)), int(round(y_pos))))

                point_string = " ".join(f"{x_pos},{y_pos}" for x_pos, y_pos in points)
                elements.append(
                    f'<polyline points="{point_string}" fill="none" stroke="{color}" '
                    'stroke-width="1.25" stroke-linejoin="round" stroke-linecap="round"/>'
                )
                for x_pos, y_pos in points:
                    elements.append(
                        f'<circle cx="{x_pos}" cy="{y_pos}" r="1.4" fill="{color}"/>'
                    )

                elements.append(
                    f'<line x1="{plot_right - 52}" y1="{legend_y}" '
                    f'x2="{plot_right - 39}" y2="{legend_y}" stroke="{color}" '
                    'stroke-width="1.25" stroke-linecap="round"/>'
                )
                elements.append(
                    f'<text x="{plot_right - 36}" y="{legend_y + 2}" '
                    'font-size="6" font-family="Arial, sans-serif">'
                    f"{MADPOIterTrainer._xml_escape(label)}</text>"
                )
                legend_y += 8

        else:
            for bin_idx in range(len(edges) - 1):
                bin_left = x_position(float(edges[bin_idx]))
                bin_right = x_position(float(edges[bin_idx + 1]))
                bin_width = max(bin_right - bin_left, 1.0)
                group_left = bin_left + bin_width * 0.12
                group_width = bin_width * 0.76
                bar_width = group_width / max(len(series), 1)
                for series_idx, (label, counts) in enumerate(series):
                    color = palette.get(label, "#2878d6")
                    count = float(counts[bin_idx])
                    bar_x = group_left + series_idx * bar_width
                    bar_y = y_position(count)
                    bar_height = max(plot_bottom - bar_y, 0.0)
                    elements.append(
                        f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" '
                        f'width="{max(bar_width - 0.5, 0.5):.1f}" '
                        f'height="{bar_height:.1f}" fill="{color}" opacity="0.82"/>'
                    )

            elements.extend(axes)
            legend_y = plot_top + 7
            for label, _ in series:
                color = palette.get(label, "#2878d6")
                elements.append(
                    f'<rect x="{plot_right - 52}" y="{legend_y - 4}" '
                    f'width="12" height="5" fill="{color}" opacity="0.82"/>'
                )
                elements.append(
                    f'<text x="{plot_right - 36}" y="{legend_y + 1}" '
                    'font-size="6" font-family="Arial, sans-serif">'
                    f"{MADPOIterTrainer._xml_escape(label)}</text>"
                )
                legend_y += 8

        elements.append("</svg></div>")
        return wandb.Html("".join(elements))

    @staticmethod
    def _format_axis_tick(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        if abs(value) >= 1.0:
            return f"{value:.1f}"
        if abs(value) >= 0.1:
            return f"{value:.2f}"
        return f"{value:.3f}"

    @staticmethod
    def _xml_escape(value: str) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _reward_distribution_counts(
        self,
        rewards: Sequence[float],
        bin_edges: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not rewards:
            return np.zeros(len(bin_edges) - 1, dtype=int), bin_edges
        clipped_rewards = np.clip(
            np.asarray(rewards, dtype=float),
            float(bin_edges[0]),
            float(bin_edges[-1]),
        )
        return np.histogram(clipped_rewards, bins=bin_edges)

    def _sample_replay_preferences(
        self,
        shards: Sequence[PreferenceReplayShard],
        sample_size: int,
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> List[PreferencePair]:
        if sample_size <= 0:
            return []

        eligible_shards: List[PreferenceReplayShard] = []
        shard_weights: List[float] = []
        for idx, shard in enumerate(shards):
            if shard.num_pairs <= 0:
                continue
            eligible_shards.append(shard)
            shard_weights.append(float(weights[idx]) if weights is not None else 1.0)
        if not eligible_shards:
            return []

        shard_counts = [0 for _ in eligible_shards]
        for _ in range(int(sample_size)):
            shard_idx = random.choices(
                range(len(eligible_shards)),
                weights=shard_weights,
                k=1,
            )[0]
            shard_counts[shard_idx] += 1

        sampled: List[PreferencePair] = []
        for shard, count in zip(eligible_shards, shard_counts):
            if count <= 0:
                continue
            records = self._load_replay_records(shard)
            for record in self._sample_replay_records(records, count):
                sampled.append(self._preference_pair_from_record(record))
        random.shuffle(sampled)
        return sampled

    def _preference_replay_shards(self) -> List[PreferenceReplayShard]:
        shards = getattr(self, "_preference_replay_shards_state", None)
        if shards is None:
            shards = []
            self._preference_replay_shards_state = shards
        return shards

    def _preference_replay_dir(self) -> str:
        return self._resolve_output_dir(
            "preference_replay",
            cache_attribute="_preference_replay_dir_path",
            config_attribute="preference_replay_dir",
        )

    def _reward_distribution_dir(self) -> str:
        return self._resolve_output_dir(
            "reward_distributions",
            cache_attribute="_reward_distribution_dir_path",
        )

    def _resolve_output_dir(
        self,
        directory: str,
        *,
        cache_attribute: str,
        config_attribute: Optional[str] = None,
    ) -> str:
        cached = getattr(self, cache_attribute, None)
        if cached:
            return cached

        path = resolve_output_path(
            directory,
            configured=(
                getattr(self.args, config_attribute, None) if config_attribute else None
            ),
            wandb_config=self.wandb_config,
            default_base=os.getcwd(),
        )
        os.makedirs(path, exist_ok=True)
        setattr(self, cache_attribute, path)
        return path

    def _write_iteration_preference_pairs(
        self,
        iteration_idx: int,
        pairs: Sequence[PreferencePair],
    ) -> Optional[PreferenceReplayShard]:
        iteration = int(iteration_idx) + 1
        replay_dir = self._preference_replay_dir()
        path = os.path.join(replay_dir, f"iteration_{iteration:04d}.json")
        payload = {
            "iteration": iteration,
            "num_pairs": len(pairs),
            "pairs": [self._preference_pair_to_record(pair) for pair in pairs],
        }

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp_path, path)

        if not pairs:
            return None
        return PreferenceReplayShard(
            iteration=iteration,
            path=path,
            num_pairs=len(pairs),
        )

    @staticmethod
    def _preference_pair_to_record(pair: PreferencePair) -> Dict[str, Any]:
        return {
            "prompts": list(pair.prompts),
            "winner_completions": list(pair.winner_completions),
            "loser_completions": list(pair.loser_completions),
            "winner_reward": float(pair.winner_reward),
            "loser_reward": float(pair.loser_reward),
            "candidate_reward_mean": float(pair.candidate_reward_mean),
            "raw_rewards": [float(value) for value in (pair.raw_rewards or [])],
            "target_raw_reward": (
                float(pair.target_raw_reward)
                if pair.target_raw_reward is not None
                else None
            ),
            "comparator_raw_reward": (
                float(pair.comparator_raw_reward)
                if pair.comparator_raw_reward is not None
                else None
            ),
        }

    def _load_replay_shard(
        self,
        shard: Optional[PreferenceReplayShard],
    ) -> List[PreferencePair]:
        if shard is None:
            return []
        return [
            self._preference_pair_from_record(record)
            for record in self._load_replay_records(shard)
        ]

    @staticmethod
    def _load_replay_records(shard: PreferenceReplayShard) -> List[Dict[str, Any]]:
        with open(shard.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("pairs", []) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    @staticmethod
    def _sample_replay_records(
        records: Sequence[Dict[str, Any]],
        count: int,
    ) -> List[Dict[str, Any]]:
        if not records or count <= 0:
            return []
        if count <= len(records):
            return random.sample(list(records), k=int(count))

        sampled: List[Dict[str, Any]] = []
        remaining = int(count)
        while remaining > 0:
            if remaining >= len(records):
                batch = list(records)
                random.shuffle(batch)
                sampled.extend(batch)
                remaining -= len(batch)
                continue
            sampled.extend(random.sample(list(records), k=remaining))
            remaining = 0
        return sampled

    def _preference_pair_from_record(self, record: Dict[str, Any]) -> PreferencePair:
        prompts = self._text_list(record.get("prompts"))
        winner_completions = self._text_list(record.get("winner_completions"))
        loser_completions = self._text_list(record.get("loser_completions"))

        agent_tensors = [
            self._preference_tensors_from_text(
                agent_idx,
                prompts[agent_idx],
                winner_completions[agent_idx],
                loser_completions[agent_idx],
            )
            for agent_idx in range(self.num_agents)
        ]
        return PreferencePair(
            prompts=prompts,
            winner_completions=winner_completions,
            loser_completions=loser_completions,
            agent_tensors=agent_tensors,
            winner_reward=float(record.get("winner_reward", 0.0)),
            loser_reward=float(record.get("loser_reward", 0.0)),
            candidate_reward_mean=float(record.get("candidate_reward_mean", 0.0)),
            raw_rewards=[
                float(value)
                for value in (
                    record.get("raw_rewards")
                    if isinstance(record.get("raw_rewards"), list)
                    else []
                )
            ],
            target_raw_reward=(
                float(record["target_raw_reward"])
                if record.get("target_raw_reward") is not None
                else None
            ),
            comparator_raw_reward=(
                float(record["comparator_raw_reward"])
                if record.get("comparator_raw_reward") is not None
                else None
            ),
        )

    def _text_list(self, value: Any) -> List[str]:
        values = list(value) if isinstance(value, list) else []
        texts = [str(item) for item in values[: self.num_agents]]
        while len(texts) < self.num_agents:
            texts.append("")
        return texts

    def _lambda_decay_weights(self, num_datasets: int) -> List[float]:
        if num_datasets < 1:
            return []
        replay_lambda = float(self.args.preference_replay_lambda)
        if replay_lambda == 1.0:
            return [1.0 / num_datasets] * num_datasets

        # TD-lambda style finite geometric weights. Age 0 is the newest dataset.
        newest_first = [
            (1.0 - replay_lambda) * (replay_lambda**age) for age in range(num_datasets)
        ]
        normalizer = sum(newest_first)
        if normalizer <= 0:
            newest_first = [1.0] + [0.0] * (num_datasets - 1)
            normalizer = 1.0
        newest_first = [weight / normalizer for weight in newest_first]
        return list(reversed(newest_first))

    def _policy_checkpoint_dir(self) -> str:
        return self._resolve_output_dir(
            "policy_checkpoints",
            cache_attribute="_policy_checkpoint_dir_path",
            config_attribute="policy_checkpoint_dir",
        )

    def _save_initial_policy_checkpoint(self) -> str:
        if getattr(self, "_initial_policy_checkpoint_saved", False):
            return os.path.join(self._policy_checkpoint_dir(), "initial")
        path = self._save_policy_checkpoint("initial")
        self._initial_policy_checkpoint_saved = True
        return path

    def _save_policy_checkpoint(self, label: str) -> str:
        checkpoint_dir = os.path.join(self._policy_checkpoint_dir(), str(label))
        os.makedirs(checkpoint_dir, exist_ok=True)

        for agent_idx, agent in enumerate(self.agents):
            agent_module = unwrap_model(agent)
            agent_dir = os.path.join(checkpoint_dir, f"agent_{agent_idx}")
            os.makedirs(agent_dir, exist_ok=True)
            agent_module.save_pretrained(agent_dir)
            if self.tokenizers:
                self.tokenizers[agent_idx].save_pretrained(agent_dir)

        return checkpoint_dir

    def _history_policy_checkpoint_path(self, iteration_idx: int) -> str:
        self._save_initial_policy_checkpoint()
        history_k = int(self.args.comparator_history_k)
        target_iteration = int(iteration_idx) - history_k
        if target_iteration < 0:
            return os.path.join(self._policy_checkpoint_dir(), "initial")

        for candidate_iteration in range(target_iteration, -1, -1):
            path = os.path.join(
                self._policy_checkpoint_dir(),
                f"iteration_{candidate_iteration:04d}",
            )
            if self._policy_checkpoint_exists(path):
                return path
        return os.path.join(self._policy_checkpoint_dir(), "initial")

    def _policy_checkpoint_exists(self, checkpoint_dir: str) -> bool:
        return all(
            os.path.isdir(os.path.join(checkpoint_dir, f"agent_{agent_idx}"))
            for agent_idx in range(self.num_agents)
        )

    def _policy_checkpoints_enabled(self) -> bool:
        return bool(getattr(self.args, "policy_checkpoint_dir", None)) or (
            getattr(self.args, "comparator_policy", None) == "history"
        )

    def _maybe_save_initial_policy_checkpoint(self) -> Optional[str]:
        if not self._policy_checkpoints_enabled():
            return None
        return self._save_initial_policy_checkpoint()

    def _maybe_save_iteration_policy_checkpoint(
        self,
        iteration_idx: int,
    ) -> Optional[str]:
        if not self._policy_checkpoints_enabled():
            return None
        return self._save_policy_checkpoint(f"iteration_{int(iteration_idx):04d}")

    def _generate_preference_pairs_for_item(
        self,
        batch_item: Dict[str, Any],
        **kwargs,
    ) -> List[PreferencePair]:
        iteration_idx = int(kwargs.pop("iteration_idx", 0))
        current_candidates = int(self.args.preference_num_candidates)
        comparator_candidates = int(
            self.args.comparator_num_candidates or current_candidates
        )

        current_outputs = self._generate_policy_outputs_for_item(
            self.agents,
            batch_item,
            num_candidates=current_candidates,
            **kwargs,
        )
        comparator_outputs = self._generate_comparator_outputs_for_item(
            batch_item,
            iteration_idx=iteration_idx,
            num_candidates=comparator_candidates,
            **kwargs,
        )

        current_completions = [
            current_outputs[i]["completions"][0] for i in range(self.num_agents)
        ]
        comparator_completions = [
            comparator_outputs[i]["completions"][0] for i in range(self.num_agents)
        ]
        prompts = [current_outputs[i]["prompts"][0] for i in range(self.num_agents)]

        joint_mode = self.args.joint_mode.lower()
        if joint_mode not in {"align", "aligned"}:
            raise ValueError(
                "MADPOIter preference generation currently supports aligned joint_mode only."
            )

        current_raw_rewards, current_rewards = self._compute_raw_and_processed_rewards(
            [prompts[0]],
            current_completions,
            batch_items=[batch_item],
        )
        (
            comparator_raw_rewards,
            comparator_rewards,
        ) = self._compute_raw_and_processed_rewards(
            [prompts[0]],
            comparator_completions,
            batch_items=[batch_item],
        )
        self._record_iteration_reward_distribution(
            target_rewards=current_raw_rewards,
            comparator_rewards=comparator_raw_rewards,
        )
        selected_pairs = self._select_policy_comparison_pairs(
            current_rewards,
            comparator_rewards,
        )
        all_rewards = list(current_rewards) + list(comparator_rewards)
        candidate_reward_mean = float(np.mean(all_rewards)) if all_rewards else 0.0
        result: List[PreferencePair] = []

        for winner, loser in selected_pairs:
            winner_source, winner_idx = winner
            loser_source, loser_idx = loser
            winner_completions = self._completion_group(
                winner_source,
                winner_idx,
                current_completions,
                comparator_completions,
            )
            loser_completions = self._completion_group(
                loser_source,
                loser_idx,
                current_completions,
                comparator_completions,
            )
            winner_reward = self._reward_at(
                winner_source,
                winner_idx,
                current_rewards,
                comparator_rewards,
            )
            loser_reward = self._reward_at(
                loser_source,
                loser_idx,
                current_rewards,
                comparator_rewards,
            )
            winner_raw_reward = self._reward_at(
                winner_source,
                winner_idx,
                current_raw_rewards,
                comparator_raw_rewards,
            )
            loser_raw_reward = self._reward_at(
                loser_source,
                loser_idx,
                current_raw_rewards,
                comparator_raw_rewards,
            )
            target_idx = winner_idx if winner_source == "current" else loser_idx
            comparator_idx = winner_idx if winner_source == "comparator" else loser_idx
            target_raw_reward = current_raw_rewards[target_idx]
            comparator_raw_reward = comparator_raw_rewards[comparator_idx]

            if winner_source == "current" and loser_source == "current":
                agent_tensors = [
                    self._preference_tensors_from_generation(
                        current_outputs[agent_idx],
                        winner_idx,
                        loser_idx,
                    )
                    for agent_idx in range(self.num_agents)
                ]
            else:
                agent_tensors = [
                    self._preference_tensors_from_text(
                        agent_idx,
                        prompts[agent_idx],
                        winner_completions[agent_idx],
                        loser_completions[agent_idx],
                    )
                    for agent_idx in range(self.num_agents)
                ]
            result.append(
                PreferencePair(
                    prompts=list(prompts),
                    winner_completions=winner_completions,
                    loser_completions=loser_completions,
                    agent_tensors=agent_tensors,
                    winner_reward=float(winner_reward),
                    loser_reward=float(loser_reward),
                    candidate_reward_mean=candidate_reward_mean,
                    raw_rewards=[
                        float(winner_raw_reward),
                        float(loser_raw_reward),
                    ],
                    target_raw_reward=float(target_raw_reward),
                    comparator_raw_reward=float(comparator_raw_reward),
                )
            )

        return result

    @staticmethod
    def _preference_tensors_from_generation(
        generation_output: Dict[str, Any],
        winner_idx: int,
        loser_idx: int,
    ) -> AgentPreferenceTensors:
        completion_ids = generation_output["completion_input_ids"][0]
        return AgentPreferenceTensors(
            prompt_input_ids=generation_output["prompt_input_ids"][0].detach().cpu(),
            winner_completion_ids=completion_ids[winner_idx].detach().cpu(),
            loser_completion_ids=completion_ids[loser_idx].detach().cpu(),
        )

    def _compute_raw_and_processed_rewards(
        self,
        prompts: Sequence[str],
        completions_list: List[List[str]],
        *,
        batch_items=None,
    ) -> Tuple[List[float], List[float]]:
        for agent_idx in range(self.num_agents):
            if not isinstance(completions_list[agent_idx], list):
                completions_list[agent_idx] = [completions_list[agent_idx]]

        min_completions = min(len(completions_list[i]) for i in range(self.num_agents))
        learned_scores = self._preference_scoring_raw_and_processed_rewards(
            prompts,
            completions_list,
            batch_items=batch_items,
            min_completions=min_completions,
        )
        if learned_scores is not None:
            return learned_scores

        try:
            reward_signature = inspect.signature(self.reward_func)
        except (TypeError, ValueError):
            reward_signature = None

        raw_rewards: List[float] = []
        processed_rewards: List[float] = []
        for completion_idx in range(min_completions):
            agent_completions = [
                completions_list[agent_idx][completion_idx]
                for agent_idx in range(self.num_agents)
            ]
            completion_args = [[completion] for completion in agent_completions]
            func_rewards = call_reward_function(
                self.reward_func,
                prompts,
                completion_args,
                num_agents=self.num_agents,
                batch_items=batch_items,
                signature=reward_signature,
            )
            processed = [self.reward_processor(reward) for reward in func_rewards]
            raw_rewards.append(float(func_rewards[0] if func_rewards else 0.0))
            processed_rewards.append(float(processed[0] if processed else 0.0))

        return raw_rewards, processed_rewards

    def _preference_scoring_raw_and_processed_rewards(
        self,
        prompts: Sequence[str],
        completions_list: List[List[str]],
        *,
        batch_items=None,
        min_completions: int,
    ) -> Optional[Tuple[List[float], List[float]]]:
        return None

    def _generate_comparator_outputs_for_item(
        self,
        batch_item: Dict[str, Any],
        *,
        iteration_idx: int,
        num_candidates: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if self.args.comparator_generation_mode == "centralized":
            return self._generate_centralized_comparator_outputs_for_item(
                batch_item,
                iteration_idx=iteration_idx,
                num_candidates=num_candidates,
                **kwargs,
            )

        if self.args.comparator_policy == "api":
            return self._generate_api_outputs_for_item(
                batch_item,
                num_candidates=num_candidates,
            )
        if self.args.comparator_policy == "current":
            return self._generate_policy_outputs_for_item(
                self.agents,
                batch_item,
                num_candidates=num_candidates,
                use_comparator_rng=True,
                **kwargs,
            )
        if self.args.comparator_policy == "current_copy":
            return self._generate_current_copy_outputs_for_item(
                batch_item,
                num_candidates=num_candidates,
                **kwargs,
            )
        if self.args.comparator_policy == "history":
            return self._generate_history_policy_outputs_for_item(
                batch_item,
                iteration_idx=iteration_idx,
                num_candidates=num_candidates,
                **kwargs,
            )
        return self._generate_policy_outputs_for_item(
            self._get_comparator_agents(),
            batch_item,
            num_candidates=num_candidates,
            use_comparator_rng=True,
            **kwargs,
        )

    def _generate_policy_outputs_for_item(
        self,
        policy_agents: Sequence[Any],
        batch_item: Dict[str, Any],
        *,
        num_candidates: int,
        use_comparator_rng: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        def _generate_agent(agent_idx: int) -> Dict[str, Any]:
            agent = policy_agents[agent_idx]
            if use_comparator_rng:
                rng_kwargs = (
                    {"agent_idx": agent_idx}
                    if getattr(self, "_comparator_rng_streams", None) is not None
                    else {}
                )
                with self._comparator_rng(agent, **rng_kwargs):
                    return self._generate_completions_with_external_prompts(
                        agent,
                        [batch_item],
                        agent_idx=agent_idx,
                        num_return_sequences=num_candidates,
                        max_new_tokens=self.args.max_new_tokens,
                        **kwargs,
                    )
            return self._generate_completions_with_external_prompts(
                agent,
                [batch_item],
                agent_idx=agent_idx,
                num_return_sequences=num_candidates,
                max_new_tokens=self.args.max_new_tokens,
                **kwargs,
            )

        return self._run_agent_tasks(_generate_agent)

    def _generate_centralized_comparator_outputs_for_item(
        self,
        batch_item: Dict[str, Any],
        *,
        iteration_idx: int,
        num_candidates: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        prompt = self._build_centralized_comparator_prompt(batch_item)
        if self.args.comparator_policy == "api":
            completions = self._call_comparator_api(
                prompt=prompt,
                agent_idx=self._centralized_comparator_agent_index(),
                batch_item=batch_item,
                num_candidates=num_candidates,
            )
            return self._split_centralized_comparator_outputs(
                completions,
                batch_item=batch_item,
                prompt=prompt,
            )
        if self.args.comparator_policy == "current":
            return self._generate_centralized_policy_output_for_agent(
                self.agents[self._centralized_comparator_agent_index()],
                batch_item,
                prompt=prompt,
                num_candidates=num_candidates,
                **kwargs,
            )
        if self.args.comparator_policy == "current_copy":
            return self._generate_centralized_current_copy_outputs_for_item(
                batch_item,
                prompt=prompt,
                num_candidates=num_candidates,
                **kwargs,
            )
        if self.args.comparator_policy == "history":
            checkpoint_dir = self._history_policy_checkpoint_path(iteration_idx)
            agent_idx = self._centralized_comparator_agent_index()
            comparator_agent = self._load_single_policy_checkpoint_agent(
                checkpoint_dir,
                agent_idx,
            )
            try:
                return self._generate_centralized_policy_output_for_agent(
                    comparator_agent,
                    batch_item,
                    prompt=prompt,
                    num_candidates=num_candidates,
                    **kwargs,
                )
            finally:
                del comparator_agent
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return self._generate_centralized_policy_output_for_agent(
            self._get_centralized_comparator_agent(),
            batch_item,
            prompt=prompt,
            num_candidates=num_candidates,
            **kwargs,
        )

    def _generate_centralized_policy_output_for_agent(
        self,
        policy_agent: Any,
        batch_item: Dict[str, Any],
        *,
        prompt: str,
        num_candidates: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        agent_idx = self._centralized_comparator_agent_index()
        rng_kwargs = (
            {"agent_idx": agent_idx}
            if getattr(self, "_comparator_rng_streams", None) is not None
            else {}
        )
        with self._comparator_rng(policy_agent, **rng_kwargs):
            generation_output = self._generate_completions(
                policy_agent,
                [batch_item],
                agent_idx=agent_idx,
                num_return_sequences=num_candidates,
                max_new_tokens=self.args.max_new_tokens,
                prompts_override=[prompt],
                **kwargs,
            )
        completions = generation_output["completions"][0]
        return self._split_centralized_comparator_outputs(
            completions,
            batch_item=batch_item,
            prompt=prompt,
        )

    def _generate_current_copy_outputs_for_item(
        self,
        batch_item: Dict[str, Any],
        *,
        num_candidates: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if self._comparator_devices_match_agent_devices():
            return self._generate_policy_outputs_for_item(
                self.agents,
                batch_item,
                num_candidates=num_candidates,
                use_comparator_rng=True,
                **kwargs,
            )

        cached_agents = getattr(self, "_current_copy_comparator_agents", None)
        if cached_agents is not None:
            return self._generate_policy_outputs_for_item(
                cached_agents,
                batch_item,
                num_candidates=num_candidates,
                use_comparator_rng=True,
                **kwargs,
            )

        comparator_agents = self._clone_current_agents_for_comparator()
        try:
            return self._generate_policy_outputs_for_item(
                comparator_agents,
                batch_item,
                num_candidates=num_candidates,
                use_comparator_rng=True,
                **kwargs,
            )
        finally:
            self._clear_transient_comparator_agents(comparator_agents)

    def _generate_centralized_current_copy_outputs_for_item(
        self,
        batch_item: Dict[str, Any],
        *,
        prompt: str,
        num_candidates: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        agent_idx = self._centralized_comparator_agent_index()
        if self._centralized_comparator_device_matches_agent_device(agent_idx):
            return self._generate_centralized_policy_output_for_agent(
                self.agents[agent_idx],
                batch_item,
                prompt=prompt,
                num_candidates=num_candidates,
                **kwargs,
            )

        cached_agents = getattr(self, "_current_copy_comparator_agents", None)
        if cached_agents is not None and cached_agents[agent_idx] is not None:
            return self._generate_centralized_policy_output_for_agent(
                cached_agents[agent_idx],
                batch_item,
                prompt=prompt,
                num_candidates=num_candidates,
                **kwargs,
            )

        comparator_agent = self._clone_current_agent_for_comparator(agent_idx)
        try:
            return self._generate_centralized_policy_output_for_agent(
                comparator_agent,
                batch_item,
                prompt=prompt,
                num_candidates=num_candidates,
                **kwargs,
            )
        finally:
            transient_agents = [comparator_agent]
            comparator_agent = None
            self._clear_transient_comparator_agents(transient_agents)

    def _centralized_comparator_agent_index(self) -> int:
        return int(getattr(self.args, "comparator_centralized_agent_index", 0))

    def _build_centralized_comparator_prompt(self, batch_item: Dict[str, Any]) -> str:
        if self.centralized_collaboration is not None:
            return self.centralized_collaboration.build_prompt(batch_item)
        agent_prompts = [formatter(batch_item) for formatter in self.formatters]
        prompt = self.centralized_comparator_adapter.build_prompt(
            batch_item,
            agent_prompts,
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "centralized_comparator_adapter.build_prompt must return a "
                "non-empty string."
            )
        return prompt

    def _split_centralized_comparator_outputs(
        self,
        completions: Sequence[str],
        *,
        batch_item: Dict[str, Any],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        if getattr(self, "centralized_collaboration", None) is not None:
            if not completions:
                raise ValueError("Centralized comparator produced no candidates.")
            # Preserve the exact joint text, including tags, for preference learning.
            return [
                {
                    "prompts": [prompt],
                    "batch_items": [batch_item],
                    "completions": [list(completions)],
                }
            ]
        agent_completions: List[List[str]] = [[] for _ in range(int(self.num_agents))]
        for completion in completions:
            parsed_outputs = self.centralized_comparator_adapter.parse_completion(
                str(completion),
                batch_item,
                int(self.num_agents),
            )
            if isinstance(parsed_outputs, (str, bytes)):
                raise TypeError(
                    "centralized_comparator_adapter.parse_completion must return "
                    "a sequence of strings, not a single string."
                )
            parsed = list(parsed_outputs)
            if len(parsed) != int(self.num_agents):
                raise ValueError(
                    "centralized_comparator_adapter.parse_completion must return "
                    f"exactly {self.num_agents} outputs; got {len(parsed)}."
                )
            if not all(isinstance(output, str) for output in parsed):
                raise TypeError(
                    "centralized_comparator_adapter.parse_completion outputs must "
                    "all be strings."
                )
            for agent_idx, output in enumerate(parsed):
                agent_completions[agent_idx].append(output)

        if not completions:
            raise ValueError("Centralized comparator produced no candidates.")

        return [
            {
                "prompts": [prompt],
                "batch_items": [batch_item],
                "completions": [completions_for_agent],
            }
            for completions_for_agent in agent_completions
        ]

    def _generate_history_policy_outputs_for_item(
        self,
        batch_item: Dict[str, Any],
        *,
        iteration_idx: int,
        num_candidates: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        checkpoint_dir = self._history_policy_checkpoint_path(iteration_idx)
        comparator_agents = self._load_policy_checkpoint_agents(checkpoint_dir)
        try:
            return self._generate_policy_outputs_for_item(
                comparator_agents,
                batch_item,
                num_candidates=num_candidates,
                use_comparator_rng=True,
                **kwargs,
            )
        finally:
            del comparator_agents
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_api_outputs_for_item(
        self,
        batch_item: Dict[str, Any],
        *,
        num_candidates: int,
    ) -> List[Dict[str, Any]]:
        def _generate_agent(agent_idx: int) -> Dict[str, Any]:
            prompt = self.formatters[agent_idx](batch_item)
            completions = self._call_comparator_api(
                prompt=prompt,
                agent_idx=agent_idx,
                batch_item=batch_item,
                num_candidates=num_candidates,
            )
            if not completions:
                raise ValueError(
                    "Comparator API returned no completions for " f"agent {agent_idx}."
                )
            return {
                "prompts": [prompt],
                "batch_items": [batch_item],
                "completions": [completions],
            }

        return self._run_agent_tasks(_generate_agent)

    def _call_comparator_api(
        self,
        *,
        prompt: str,
        agent_idx: int,
        batch_item: Dict[str, Any],
        num_candidates: int,
    ) -> List[str]:
        api_format = str(self.args.comparator_api_format or "generic").lower()
        if api_format in {"openai", "openai_chat", "chat"}:
            return self._call_openai_chat_comparator_api(
                prompt=prompt,
                num_candidates=num_candidates,
                api_format=api_format,
            )
        if api_format in {"anthropic", "anthropic_messages", "messages"}:
            return self._call_single_completion_api(
                prompt=prompt,
                num_candidates=num_candidates,
                api_format=api_format,
                build_payload=self._build_anthropic_messages_payload,
                label="an Anthropic Messages",
            )
        if api_format in {"openai_responses", "responses", "codex"}:
            return self._call_single_completion_api(
                prompt=prompt,
                num_candidates=num_candidates,
                api_format=api_format,
                build_payload=self._build_openai_responses_payload,
                label="an OpenAI Responses",
            )

        payload = self._build_generic_api_payload(
            prompt=prompt,
            agent_idx=agent_idx,
            batch_item=batch_item,
            num_candidates=num_candidates,
        )

        response_data = self._send_comparator_api_request(payload)
        return self._extract_api_completions(response_data, api_format=api_format)

    def _call_openai_chat_comparator_api(
        self,
        *,
        prompt: str,
        num_candidates: int,
        api_format: str,
    ) -> List[str]:
        max_n = self._comparator_api_max_n_per_request(num_candidates)
        completions: List[str] = []
        while len(completions) < num_candidates:
            request_n = min(max_n, num_candidates - len(completions))
            payload = self._build_openai_chat_payload(prompt, request_n)
            response_data = self._send_comparator_api_request(payload)
            batch_completions = self._extract_api_completions(
                response_data,
                api_format=api_format,
            )
            if not batch_completions:
                raise ValueError(
                    "Comparator API returned no completions for an OpenAI-format "
                    f"request with n={request_n}."
                )
            completions.extend(batch_completions)
        return completions[:num_candidates]

    def _call_single_completion_api(
        self,
        *,
        prompt: str,
        num_candidates: int,
        api_format: str,
        build_payload: Callable[[str], Dict[str, Any]],
        label: str,
    ) -> List[str]:
        """Collect candidates one request at a time for APIs without ``n``."""
        completions: List[str] = []
        while len(completions) < num_candidates:
            response_data = self._send_comparator_api_request(build_payload(prompt))
            batch_completions = self._extract_api_completions(
                response_data,
                api_format=api_format,
            )
            if not batch_completions:
                raise ValueError(
                    f"Comparator API returned no completions for {label} request."
                )
            completions.extend(batch_completions[:1])
        return completions[:num_candidates]

    def _comparator_api_max_n_per_request(self, num_candidates: int) -> int:
        configured = getattr(self.args, "comparator_api_max_n_per_request", None)
        api_format = str(self.args.comparator_api_format or "generic").lower()
        if api_format in {
            "anthropic",
            "anthropic_messages",
            "messages",
            "openai_responses",
            "responses",
            "codex",
        }:
            return 1
        if configured is not None:
            return min(max(int(configured), 1), int(num_candidates))
        if self._is_deepseek_comparator_api():
            return 1
        return int(num_candidates)

    def _is_deepseek_comparator_api(self) -> bool:
        api_url = str(getattr(self.args, "comparator_api_url", "") or "").lower()
        api_model = str(getattr(self.args, "comparator_api_model", "") or "").lower()
        return "api.deepseek.com" in api_url or api_model.startswith("deepseek-")

    def _send_comparator_api_request(self, payload: Dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(
            str(self.args.comparator_api_url),
            data=data,
            headers=self._comparator_api_headers(),
            method="POST",
        )
        try:
            with urlrequest.urlopen(
                request,
                timeout=float(self.args.comparator_api_timeout),
            ) as response:
                raw = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 402:
                raise RuntimeError(
                    "Comparator API request failed with HTTP 402: insufficient "
                    f"balance. Provider response: {detail}"
                ) from exc
            raise ValueError(
                f"Comparator API request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urlerror.URLError as exc:
            raise ValueError(f"Comparator API request failed: {exc}") from exc

        try:
            response_data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Comparator API returned non-JSON response: {raw[:500]}"
            ) from exc

        return response_data

    def _build_generic_api_payload(
        self,
        *,
        prompt: str,
        agent_idx: int,
        batch_item: Dict[str, Any],
        num_candidates: int,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "agent_idx": agent_idx,
            "num_return_sequences": num_candidates,
            "max_new_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "batch_item": self._jsonable(batch_item),
        }
        if self.args.comparator_api_model:
            payload["model"] = self.args.comparator_api_model
        if isinstance(self.args.comparator_api_extra_body, dict):
            payload.update(self.args.comparator_api_extra_body)
        return payload

    def _build_openai_chat_payload(
        self,
        prompt: str,
        num_candidates: int,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.args.comparator_api_model,
            "messages": [{"role": "user", "content": prompt}],
            "n": num_candidates,
            "max_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
        }
        if isinstance(self.args.comparator_api_extra_body, dict):
            payload.update(self.args.comparator_api_extra_body)
        return {key: value for key, value in payload.items() if value is not None}

    def _build_anthropic_messages_payload(self, prompt: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.args.comparator_api_model,
            "max_tokens": self.args.max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.args.temperature,
        }
        if isinstance(self.args.comparator_api_extra_body, dict):
            payload.update(self.args.comparator_api_extra_body)
        return {key: value for key, value in payload.items() if value is not None}

    def _build_openai_responses_payload(self, prompt: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.args.comparator_api_model,
            "input": prompt,
            "max_output_tokens": self.args.max_new_tokens,
        }
        if isinstance(self.args.comparator_api_extra_body, dict):
            payload.update(self.args.comparator_api_extra_body)
        return {key: value for key, value in payload.items() if value is not None}

    def _comparator_api_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_format = str(self.args.comparator_api_format or "generic").lower()
        if isinstance(self.args.comparator_api_headers, dict):
            headers.update(self.args.comparator_api_headers)

        if api_format in {"anthropic", "anthropic_messages", "messages"}:
            headers.setdefault("anthropic-version", "2023-06-01")

        api_key = self.args.comparator_api_key
        if not api_key and self.args.comparator_api_key_env:
            api_key = os.environ.get(str(self.args.comparator_api_key_env))
        if api_key:
            key_header = str(self.args.comparator_api_key_header or "Authorization")
            prefix = str(self.args.comparator_api_key_prefix or "").strip()
            if api_format in {"anthropic", "anthropic_messages", "messages"}:
                if key_header.lower() == "authorization":
                    key_header = "x-api-key"
                if prefix.lower() == "bearer":
                    prefix = ""
            header_value = f"{prefix} {api_key}" if prefix else str(api_key)
            headers[key_header] = header_value
        return headers

    def _extract_api_completions(
        self,
        response_data: Any,
        *,
        api_format: str,
    ) -> List[str]:
        if api_format in {"openai", "openai_chat", "chat"}:
            return self._extract_openai_chat_completions(response_data)
        if api_format in {"anthropic", "anthropic_messages", "messages"}:
            return self._extract_anthropic_messages_completions(response_data)
        if api_format in {"openai_responses", "responses", "codex"}:
            return self._extract_openai_responses_completions(response_data)

        value = self._get_dotted(response_data, self.args.comparator_api_response_field)
        if value is None and isinstance(response_data, dict):
            value = (
                response_data.get("completions")
                or response_data.get("responses")
                or response_data.get("outputs")
                or response_data.get("choices")
            )
        return self._normalize_completion_items(value)

    @staticmethod
    def _extract_openai_chat_completions(response_data: Any) -> List[str]:
        choices = (
            response_data.get("choices") if isinstance(response_data, dict) else None
        )
        completions: List[str] = []
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    completions.append(str(message["content"]))
                    continue
                if choice.get("text") is not None:
                    completions.append(str(choice["text"]))
        return completions

    @staticmethod
    def _extract_anthropic_messages_completions(response_data: Any) -> List[str]:
        content = (
            response_data.get("content") if isinstance(response_data, dict) else None
        )
        completions: List[str] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text") is not None:
                    completions.append(str(block["text"]))
        return completions

    @staticmethod
    def _extract_openai_responses_completions(response_data: Any) -> List[str]:
        output = (
            response_data.get("output") if isinstance(response_data, dict) else None
        )
        completions: List[str] = []
        if not isinstance(output, list):
            return completions
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") in {"output_text", "text"}:
                        if block.get("text") is not None:
                            completions.append(str(block["text"]))
        return completions

    @classmethod
    def _normalize_completion_items(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return [str(value)]

        completions: List[str] = []
        for item in value:
            if isinstance(item, str):
                completions.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("completion")
                    or item.get("response")
                )
                if text is None and isinstance(item.get("message"), dict):
                    text = item["message"].get("content")
                if text is not None:
                    completions.append(str(text))
            elif item is not None:
                completions.append(str(item))
        return completions

    @staticmethod
    def _get_dotted(data: Any, field: Optional[str]) -> Any:
        if not field:
            return None
        current = data
        for part in str(field).split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            pass
        if isinstance(value, dict):
            return {str(key): cls._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def _select_policy_comparison_pairs(
        self,
        current_rewards: Sequence[float],
        comparator_rewards: Sequence[float],
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        candidate_pairs: List[
            Tuple[float, float, int, Tuple[str, int], Tuple[str, int]]
        ] = []
        num_pairs = min(len(current_rewards), len(comparator_rewards))
        for pair_idx in range(num_pairs):
            current_reward = float(current_rewards[pair_idx])
            comparator_reward = float(comparator_rewards[pair_idx])
            gap = current_reward - comparator_reward
            if gap > 0:
                candidate_pairs.append(
                    (
                        abs(gap),
                        comparator_reward,
                        pair_idx,
                        ("current", pair_idx),
                        ("comparator", pair_idx),
                    )
                )
            elif gap < 0:
                candidate_pairs.append(
                    (
                        abs(gap),
                        comparator_reward,
                        pair_idx,
                        ("comparator", pair_idx),
                        ("current", pair_idx),
                    )
                )

        if not candidate_pairs:
            return []

        mode = self.args.pair_selection
        if mode == "random":
            random.shuffle(candidate_pairs)
        elif mode == "all":
            candidate_pairs.sort(key=lambda item: item[2])
        elif mode == "comparator_reward":
            candidate_pairs.sort(key=lambda item: (item[1], item[0]), reverse=True)
        else:
            candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        limit = self.args.preference_pairs_per_sample
        if mode != "all" and limit is not None:
            candidate_pairs = candidate_pairs[: int(limit)]
        return [(winner, loser) for _, _, _, winner, loser in candidate_pairs]

    def _preference_tensors_from_text(
        self,
        agent_idx: int,
        prompt: str,
        winner_completion: str,
        loser_completion: str,
    ) -> AgentPreferenceTensors:
        tokenizer = self.tokenizers[agent_idx]
        agent_module = unwrap_model(self.agents[agent_idx])
        apply_tokenizer_specials(tokenizer, [agent_module])
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long,
        )
        winner_ids = torch.tensor(
            tokenizer.encode(winner_completion, add_special_tokens=False),
            dtype=torch.long,
        )
        loser_ids = torch.tensor(
            tokenizer.encode(loser_completion, add_special_tokens=False),
            dtype=torch.long,
        )
        return AgentPreferenceTensors(
            prompt_input_ids=prompt_ids.detach().cpu(),
            winner_completion_ids=winner_ids.detach().cpu(),
            loser_completion_ids=loser_ids.detach().cpu(),
        )

    def _get_comparator_agents(self) -> Sequence[Any]:
        if self.args.comparator_policy == "current":
            return self.agents
        if self.args.comparator_policy == "current_copy":
            raise ValueError(
                "current_copy comparator agents are transient and must be generated "
                "through _generate_current_copy_outputs_for_item."
            )
        if self.args.comparator_policy == "history":
            raise ValueError("History comparator agents must be loaded per iteration.")
        if getattr(self, "_comparator_agents", None) is None:
            self._comparator_agents = self._load_comparator_agents()
        return self._comparator_agents

    def _get_centralized_comparator_agent(self) -> Any:
        if self.args.comparator_policy == "current":
            return self.agents[self._centralized_comparator_agent_index()]
        if self.args.comparator_policy == "current_copy":
            raise ValueError(
                "current_copy centralized comparator agent is transient and must be "
                "generated through _generate_centralized_current_copy_outputs_for_item."
            )
        if self.args.comparator_policy == "history":
            raise ValueError("History comparator agents must be loaded per iteration.")
        if getattr(self, "_centralized_comparator_agent", None) is None:
            source = self._centralized_comparator_source()
            self._centralized_comparator_agent = self._load_single_frozen_policy_agent(
                source,
                self._centralized_comparator_agent_index(),
            )
        return self._centralized_comparator_agent

    def _centralized_comparator_source(self) -> Any:
        if self.args.comparator_agents is not None:
            sources = self.args.comparator_agents
            if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
                raise ValueError("comparator_agents must be a non-empty sequence.")
            if len(sources) == 1:
                return list(sources)[0]
            if len(sources) == self.num_agents:
                return list(sources)[self._centralized_comparator_agent_index()]
            raise ValueError(
                "comparator_agents length must be 1 or num_agents when "
                "comparator_generation_mode='centralized'."
            )
        if self.args.comparator_model_name is None:
            raise ValueError(
                "comparator_model_name or comparator_agents is required when "
                "comparator_policy='model'."
            )
        return self.args.comparator_model_name

    def _load_comparator_agents(self) -> List[Any]:
        comparator_sources, _ = resolve_model_sources(
            kind="comparator_agents",
            model=self.args.comparator_model_name,
            models=self.args.comparator_agents,
            expected_count=self.num_agents,
            expected_label=f"num_agents ({self.num_agents})",
            model_label="comparator_model_name",
        )
        return self._load_frozen_policy_agents(comparator_sources)

    def _comparator_devices_match_agent_devices(self) -> bool:
        agent_devices = list(getattr(self, "agent_devices", []) or [])
        if len(agent_devices) != self.num_agents:
            return False
        comparator_devices = self._resolve_comparator_devices()
        if len(comparator_devices) != self.num_agents:
            return False
        return all(
            torch.device(str(agent_devices[idx])) == comparator_devices[idx]
            for idx in range(self.num_agents)
        )

    def _centralized_comparator_device_matches_agent_device(
        self,
        agent_idx: Optional[int] = None,
    ) -> bool:
        idx = (
            self._centralized_comparator_agent_index()
            if agent_idx is None
            else int(agent_idx)
        )
        agent_devices = list(getattr(self, "agent_devices", []) or [])
        if len(agent_devices) <= idx:
            return False
        comparator_devices = self._resolve_comparator_devices()
        if len(comparator_devices) <= idx:
            return False
        return torch.device(str(agent_devices[idx])) == comparator_devices[idx]

    def _prepare_iteration_current_copy_comparator(self) -> None:
        if getattr(self.args, "comparator_policy", None) != "current_copy":
            return
        if (
            getattr(self.args, "comparator_generation_mode", "decentralized")
            == "centralized"
        ):
            agent_idx = self._centralized_comparator_agent_index()
            if self._centralized_comparator_device_matches_agent_device(agent_idx):
                return
            self._clear_iteration_current_copy_comparator()
            comparator_agents: List[Any] = [None] * self.num_agents
            comparator_agents[agent_idx] = self._clone_current_agent_for_comparator(
                agent_idx
            )
            self._current_copy_comparator_agents = comparator_agents
            return
        if self._comparator_devices_match_agent_devices():
            return
        self._clear_iteration_current_copy_comparator()
        self._current_copy_comparator_agents = (
            self._clone_current_agents_for_comparator()
        )

    def _clear_iteration_current_copy_comparator(self) -> None:
        comparator_agents = getattr(self, "_current_copy_comparator_agents", None)
        if comparator_agents is None:
            return
        self._clear_transient_comparator_agents(comparator_agents)
        self._current_copy_comparator_agents = None

    def _clear_iteration_model_comparator(self) -> None:
        if getattr(self.args, "comparator_policy", None) != "model":
            return

        comparator_agents: List[Any] = []
        centralized_agent = getattr(self, "_centralized_comparator_agent", None)
        if centralized_agent is not None:
            comparator_agents.append(centralized_agent)
        cached_agents = getattr(self, "_comparator_agents", None)
        if cached_agents is not None:
            comparator_agents.extend(
                agent for agent in cached_agents if agent is not None
            )

        self._centralized_comparator_agent = None
        self._comparator_agents = None
        if comparator_agents:
            self._clear_transient_comparator_agents(comparator_agents)

    def _clone_current_agents_for_comparator(self) -> List[Any]:
        return [
            self._clone_current_agent_for_comparator(agent_idx)
            for agent_idx in range(self.num_agents)
        ]

    def _clone_current_agent_for_comparator(self, agent_idx: int) -> Any:
        source_agent = unwrap_model(self.agents[agent_idx])
        comparator_device = self._resolve_comparator_devices()[agent_idx]
        source_param = next(source_agent.parameters())
        source_dtype = source_param.dtype
        config = copy.deepcopy(getattr(source_agent, "config", None))
        if config is None:
            raise ValueError("current_copy comparator requires agents with config.")
        if hasattr(config, "torch_dtype"):
            config.torch_dtype = source_dtype

        previous_default_dtype = torch.get_default_dtype()
        if source_dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            torch.set_default_dtype(source_dtype)
        try:
            comparator_agent = source_agent.__class__(config)
        finally:
            torch.set_default_dtype(previous_default_dtype)

        state_dict = {
            key: value.detach().cpu()
            for key, value in source_agent.state_dict().items()
        }
        comparator_agent.load_state_dict(state_dict, strict=True)
        del state_dict
        comparator_agent.to(comparator_device)
        freeze_model(comparator_agent)
        apply_tokenizer_specials(self.tokenizers[agent_idx], [comparator_agent])
        return comparator_agent

    @staticmethod
    def _clear_transient_comparator_agents(comparator_agents: Sequence[Any]) -> None:
        if isinstance(comparator_agents, list):
            for agent_idx in range(len(comparator_agents)):
                comparator_agents[agent_idx] = None
        else:
            for agent in comparator_agents:
                del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_policy_checkpoint_agents(self, checkpoint_dir: str) -> List[Any]:
        checkpoint_sources = [
            os.path.join(checkpoint_dir, f"agent_{agent_idx}")
            for agent_idx in range(self.num_agents)
        ]
        missing = [path for path in checkpoint_sources if not os.path.isdir(path)]
        if missing:
            raise ValueError(
                "Missing comparator history checkpoint agent directories: "
                f"{missing}."
            )
        return self._load_frozen_policy_agents(checkpoint_sources)

    def _load_single_policy_checkpoint_agent(
        self,
        checkpoint_dir: str,
        agent_idx: int,
    ) -> Any:
        checkpoint_source = os.path.join(checkpoint_dir, f"agent_{agent_idx}")
        if not os.path.isdir(checkpoint_source):
            raise ValueError(
                "Missing comparator history checkpoint agent directory: "
                f"{checkpoint_source}."
            )
        return self._load_single_frozen_policy_agent(checkpoint_source, agent_idx)

    def _load_frozen_policy_agents(self, sources: Sequence[Any]) -> List[Any]:
        comparator_devices = self._resolve_comparator_devices()
        model_kwargs = resolve_model_kwargs(self.model_config)
        if sources and all(isinstance(src, str) for src in sources):
            comparator_agents = [
                AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
                for name in sources
            ]
        else:
            comparator_agents = list(sources)

        for agent_idx, comparator_agent in enumerate(comparator_agents):
            comparator_agent.to(comparator_devices[agent_idx])
            freeze_model(comparator_agent)
            apply_tokenizer_specials(self.tokenizers[agent_idx], [comparator_agent])

        return comparator_agents

    def _load_single_frozen_policy_agent(self, source: Any, agent_idx: int) -> Any:
        model_kwargs = resolve_model_kwargs(self.model_config)
        comparator_agent = (
            AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
            if isinstance(source, str)
            else source
        )
        comparator_device = self._resolve_comparator_devices()[agent_idx]
        comparator_agent.to(comparator_device)
        freeze_model(comparator_agent)
        apply_tokenizer_specials(self.tokenizers[agent_idx], [comparator_agent])
        return comparator_agent

    def _resolve_comparator_devices(self) -> List[torch.device]:
        return DeviceScheduler.resolve_devices(
            getattr(self.args, "comparator_devices", None)
            or getattr(self.args, "agent_devices", None),
            self.num_agents,
            kind="comparator_devices",
        )

    @staticmethod
    def _completion_group(
        source: str,
        index: int,
        current_completions: Sequence[Sequence[str]],
        comparator_completions: Sequence[Sequence[str]],
    ) -> List[str]:
        completions = (
            current_completions if source == "current" else comparator_completions
        )
        return [str(agent_completions[index]) for agent_completions in completions]

    @staticmethod
    def _reward_at(
        source: str,
        index: int,
        current_rewards: Sequence[float],
        comparator_rewards: Sequence[float],
    ) -> float:
        rewards = current_rewards if source == "current" else comparator_rewards
        return float(rewards[index])


class MARLHFIterTrainer(MADPOIterTrainer, MARLHFTrainer):
    """
    Iterative MARLHF.

    Each iteration refreshes preference pairs, retrains the joint reward model,
    then runs the configured online RL algorithm against that reward model.
    """

    default_config_cls = MARLHFIterConfig
    algorithm_name = "MARLHFIter"

    def _checkpoint_validate_progress(self, state):
        from .checkpointing import validate_preference_progress

        validate_preference_progress(self, state, "marlhf_iter")

    def _preference_scoring_raw_and_processed_rewards(
        self,
        prompts: Sequence[str],
        completions_list: List[List[str]],
        *,
        batch_items=None,
        min_completions: int,
    ) -> Optional[Tuple[List[float], List[float]]]:
        if self.args.preference_scoring_reward != "reward_model":
            return None
        if self.reward_model is None:
            return None
        if min_completions <= 0:
            return [], []

        scores = self._compute_reward_model_rewards(
            prompts,
            completions_list,
            batch_items=batch_items,
        )
        return list(scores), list(scores)

    def train(
        self,
        resume_from_checkpoint=None,
        checkpoint_dir=None,
        checkpoint_interval=1,
        checkpoint_total_limit=None,
        **kwargs,
    ):
        if (
            isinstance(checkpoint_interval, bool)
            or not isinstance(checkpoint_interval, numbers.Integral)
            or checkpoint_interval < 1
        ):
            raise ValueError("checkpoint_interval must be a positive integer.")
        if (
            resume_from_checkpoint is not None
            or checkpoint_dir is not None
            or checkpoint_interval != 1
            or checkpoint_total_limit is not None
        ):
            from .checkpointing import train_iterations_checkpointed

            with self._checkpoint_context(
                resume_from_checkpoint=resume_from_checkpoint,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                checkpoint_total_limit=checkpoint_total_limit,
            ):
                return train_iterations_checkpointed(self, reward_model=True, **kwargs)

        self._checkpoint_untracked_preference = True
        if int(self.args.num_turns) != 1:
            raise ValueError("MARLHFIter currently supports num_turns=1 only.")

        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        self._maybe_save_initial_policy_checkpoint()
        total_pairs = 0
        for iteration_idx in range(int(self.args.num_iterations)):
            self._reward_model_active = False
            self._evaluating_with_task_reward = False
            if self.args.preference_scoring_reward != "reward_model":
                self._clear_reward_model()
            self._reset_iteration_reward_distribution()
            self._prepare_iteration_current_copy_comparator()
            try:
                preference_pairs = self._build_preference_dataset(
                    iteration_idx=iteration_idx,
                    **kwargs,
                )
            finally:
                self._clear_iteration_current_copy_comparator()
                self._clear_iteration_model_comparator()
            current_pair_count = len(preference_pairs)
            train_pairs = self._select_iteration_preference_pairs(
                preference_pairs,
                iteration_idx=iteration_idx,
            )
            total_pairs += len(train_pairs)
            self._log_iteration_replay(
                iteration_idx,
                train_pairs=train_pairs,
                current_pair_count=current_pair_count,
                train_pair_count=len(train_pairs),
            )
            preference_pairs.clear()
            if not train_pairs:
                if self.verbose:
                    print(
                        "MARLHFIter: no replay preference pairs were available "
                        f"for iteration {iteration_idx + 1} "
                        f"(current generated {current_pair_count})."
                    )
                self._maybe_save_iteration_policy_checkpoint(iteration_idx)
                continue

            self._clear_reward_model()
            self._init_reward_model()
            self._train_reward_model(train_pairs)
            self._reward_model_active = True

            self._train_online_rl(**kwargs)
            self._maybe_save_iteration_policy_checkpoint(iteration_idx)
            if self.args.preference_scoring_reward != "reward_model":
                self._clear_reward_model()

        if total_pairs == 0 and self.verbose:
            print("MARLHFIter: no non-tied preference pairs were generated.")
