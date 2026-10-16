import copy
import gc
import numbers
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.overrides import TorchFunctionMode
from tqdm import tqdm  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from comlrl.utils.tokenizer_utils import ensure_pad_token

from .collaboration import joint_policy_gradient_loss
from .madpo import MADPOConfig, MADPOTrainer, PreferencePair
from ..actor_critic import IACConfig, IACTrainer, MAACConfig, MAACTrainer
from ..reinforce.magrpo import MAGRPOTrainer


_RL_ALGORITHM_ALIASES = {
    "mareforce": "mareinforce",
}
_MAGRPO_ADVANTAGE_MODES = {
    "magrpo": "mean",
    "mareinforce": "raw",
    "maremax": "max",
    "marloo": "rloo",
}
_ACTOR_CRITIC_ALGORITHMS = {"maac", "iac"}


def _normalize_rl_algorithm(name: Optional[str]) -> str:
    algorithm = str(name or "magrpo").strip().lower()
    return _RL_ALGORITHM_ALIASES.get(algorithm, algorithm)


def _apply_magrpo_family_args(args: Any) -> None:
    algorithm = _normalize_rl_algorithm(getattr(args, "rl_algorithm", "magrpo"))
    setattr(args, "rl_algorithm", algorithm)
    if algorithm in _MAGRPO_ADVANTAGE_MODES:
        setattr(args, "advantage_mode", _MAGRPO_ADVANTAGE_MODES[algorithm])


class JointRewardModel(nn.Module):
    """Causal-LM backbone with a scalar reward head on the last non-pad token."""

    def __init__(self, backbone: nn.Module, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone, "config", None)
        hidden_size = (
            getattr(config, "hidden_size", None)
            or getattr(config, "n_embd", None)
            or getattr(config, "d_model", None)
        )
        if hidden_size is None:
            raise ValueError("Could not infer reward model hidden size.")
        self.reward_head = nn.Linear(int(hidden_size), 1)
        reference_param = next(self.backbone.parameters())
        self.reward_head.to(device=reference_param.device, dtype=reference_param.dtype)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        positions = torch.arange(
            attention_mask.size(1), device=hidden_states.device
        ).expand_as(attention_mask)
        last_token = positions.masked_fill(attention_mask == 0, 0).max(dim=1).values
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, last_token]
        return self.reward_head(pooled).squeeze(-1)


@dataclass
class MARLHFConfig(MADPOConfig):
    """Configuration for offline joint reward modeling followed by online MARL."""

    rl_algorithm: str = "magrpo"
    num_generations: int = 4
    reward_model_name: Optional[str] = None
    reward_model_device: Optional[str] = "cuda:0"
    reward_learning_rate: float = 1.0e-5
    reward_num_train_epochs: int = 1
    reward_train_batch_size: int = 1
    reward_max_length: Optional[int] = None
    reward_freeze_backbone: bool = False
    critic_model_name: Optional[str] = None
    critic_learning_rate: Optional[float] = None
    critic_devices: Optional[Union[str, Sequence[str]]] = None
    use_separate_critic: bool = True
    critic_type: str = "v"
    value_loss_coef: float = 0.6
    value_clip_range: Optional[float] = 0.2
    critic_value_head_hidden_dim: Optional[int] = None
    value_head_hidden_dim: Optional[int] = None

    def __post_init__(self) -> None:
        algorithm = _normalize_rl_algorithm(self.rl_algorithm)
        super().__post_init__()
        allowed = {
            "magrpo",
            "mareinforce",
            "maremax",
            "marloo",
            "maac",
            "iac",
        }
        if algorithm not in allowed:
            raise ValueError(
                "rl_algorithm must be one of: magrpo, mareinforce, maremax, marloo, maac, iac."
            )
        self.rl_algorithm = algorithm
        if algorithm in _MAGRPO_ADVANTAGE_MODES and self.num_generations < 2:
            raise ValueError(
                "num_generations must be >= 2 for MAGRPO-family MARLHF backends."
            )
        if algorithm in _ACTOR_CRITIC_ALGORITHMS and self.num_generations < 1:
            raise ValueError("num_generations must be >= 1.")
        if self.reward_learning_rate <= 0:
            raise ValueError("reward_learning_rate must be > 0.")
        if self.reward_num_train_epochs < 1:
            raise ValueError("reward_num_train_epochs must be >= 1.")
        if self.reward_train_batch_size < 1:
            raise ValueError("reward_train_batch_size must be >= 1.")
        if self.reward_max_length is not None and self.reward_max_length < 1:
            raise ValueError("reward_max_length must be >= 1 or null.")
        if self.reward_model_device is not None:
            self.reward_model_device = str(self.reward_model_device)
        if self.critic_learning_rate is not None and self.critic_learning_rate <= 0:
            raise ValueError("critic_learning_rate must be > 0 or null.")
        critic_type = str(self.critic_type or "v").strip().lower()
        if critic_type not in {"v", "q"}:
            raise ValueError("critic_type must be one of: v, q.")
        self.critic_type = critic_type
        if self.value_loss_coef <= 0:
            raise ValueError("value_loss_coef must be > 0.")
        if self.value_clip_range is not None and self.value_clip_range <= 0:
            raise ValueError("value_clip_range must be > 0 or null.")
        _apply_magrpo_family_args(self)


class MARLHFTrainer(MADPOTrainer):
    """
    Multi-Agent RLHF with an offline joint reward model.

    The preference dataset is generated with the task reward. The learned joint
    reward model is then fixed and used as the reward source for the configured
    online RL algorithm.
    """

    default_config_cls = MARLHFConfig
    algorithm_name = "MARLHF"

    def __init__(
        self,
        *args,
        metrics_callback=None,
        critic_model=None,
        critics=None,
        **kwargs,
    ):
        if kwargs.get("args") is not None:
            _apply_magrpo_family_args(kwargs["args"])
        self.metrics_callback = metrics_callback
        self.critic_model_source = critic_model
        self.critic_sources = critics
        super().__init__(*args, **kwargs)
        if (
            self.centralized_collaboration is not None
            and self.metrics_callback is not None
        ):
            self.metrics_callback = (
                self.centralized_collaboration.wrap_metrics_callback(
                    self.metrics_callback
                )
            )
        self.reward_model: Optional[JointRewardModel] = None
        self.reward_tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.reward_optimizer: Optional[torch.optim.Optimizer] = None
        self.reward_device = self._resolve_reward_model_device()
        self._reward_model_active = False
        self._evaluating_with_task_reward = False

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
            with self._checkpoint_context(
                resume_from_checkpoint=resume_from_checkpoint,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                checkpoint_total_limit=checkpoint_total_limit,
            ):
                return self._train_marlhf_checkpointed(**kwargs)

        self._checkpoint_untracked_preference = True
        if int(self.args.num_turns) != 1:
            raise ValueError("MARLHF currently supports num_turns=1 only.")

        preference_pairs = self._build_preference_dataset(**kwargs)
        if not preference_pairs:
            if self.verbose:
                print("MARLHF: no non-tied preference pairs were generated.")
            return

        self._init_reward_model()
        self._train_reward_model(preference_pairs)
        self._reward_model_active = True
        self._train_online_rl(**kwargs)

    def _checkpoint_validate_progress(self, state):
        from .checkpointing import validate_preference_progress

        validate_preference_progress(self, state, "marlhf")

    def _checkpoint_validate_extra_state(self, state, checkpoint_dir):
        super()._checkpoint_validate_extra_state(state, checkpoint_dir)
        child_state = state.get("actor_critic")
        if child_state is None:
            return
        if self.args.rl_algorithm not in _ACTOR_CRITIC_ALGORITHMS:
            raise ValueError("Unexpected MARLHF actor-critic checkpoint state.")

        from comlrl.utils.checkpoint import (
            _validate_state,
            capture_rng_state,
            restore_rng_state,
        )

        # AC constructors can move supplied models and attach heads. A meta
        # probe checks their structure without allocating another set of LLM
        # weights on a training device or changing the live parent.
        rng = capture_rng_state()
        try:
            try:
                child = self._checkpoint_actor_critic_probe("meta")
            except (RuntimeError, NotImplementedError) as error:
                # Custom factories may inspect tensor values during setup.
                # Keep that extension working, but never use GPU copies here.
                if "meta" not in str(error).lower():
                    raise
                restore_rng_state(rng)
                child = self._checkpoint_actor_critic_probe("cpu")
            _validate_state(child, child_state)
            validator = getattr(child, "_checkpoint_validate_extra_state", None)
            if callable(validator):
                validator(child_state["extra"], checkpoint_dir)
            validator = getattr(child, "_checkpoint_validate_assets", None)
            if callable(validator) and "assets" in child_state:
                validator(checkpoint_dir, child_state["assets"])
            return {
                "rng_devices": [
                    {
                        "saved": child_state.get("rng_devices"),
                        "current": child._checkpoint_rng_devices_override,
                    }
                ]
            }
        finally:
            restore_rng_state(rng)

    def _checkpoint_actor_critic_probe(self, device):
        probe = copy.copy(self)
        fields = (
            "args",
            "agents",
            "tokenizers",
            "critic_sources",
            "critic_model_source",
            "model_config",
            "agent_devices",
            "wandb_config",
        )
        # Callbacks and datasets remain read-only constructor inputs. Their
        # identities can matter, and they may own noncopyable locks/resources.
        inputs = {name: getattr(self, name) for name in fields if hasattr(self, name)}
        memo = {}

        def isolate(value):
            if isinstance(value, torch.nn.Module):
                for tensor in (*value.parameters(), *value.buffers()):
                    if id(tensor) in memo:
                        continue
                    if device == "meta":
                        copied = torch.empty_like(tensor, device="meta")
                    else:
                        copied = tensor.detach().to(device="cpu", copy=True)
                    if isinstance(tensor, torch.nn.Parameter):
                        copied = torch.nn.Parameter(
                            copied, requires_grad=tensor.requires_grad
                        )
                    memo[id(tensor)] = copied
            elif isinstance(value, dict):
                for item in value.values():
                    isolate(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    isolate(item)

        isolate(inputs)
        vars(probe).update(copy.deepcopy(inputs, memo))
        probe._checkpoint_validation_device = device

        class ProbeDeviceGuard(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                requested = kwargs.get("device")
                name = getattr(func, "__name__", "")
                if name in {"cuda", "mps", "xpu"}:
                    raise ValueError("Checkpoint validation requires CPU/meta devices.")
                if name == "to" and len(args) > 1:
                    target = args[1]
                    if isinstance(target, torch.Tensor):
                        requested = target.device
                    elif isinstance(target, (str, int, torch.device)):
                        requested = target
                if requested is not None and (
                    isinstance(requested, int)
                    or torch.device(requested).type not in {"cpu", "meta"}
                ):
                    raise ValueError("Checkpoint validation requires CPU/meta devices.")
                return func(*args, **kwargs)

        with torch.device(device), ProbeDeviceGuard():
            if device == "meta":
                probe._checkpoint_prepare_meta_critics()
            child = probe._create_actor_critic_trainer()
        planned_args = getattr(probe, "_checkpoint_planned_ac_args", None)
        if planned_args is None:
            planned_args = (
                MAACConfig(**probe._build_actor_critic_common_config())
                if self.args.rl_algorithm == "maac"
                else probe._build_iac_config()
            )
        child._checkpoint_rng_devices_override = (
            probe._checkpoint_actor_critic_device_plan(child, planned_args)
        )
        return child

    def _checkpoint_prepare_meta_critics(self):
        """Construct identifier-based critics from configuration, without weights."""
        from transformers import AutoConfig, GenerationConfig
        from ..actor_critic.iac import IACTrainer as BuiltinIACTrainer
        from ..actor_critic.maac import MAACTrainer as BuiltinMAACTrainer
        from comlrl.utils.model_loading import resolve_model_kwargs

        # Replacement factories keep their original inputs and CPU fallback.
        # The built-in factory accepts these same preloaded model objects.
        if self.args.rl_algorithm == "maac":
            if MAACTrainer is not BuiltinMAACTrainer:
                return
            count = 1
        else:
            if IACTrainer is not BuiltinIACTrainer or not self.args.use_separate_critic:
                return
            count = self._build_iac_config().num_agents
        sources = self.critic_sources
        if sources is None:
            sources = [self._resolve_critic_model_source()] * count
        config = self._build_actor_critic_model_config()
        kwargs = resolve_model_kwargs(config, config.get("critic_model_kwargs", {}))
        critics = []
        for source in sources:
            if isinstance(source, str):
                model_config = AutoConfig.from_pretrained(source)
                model = AutoModelForCausalLM.from_config(model_config, **kwargs)
                try:
                    model.generation_config = GenerationConfig.from_pretrained(source)
                except OSError:
                    # from_pretrained also derives generation defaults when
                    # the optional generation_config.json is absent.
                    pass
                model.eval()
                critics.append(model)
            else:
                critics.append(source)
        self.critic_sources = critics

    def _checkpoint_actor_critic_device_plan(self, child, args):
        """Resolve the online child's real placement, not its probe devices."""
        from comlrl.schedulers import DeviceScheduler
        from comlrl.schedulers.device_scheduler import normalize_parallel_training

        if normalize_parallel_training(args.parallel_training) == "mp":
            if self.args.rl_algorithm == "maac":
                agent_devices = DeviceScheduler.resolve_devices(
                    args.agent_devices, args.num_agents, kind="agent_devices"
                )
                critic_devices = [
                    DeviceScheduler.assign_shared_critic_device(
                        agent_devices, args.critic_devices
                    )
                ]
            else:
                agent_devices, critic_devices = DeviceScheduler.assign_devices(
                    args.num_agents,
                    args.agent_devices,
                    args.critic_devices,
                    use_separate_critic=args.use_separate_critic,
                )
        else:
            device = DeviceScheduler.resolve_single_device(
                args.agent_devices, args.critic_devices
            )
            agent_devices = [device] * args.num_agents
            critic_devices = [device] * args.num_agents

        def canonical(device):
            device = torch.device(device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            return str(device)

        return {
            f"{name}.{index}": canonical(devices[index])
            for name, devices in (
                ("agents", agent_devices),
                ("critics", critic_devices),
            )
            for index in range(len(getattr(child, name, [])))
        }

    def _train_marlhf_checkpointed(self, **kwargs):
        """Resume after reward training or a complete online-RL epoch."""
        if int(self.args.num_turns) != 1:
            raise ValueError("MARLHF currently supports num_turns=1 only.")
        saved = getattr(self, "_checkpoint_progress", {})
        if not isinstance(saved, dict):
            raise ValueError("MARLHF checkpoint progress must be a dictionary.")
        progress = dict(saved)
        if progress and (
            progress.get("kind") != "marlhf"
            or progress.get("phase") not in ("online_rl", "complete")
            or (progress["phase"] == "online_rl" and "rl" not in progress)
        ):
            raise ValueError(
                "The checkpoint does not contain MARLHF training progress."
            )
        rl = progress.get("rl", {})
        expected_kind = (
            "magrpo"
            if self.args.rl_algorithm in _MAGRPO_ADVANTAGE_MODES
            else "actor_critic"
        )
        if not isinstance(rl, dict) or (
            rl
            and (
                rl.get("kind") != expected_kind
                or type(rl.get("epoch")) is not int
                or not 0 <= rl["epoch"] <= int(self.args.num_train_epochs)
            )
        ):
            raise ValueError("Invalid MARLHF online-RL checkpoint cursor.")
        if progress.get("phase") == "complete":
            if not rl or rl["epoch"] == int(self.args.num_train_epochs):
                return
            progress["phase"] = "online_rl"
        if progress.get("phase") not in (None, "online_rl"):
            raise ValueError("Invalid MARLHF checkpoint phase.")
        self._checkpoint_init_wandb()
        if not progress:
            pairs = self._build_preference_dataset(**kwargs)
            if not pairs:
                if self.verbose:
                    print("MARLHF: no non-tied preference pairs were generated.")
                self._checkpoint_commit(
                    {"kind": "marlhf", "phase": "complete"}, force=True
                )
                return
            self._init_reward_model()
            self._train_reward_model(pairs)
            self._reward_model_active = True
            progress = {"kind": "marlhf", "phase": "online_rl", "rl": {}}
            self._checkpoint_commit(progress, force=True)
        if self.reward_model is None:
            raise ValueError("MARLHF online-RL checkpoint is missing its reward model.")
        self._reward_model_active = True

        def commit_rl(rl_progress, force=False):
            self._checkpoint_commit(
                {"kind": "marlhf", "phase": "online_rl", "rl": rl_progress},
                force=force,
            )

        if self.args.rl_algorithm in _MAGRPO_ADVANTAGE_MODES:
            _apply_magrpo_family_args(self.args)
            self.advantage_mode = self.args.advantage_mode
            MAGRPOTrainer._train_checkpoint_epochs(
                self, progress.get("rl", {}), commit_rl, **kwargs
            )
        elif self.args.rl_algorithm in _ACTOR_CRITIC_ALGORITHMS:
            self._train_actor_critic_rl(
                _checkpoint_progress=progress.get("rl", {}),
                _checkpoint_commit=commit_rl,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported rl_algorithm: {self.args.rl_algorithm}")
        self._checkpoint_commit(
            {
                "kind": "marlhf",
                "phase": "complete",
                "rl": self._checkpoint_progress.get("rl", {}),
            },
            force=True,
        )

    def _train_online_rl(self, **kwargs) -> None:
        if self.args.rl_algorithm in _MAGRPO_ADVANTAGE_MODES:
            _apply_magrpo_family_args(self.args)
            self.advantage_mode = self.args.advantage_mode
            MAGRPOTrainer.train(self, **kwargs)
        elif self.args.rl_algorithm in _ACTOR_CRITIC_ALGORITHMS:
            self._train_actor_critic_rl(**kwargs)
        else:
            raise ValueError(f"Unsupported rl_algorithm: {self.args.rl_algorithm}")

    def evaluate(self, num_eval_samples: Optional[int] = None) -> Dict[str, float]:
        previous = self._evaluating_with_task_reward
        self._evaluating_with_task_reward = True
        try:
            return super().evaluate(num_eval_samples=num_eval_samples)
        finally:
            self._evaluating_with_task_reward = previous

    def _compute_rewards(
        self,
        prompts,
        completions_list,
        batch_items=None,
    ) -> List[float]:
        if self._reward_model_active and not self._evaluating_with_task_reward:
            return self._compute_reward_model_rewards(
                prompts,
                completions_list,
                batch_items=batch_items,
            )
        return MAGRPOTrainer._compute_rewards(
            self,
            prompts,
            completions_list,
            batch_items=batch_items,
        )

    def _compute_loss_with_gradients(self, agent, completions_data, returns):
        if self.centralized_collaboration is None:
            return super()._compute_loss_with_gradients(
                agent, completions_data, returns
            )
        return joint_policy_gradient_loss(
            self,
            agent,
            completions_data,
            returns,
            sequence_log_prob=partial(self._sequence_log_prob, 0),
        )

    def _train_actor_critic_rl(
        self, _checkpoint_progress=None, _checkpoint_commit=None, **kwargs
    ) -> None:
        restoring_child = (
            _checkpoint_commit is not None
            and getattr(self, "_checkpoint_pending_ac_state", None) is not None
        )
        if restoring_child:
            from comlrl.utils.checkpoint import capture_rng_state, restore_rng_state

            rng_before_init = capture_rng_state()
        ac_trainer = self._create_actor_critic_trainer()
        if _checkpoint_commit is None:
            ac_trainer.train()
        else:
            self._checkpoint_ac_trainer = ac_trainer
            if restoring_child:
                from comlrl.utils.checkpoint import restore_trainer_state

                restore_trainer_state(
                    ac_trainer,
                    self._checkpoint_pending_ac_state,
                    self._checkpoint_loaded_from,
                    restore_rng=False,
                )
                self._checkpoint_pending_ac_state = None
                restore_rng_state(rng_before_init)

            def commit_child(child_progress, force=False):
                self.env_step = int(ac_trainer.env_step)
                ac_trainer._checkpoint_progress = child_progress
                _checkpoint_commit(child_progress, force=force)

            ac_trainer._train_checkpoint_epochs(
                _checkpoint_progress or {}, commit_child
            )
        self.env_step = int(getattr(ac_trainer, "env_step", self.env_step))
        self.agents = [agent.model for agent in ac_trainer.agents]

    def _create_actor_critic_trainer(self):
        def prepare_args(args):
            device = getattr(self, "_checkpoint_validation_device", None)
            if device is not None:
                self._checkpoint_planned_ac_args = copy.deepcopy(args)
                args.agent_devices = device
                args.critic_devices = device
                args.reference_devices = device
            return args

        def prepare_model_config():
            return self._build_actor_critic_model_config()

        trainer_ref: Dict[str, Any] = {"trainer": None}

        def reward_model_reward_func(
            *agent_completions,
            prompts=None,
            batch_items=None,
            **_unused,
        ) -> List[float]:
            completions_per_agent = self._normalize_ac_completions(agent_completions)
            active_trainer = trainer_ref.get("trainer")
            if active_trainer is not None and getattr(
                active_trainer, "_in_eval", False
            ):
                return MAGRPOTrainer._compute_rewards(
                    self,
                    prompts or [""],
                    completions_per_agent,
                    batch_items=batch_items,
                )
            return self._compute_reward_model_rewards(
                prompts or [""],
                completions_per_agent,
                batch_items=batch_items,
            )

        if self.args.rl_algorithm == "maac":
            ac_args = prepare_args(
                MAACConfig(**self._build_actor_critic_common_config())
            )
            ac_trainer = MAACTrainer(
                agent_model=None,
                agents=self.agents,
                tokenizer=self.tokenizers,
                reward_func=reward_model_reward_func,
                reward_processor=None,
                formatters=self.formatters,
                metrics_callback=self.metrics_callback,
                external_transition=self.external_transition,
                args=ac_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                model_config=prepare_model_config(),
                wandb_config=None,
                critic_model=self._resolve_critic_model_source(),
                critics=self.critic_sources,
            )
        else:
            ac_args = prepare_args(self._build_iac_config())
            ac_trainer = IACTrainer(
                agent_model=None,
                agents=self.agents,
                tokenizer=self.tokenizers,
                reward_func=reward_model_reward_func,
                reward_processor=None,
                formatters=self.formatters,
                metrics_callback=self.metrics_callback,
                external_transition=self.external_transition,
                args=ac_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                model_config=prepare_model_config(),
                wandb_config=None,
                critic_model=(
                    self._resolve_critic_model_source()
                    if bool(self.args.use_separate_critic)
                    else None
                ),
                critics=(
                    self.critic_sources if bool(self.args.use_separate_critic) else None
                ),
            )

        trainer_ref["trainer"] = ac_trainer
        ac_trainer.wandb_config = self.wandb_config
        ac_trainer.wandb_initialized = self.wandb_initialized
        ac_trainer.verbose = self.verbose
        if self.centralized_collaboration is not None:
            ac_trainer.env_step = self.env_step
        return ac_trainer

    def _normalize_ac_completions(self, agent_completions) -> List[List[str]]:
        completion_lists = [list(completions) for completions in agent_completions]
        if (
            len(completion_lists) == 1
            and completion_lists[0]
            and all(isinstance(item, (list, tuple)) for item in completion_lists[0])
        ):
            completion_lists = [list(item) for item in completion_lists[0]]
        if len(completion_lists) != self.num_agents:
            raise ValueError(
                "MARLHF reward model expected one completion list per agent."
            )
        return completion_lists

    def _build_actor_critic_common_config(self) -> Dict[str, Any]:
        parallel_training = str(getattr(self.args, "parallel_training", "none")).lower()
        critic_devices = getattr(self.args, "critic_devices", None)
        if parallel_training == "mp" and critic_devices is None:
            critic_devices = getattr(self.args, "agent_devices", None)
        return {
            "agent_learning_rate": self.args.agent_learning_rate,
            "critic_learning_rate": (
                self.args.critic_learning_rate or self.args.agent_learning_rate
            ),
            "rollout_buffer_size": self.args.rollout_buffer_size,
            "train_batch_size": self.args.train_batch_size,
            "value_loss_coef": self.args.value_loss_coef,
            "advantage_normalization": self.args.advantage_normalization,
            "max_new_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "num_train_epochs": self.args.num_train_epochs,
            "num_agents": self.num_agents,
            "num_generations": self.args.num_generations,
            "num_turns": self.args.num_turns,
            "parallel_training": parallel_training,
            "agent_devices": getattr(self.args, "agent_devices", None),
            "critic_devices": critic_devices,
            "external_prompt_passthrough": self.args.external_prompt_passthrough,
            "discount": self.args.discount,
            "critic_type": self.args.critic_type,
            "early_termination_threshold": self.args.early_termination_threshold,
            "eval_interval": self.args.eval_interval,
            "eval_num_samples": self.args.eval_num_samples,
            "eval_batch_size": self.args.eval_batch_size,
            "logging_steps": self.args.logging_steps,
            "reference_kl_enabled": self.args.reference_kl_enabled,
            "reference_kl_coef": self.args.reference_kl_coef,
            "reference_devices": self.args.reference_devices,
        }

    def _build_iac_config(self) -> IACConfig:
        config = self._build_actor_critic_common_config()
        config.update(
            {
                "use_separate_critic": self.args.use_separate_critic,
                "value_clip_range": self.args.value_clip_range,
                "critic_value_head_hidden_dim": (
                    self.args.critic_value_head_hidden_dim
                ),
                "value_head_hidden_dim": self.args.value_head_hidden_dim,
            }
        )
        return IACConfig(**config)

    def _build_actor_critic_model_config(self) -> Dict[str, Any]:
        actor_kwargs = self._model_kwargs_from_config()
        critic_kwargs = dict(actor_kwargs)
        if isinstance(self.model_config, dict):
            nested_critic = self.model_config.get("critic_model_kwargs")
            if isinstance(nested_critic, dict):
                critic_kwargs = self._model_kwargs_from_config(nested_critic)
        return {
            "model_kwargs": actor_kwargs,
            "critic_model_kwargs": critic_kwargs,
        }

    def _model_kwargs_from_config(
        self,
        source_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = source_config if source_config is not None else self.model_config
        model_kwargs: Dict[str, Any] = {}
        if isinstance(config, dict):
            torch_dtype = config.get("torch_dtype") or config.get("dtype")
            if torch_dtype is None and isinstance(config.get("model_kwargs"), dict):
                nested = config["model_kwargs"]
                torch_dtype = nested.get("torch_dtype") or nested.get("dtype")
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            attn_implementation = "sdpa"
            nested = config.get("model_kwargs")
            if "attn_implementation" in config:
                attn_implementation = config.get("attn_implementation")
            elif isinstance(nested, dict) and "attn_implementation" in nested:
                attn_implementation = nested.get("attn_implementation")
            elif source_config is not None and isinstance(self.model_config, dict):
                nested = self.model_config.get("model_kwargs")
                if "attn_implementation" in self.model_config:
                    attn_implementation = self.model_config.get("attn_implementation")
                elif isinstance(nested, dict) and "attn_implementation" in nested:
                    attn_implementation = nested.get("attn_implementation")
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation
        else:
            model_kwargs["attn_implementation"] = "sdpa"
        return model_kwargs

    def _resolve_critic_model_source(self):
        if self.critic_sources is not None:
            return None
        source = (
            self.critic_model_source or self.args.critic_model_name or self.model_name
        )
        if not source or not isinstance(source, str):
            raise ValueError(
                "critic_model_name must be provided for MARLHF with MAAC/IAC "
                "when agent models are objects."
            )
        return source

    def _init_reward_model(self) -> None:
        source = self.args.reward_model_name or self.model_name
        if not source or not isinstance(source, str):
            raise ValueError(
                "reward_model_name must be provided when agent models are objects."
            )

        tokenizer = AutoTokenizer.from_pretrained(source)
        tokenizer = ensure_pad_token(tokenizer)
        model_kwargs = self._model_kwargs_from_config()
        backbone = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
        self.reward_model = JointRewardModel(
            backbone,
            freeze_backbone=self.args.reward_freeze_backbone,
        ).to(self.reward_device)
        self.reward_tokenizer = tokenizer
        self.reward_optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=float(self.args.reward_learning_rate),
        )

    def _clear_reward_model(self) -> None:
        if self.reward_optimizer is not None:
            self.reward_optimizer.zero_grad(set_to_none=True)
        self.reward_optimizer = None
        self.reward_model = None
        self.reward_tokenizer = None
        self._reward_model_active = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_reward_model_device(self) -> torch.device:
        configured = getattr(self.args, "reward_model_device", "cuda:0")
        if configured is None:
            return self.device
        return torch.device(str(configured))

    def _train_reward_model(self, preference_pairs: List[PreferencePair]) -> None:
        if self.reward_model is None or self.reward_optimizer is None:
            raise RuntimeError("Reward model has not been initialized.")

        batch_size = int(self.args.reward_train_batch_size)
        reward_loader = DataLoader(
            preference_pairs,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda examples: examples,
            num_workers=0,
        )
        self.reward_model.train()

        for epoch in range(int(self.args.reward_num_train_epochs)):
            iterator = reward_loader
            if self.verbose:
                iterator = tqdm(
                    reward_loader,
                    total=len(reward_loader),
                    desc=(
                        "MARLHF reward model "
                        f"{epoch + 1}/{int(self.args.reward_num_train_epochs)}"
                    ),
                )

            for batch in iterator:
                winner_texts = [
                    self._format_joint_reward_text(pair, use_winner=True)
                    for pair in batch
                ]
                loser_texts = [
                    self._format_joint_reward_text(pair, use_winner=False)
                    for pair in batch
                ]
                winner_scores = self._score_reward_texts(winner_texts)
                loser_scores = self._score_reward_texts(loser_texts)
                loss = -F.logsigmoid(winner_scores - loser_scores).mean()

                self.reward_optimizer.zero_grad()
                loss.backward()
                self.reward_optimizer.step()

        self.reward_model.eval()

    def _compute_reward_model_rewards(
        self,
        prompts: Sequence[str],
        completions_list: List[List[str]],
        batch_items=None,
    ) -> List[float]:
        if self.reward_model is None:
            raise RuntimeError("Reward model has not been initialized.")
        if self.args.joint_mode.lower() not in {
            "align",
            "aligned",
        }:
            raise ValueError(
                "MARLHF reward model currently supports aligned joint_mode only."
            )

        min_completions = min(len(completions_list[i]) for i in range(self.num_agents))
        if min_completions <= 0:
            return []

        if not batch_items:
            raise ValueError("MARLHF reward model scoring requires batch_items.")
        item = batch_items[0]
        agent_prompts = [self.formatters[i](item) for i in range(self.num_agents)]

        texts: List[str] = []
        for completion_idx in range(min_completions):
            completions = [
                completions_list[agent_idx][completion_idx]
                for agent_idx in range(self.num_agents)
            ]
            texts.append(self._format_joint_text(agent_prompts, completions))

        with torch.no_grad():
            scores = self._score_reward_texts(texts)
        return [float(score) for score in scores.detach().cpu().tolist()]

    def _score_reward_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if self.reward_model is None or self.reward_tokenizer is None:
            raise RuntimeError("Reward model has not been initialized.")
        encoded = self.reward_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.args.reward_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.reward_device) for key, value in encoded.items()}
        return self.reward_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    def _format_joint_reward_text(
        self,
        pair: PreferencePair,
        *,
        use_winner: bool,
    ) -> str:
        completions = pair.winner_completions if use_winner else pair.loser_completions
        return self._format_joint_text(pair.prompts, completions)

    def _format_joint_text(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
    ) -> str:
        parts = ["Joint multi-agent response"]
        for agent_idx in range(self.num_agents):
            prompt = prompts[agent_idx] if agent_idx < len(prompts) else ""
            completion = completions[agent_idx] if agent_idx < len(completions) else ""
            parts.append(f"Agent {agent_idx + 1} prompt:\n{prompt}")
            parts.append(f"Agent {agent_idx + 1} response:\n{completion}")
        return "\n\n".join(parts)
