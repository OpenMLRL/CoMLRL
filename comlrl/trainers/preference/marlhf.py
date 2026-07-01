from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from comlrl.utils.tokenizer_utils import ensure_pad_token

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
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_indices, lengths]
        return self.reward_head(pooled).squeeze(-1)


@dataclass
class MARLHFConfig(MADPOConfig):
    """Configuration for offline joint reward modeling followed by online MARL."""

    rl_algorithm: str = "magrpo"
    reward_model_name: Optional[str] = None
    reward_learning_rate: float = 1.0e-5
    reward_num_train_epochs: int = 1
    reward_train_batch_size: int = 2
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
        original_num_generations = self.num_generations
        if algorithm in _ACTOR_CRITIC_ALGORITHMS and original_num_generations < 2:
            self.num_generations = 2
        super().__post_init__()
        self.num_generations = original_num_generations
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
        self.reward_model: Optional[JointRewardModel] = None
        self.reward_tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.reward_optimizer: Optional[torch.optim.Optimizer] = None
        self._reward_model_active = False
        self._evaluating_with_task_reward = False

    def train(self, **kwargs):
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

        if self.args.rl_algorithm in _MAGRPO_ADVANTAGE_MODES:
            _apply_magrpo_family_args(self.args)
            self.advantage_mode = self.args.advantage_mode
            MAGRPOTrainer.train(self, **kwargs)
            return

        if self.args.rl_algorithm in _ACTOR_CRITIC_ALGORITHMS:
            self._train_actor_critic_rl(**kwargs)
            return

        raise ValueError(f"Unsupported rl_algorithm: {self.args.rl_algorithm}")

    def evaluate(self, num_eval_samples: Optional[int] = None) -> Dict[str, float]:
        previous = self._evaluating_with_task_reward
        self._evaluating_with_task_reward = True
        try:
            return MAGRPOTrainer.evaluate(self, num_eval_samples=num_eval_samples)
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

    def _train_actor_critic_rl(self, **kwargs) -> None:
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
            ac_args = self._build_maac_config()
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
                model_config=self._build_actor_critic_model_config(),
                wandb_config=None,
                critic_model=self._resolve_critic_model_source(),
                critics=self.critic_sources,
            )
        else:
            ac_args = self._build_iac_config()
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
                model_config=self._build_actor_critic_model_config(),
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
        ac_trainer.train()
        self.env_step = int(getattr(ac_trainer, "env_step", self.env_step))
        self.agents = [agent.model for agent in ac_trainer.agents]

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

    def _build_maac_config(self) -> MAACConfig:
        return MAACConfig(**self._build_actor_critic_common_config())

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
        ).to(self.device)
        self.reward_tokenizer = tokenizer
        self.reward_optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=float(self.args.reward_learning_rate),
        )

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
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
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
