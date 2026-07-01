from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from comlrl.utils.tokenizer_utils import ensure_pad_token

from .madpo import MADPOConfig, MADPOTrainer, PreferencePair
from .magrpo import MAGRPOTrainer


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

    def __post_init__(self) -> None:
        super().__post_init__()
        algorithm = str(self.rl_algorithm or "magrpo").strip().lower()
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
        if self.reward_learning_rate <= 0:
            raise ValueError("reward_learning_rate must be > 0.")
        if self.reward_num_train_epochs < 1:
            raise ValueError("reward_num_train_epochs must be >= 1.")
        if self.reward_train_batch_size < 1:
            raise ValueError("reward_train_batch_size must be >= 1.")
        if self.reward_max_length is not None and self.reward_max_length < 1:
            raise ValueError("reward_max_length must be >= 1 or null.")


class MARLHFTrainer(MADPOTrainer):
    """
    Multi-Agent RLHF with an offline joint reward model.

    The preference dataset is generated with the task reward. The learned joint
    reward model is then fixed and used as the reward source for the configured
    online RL algorithm. The first implementation supports MAGRPO as the RL
    stage and keeps the broader algorithm field in config for future dispatch.
    """

    default_config_cls = MARLHFConfig
    algorithm_name = "MARLHF"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_model: Optional[JointRewardModel] = None
        self.reward_tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.reward_optimizer: Optional[torch.optim.Optimizer] = None
        self._reward_model_active = False
        self._evaluating_with_task_reward = False

    def train(self, **kwargs):
        if int(self.args.num_turns) != 1:
            raise ValueError("MARLHF currently supports num_turns=1 only.")
        if self.args.rl_algorithm != "magrpo":
            raise NotImplementedError(
                "MARLHF currently trains policies with rl_algorithm='magrpo'. "
                "The config field accepts the planned algorithms for future dispatch."
            )

        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        preference_pairs = self._build_preference_dataset(**kwargs)
        if not preference_pairs:
            if self.verbose:
                print("MARLHF: no non-tied preference pairs were generated.")
            return

        self._init_reward_model()
        self._train_reward_model(preference_pairs)
        self._reward_model_active = True

        MAGRPOTrainer.train(self, **kwargs)

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

    def _init_reward_model(self) -> None:
        source = self.args.reward_model_name or self.model_name
        if not source or not isinstance(source, str):
            raise ValueError(
                "reward_model_name must be provided when agent models are objects."
            )

        tokenizer = AutoTokenizer.from_pretrained(source)
        tokenizer = ensure_pad_token(tokenizer)
        model_kwargs: Dict[str, Any] = {}
        if isinstance(self.model_config, dict):
            torch_dtype = self.model_config.get("torch_dtype") or self.model_config.get(
                "dtype"
            )
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
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

                if self.wandb_initialized and wandb.run is not None:
                    accuracy = float(
                        (winner_scores.detach() > loser_scores.detach())
                        .float()
                        .mean()
                        .item()
                    )
                    wandb.log(
                        {
                            "reward_model/loss": float(loss.detach().cpu().item()),
                            "reward_model/accuracy": accuracy,
                        },
                        step=int(self.env_step),
                    )

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
