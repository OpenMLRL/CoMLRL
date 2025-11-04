from __future__ import annotations

import inspect
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from comlrl.models.actor_critic import CausalLMWithValueHead


@dataclass
class IPPOConfig(TrainingArguments):
    """
    Configuration for Independent PPO with parameter sharing.

    The defaults mirror the MAGRPO configuration where possible while adding
    PPO-specific controls.
    """

    num_agents: int = field(
        default=1,
        metadata={"help": "Independent PPO currently supports a single agent."},
    )
    num_turns: int = field(
        default=1, metadata={"help": "Independent PPO currently supports one turn."}
    )
    rollout_buffer_size: int = field(
        default=4,
        metadata={
            "help": "Number of rollouts to accumulate before performing PPO updates."
        },
    )
    ppo_epochs: int = field(
        default=4, metadata={"help": "Number of PPO update epochs per batch."}
    )
    max_new_tokens: int = field(
        default=256, metadata={"help": "Maximum number of tokens to sample per rollout"}
    )
    temperature: float = field(
        default=0.7, metadata={"help": "Temperature used during sampling."}
    )
    top_p: float = field(
        default=0.9, metadata={"help": "Nucleus sampling top-p value during rollout."}
    )
    do_sample: bool = field(
        default=True,
        metadata={
            "help": "Whether to use stochastic sampling during rollout generation."
        },
    )
    clip_range: float = field(
        default=0.2, metadata={"help": "PPO policy ratio clipping range."}
    )
    clip_range_value: float = field(
        default=0.2, metadata={"help": "PPO value function clipping range."}
    )
    value_loss_coef: float = field(
        default=0.5, metadata={"help": "Coefficient for value loss contribution."}
    )
    entropy_coef: float = field(
        default=0.01, metadata={"help": "Coefficient for entropy bonus."}
    )
    gamma: float = field(default=1.0, metadata={"help": "Discount factor."})
    gae_lambda: float = field(
        default=0.95,
        metadata={"help": "Lambda used for generalized advantage estimation."},
    )
    advantage_normalization: bool = field(
        default=True,
        metadata={"help": "Normalize advantages within each rollout batch."},
    )
    advantage_clip_value: Optional[float] = field(
        default=5.0,
        metadata={
            "help": "Clip normalized advantages to [-x, x] to stabilize training."
        },
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Global gradient norm clipping value."}
    )
    target_kl: float = field(
        default=1.0,
        metadata={"help": "Stop PPO epochs early when approx KL exceeds this value."},
    )


@dataclass
class RolloutSample:
    prompt: str
    completion: str
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    full_input_ids: torch.Tensor
    full_attention_mask: torch.Tensor
    response_input_ids: torch.Tensor
    prompt_len: int
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    reward: torch.Tensor
    returns: torch.Tensor
    advantage: torch.Tensor
    normalized_advantage: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IPPOTrainer:
    """
    Independent PPO trainer with parameter sharing between actor and critic.

    Args:
        model: Hugging Face model identifier or an instantiated Causal LM.
        tokenizer: Tokenizer associated with the model.
        reward_func: Callable that scores completions.
        reward_processor: Optional post-processor applied to raw rewards.
        formatters: Optional callable or list of callables that transform dataset items.
        args: IPPOConfig instance.
        train_dataset: Dataset used for training rollouts.
        eval_dataset: Optional evaluation dataset (not yet supported).
        model_config: Optional kwargs forwarded when loading pretrained model/tokenizer.
        wandb_config: Optional Weights & Biases configuration.
    """

    def __init__(
        self,
        model: Optional[Union[str, PreTrainedModel]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        reward_func: Optional[Callable[..., Sequence[float]]] = None,
        reward_processor: Optional[Callable[[float], float]] = None,
        formatters: Optional[
            Union[Callable[[Dict[str, Any]], str], Sequence[Callable]]
        ] = None,
        args: Optional[IPPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        metrics_callback: Optional[
            Callable[[List[RolloutSample]], Dict[str, float]]
        ] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not found. IPPOTrainer requires GPU for training.")

        if reward_func is None or not callable(reward_func):
            raise ValueError("reward_func must be a callable that returns rewards.")

        self.args = args if args is not None else IPPOConfig()

        if self.args.num_agents != 1:
            raise NotImplementedError(
                "Independent PPO currently supports num_agents == 1."
            )
        if self.args.num_turns != 1:
            raise NotImplementedError(
                "Independent PPO currently supports num_turns == 1."
            )

        if self.args.per_device_train_batch_size != 1:
            raise ValueError("IPPOTrainer requires per_device_train_batch_size == 1.")
        if self.args.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")
        if self.args.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be >= 1.")

        self.device = torch.device("cuda")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reward_func = reward_func
        self._reward_signature = inspect.signature(reward_func)
        self.reward_processor = reward_processor or (lambda x: x)
        self._setup_formatter(formatters)

        self.model_config = model_config or {}
        self.tokenizer = tokenizer
        self.actor_critic = self._load_model(model)
        self.actor_critic.to(self.device)

        self.metrics_callback = metrics_callback

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when using IPPOTrainer.")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id for batching.")

        self.actor_critic.model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.actor_critic.model.config.eos_token_id = self.tokenizer.eos_token_id

        self.optimizer = torch.optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

        self.global_step = 0
        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if wandb_config is not None:
            self._init_wandb()

        self.rollout_buffer: List[RolloutSample] = []

    def _init_wandb(self) -> None:
        if self.wandb_initialized:
            return

        wandb_project = self.wandb_config.get("project", "mlrl-ippo")
        wandb_entity = self.wandb_config.get("entity")
        wandb_name = self.wandb_config.get("name", "ippo-run")
        wandb_dir = self.wandb_config.get("dir")

        config_dict = {
            "model_name": getattr(self.actor_critic.model.config, "_name_or_path", ""),
            "learning_rate": self.args.learning_rate,
            "rollout_buffer_size": self.args.rollout_buffer_size,
            "ppo_epochs": self.args.ppo_epochs,
            "clip_range": self.args.clip_range,
            "clip_range_value": self.args.clip_range_value,
            "entropy_coef": self.args.entropy_coef,
            "value_loss_coef": self.args.value_loss_coef,
            "max_new_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
        }

        init_kwargs = {
            "project": wandb_project,
            "entity": wandb_entity,
            "name": wandb_name,
            "config": config_dict,
        }

        if wandb_dir is not None:
            os.makedirs(wandb_dir, exist_ok=True)
            init_kwargs["dir"] = wandb_dir

        tags = self.wandb_config.get("tags")
        if isinstance(tags, list):
            init_kwargs["tags"] = tags

        wandb.init(**init_kwargs)
        self.wandb_initialized = True

    def _setup_formatter(
        self,
        formatters: Optional[
            Union[Callable[[Dict[str, Any]], str], Sequence[Callable]]
        ],
    ) -> None:
        default_formatter = lambda x: x.get("prompt", "")

        if formatters is None:
            self.formatter = default_formatter
        elif callable(formatters):
            self.formatter = formatters
        else:
            raise ValueError(
                "formatters must be None or a single callable for IPPOTrainer."
            )

    def _load_model(
        self,
        model: Optional[Union[str, PreTrainedModel]],
    ) -> CausalLMWithValueHead:
        if model is None:
            raise ValueError("A base model or model identifier must be provided.")

        if isinstance(model, PreTrainedModel):
            base_model = model
        elif isinstance(model, str):
            base_model = AutoModelForCausalLM.from_pretrained(
                model, **self.model_config.get("model_kwargs", {})
            )
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model, **self.model_config.get("tokenizer_kwargs", {})
                )
        else:
            raise TypeError("model must be a str or PreTrainedModel instance.")

        return CausalLMWithValueHead(base_model)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=lambda examples: examples,
        )

    def _format_prompt(self, item: Dict[str, Any]) -> str:
        prompt = self.formatter(item)
        if not isinstance(prompt, str):
            raise ValueError("Formatter must return a string prompt.")
        return prompt

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _call_reward_func(
        self, prompts: Sequence[str], completions: Sequence[str]
    ) -> List[float]:
        params = self._reward_signature.parameters
        if len(params) == 1:
            raw = self.reward_func(completions)
        elif len(params) >= 2:
            raw = self.reward_func(prompts, completions)
        else:
            raw = self.reward_func(completions)

        if isinstance(raw, torch.Tensor):
            rewards = raw.detach().cpu().tolist()
        elif isinstance(raw, (list, tuple)):
            rewards = list(raw)
        else:
            rewards = [float(raw)]
        return [float(self.reward_processor(r)) for r in rewards]

    def _logprobs_and_entropy(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
        prompt_len: int,
    ) -> Dict[str, torch.Tensor]:
        seq_len = target_tokens.size(1)
        if seq_len == 0:
            zero = torch.zeros(target_tokens.size(0), device=logits.device)
            return {"logprobs": zero, "entropy": zero}

        start_index = max(prompt_len - 1, 0)
        slice_logits = logits[:, start_index : start_index + seq_len, :]
        log_probs = F.log_softmax(slice_logits, dim=-1)
        probs = log_probs.exp()

        gathered_log_probs = log_probs.gather(
            dim=-1, index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        entropies = -(probs * log_probs).sum(dim=-1)

        return {
            "logprobs": gathered_log_probs,
            "entropy": entropies,
        }

    def _collect_rollout(self, item: Dict[str, Any]) -> RolloutSample:
        prompt = self._format_prompt(item)
        prompt_inputs = self._encode_prompt(prompt)
        prompt_input_ids = prompt_inputs["input_ids"]
        prompt_attention_mask = prompt_inputs["attention_mask"]

        prompt_len = prompt_input_ids.size(1)

        generation_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": bool(self.args.do_sample),
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        sequences = self.actor_critic.generate(**generation_kwargs)
        if sequences.size(1) <= prompt_len:
            raise ValueError("Model returned empty completion during rollout.")

        response_tokens = sequences[:, prompt_len:]
        completion_text = self.tokenizer.decode(
            response_tokens[0], skip_special_tokens=True
        ).strip()
        response_token_len = response_tokens.size(1)
        response_char_length = len(completion_text)

        full_attention_mask = torch.ones_like(sequences, device=self.device)

        with torch.no_grad():
            rollout_outputs = self.actor_critic(
                input_ids=sequences, attention_mask=full_attention_mask
            )
        logprob_entropy = self._logprobs_and_entropy(
            rollout_outputs.logits, response_tokens, prompt_len
        )
        start_index = max(prompt_len - 1, 0)
        old_values = rollout_outputs.values[
            :, start_index : start_index + response_token_len
        ]

        rewards = self._call_reward_func([prompt], [completion_text])
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        token_rewards = torch.zeros_like(old_values)
        token_rewards[:, -1] = reward_tensor
        returns, advantages = self._compute_gae(token_rewards, old_values)

        return RolloutSample(
            prompt=prompt,
            completion=completion_text,
            prompt_input_ids=prompt_input_ids.detach(),
            prompt_attention_mask=prompt_attention_mask.detach(),
            full_input_ids=sequences.detach(),
            full_attention_mask=full_attention_mask.detach(),
            response_input_ids=response_tokens.detach(),
            prompt_len=prompt_len,
            old_logprobs=logprob_entropy["logprobs"].detach(),
            old_values=old_values.detach(),
            reward=reward_tensor.detach(),
            returns=returns.detach(),
            advantage=advantages.detach(),
            entropy=logprob_entropy["entropy"].detach(),
            metadata={
                "char_length": response_char_length,
                "token_length": response_token_len,
            },
        )

    def _prepare_advantages(self, rollouts: List[RolloutSample]) -> None:
        flat_advantages = torch.cat([r.advantage.view(-1) for r in rollouts], dim=0)
        if self.args.advantage_normalization and flat_advantages.numel() > 1:
            mean = flat_advantages.mean()
            std = flat_advantages.std(unbiased=False).clamp(min=1e-6)
            for r in rollouts:
                r.normalized_advantage = (r.advantage - mean) / std
        else:
            for r in rollouts:
                r.normalized_advantage = r.advantage

        clip_val = getattr(self.args, "advantage_clip_value", None)
        if clip_val is not None and clip_val > 0:
            for r in rollouts:
                r.normalized_advantage = torch.clamp(
                    r.normalized_advantage, -clip_val, clip_val
                )

    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = float(self.args.gamma)
        lam = float(getattr(self.args, "gae_lambda", 0.95))

        next_values = torch.zeros_like(values)
        if values.size(1) > 1:
            next_values[:, :-1] = values[:, 1:]

        deltas = rewards + gamma * next_values - values

        advantages = torch.zeros_like(values)
        last_advantage = torch.zeros(values.size(0), device=values.device)

        for t in range(values.size(1) - 1, -1, -1):
            last_advantage = deltas[:, t] + gamma * lam * last_advantage
            advantages[:, t] = last_advantage

        returns = advantages + values
        return returns, advantages

    def _ppo_step(self, sample: RolloutSample) -> Dict[str, torch.Tensor]:
        outputs = self.actor_critic(
            input_ids=sample.full_input_ids,
            attention_mask=sample.full_attention_mask,
        )
        logprob_entropy = self._logprobs_and_entropy(
            outputs.logits, sample.response_input_ids, sample.prompt_len
        )
        new_logprobs = logprob_entropy["logprobs"]
        entropy = logprob_entropy["entropy"]

        ratio = torch.exp(new_logprobs - sample.old_logprobs)
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.args.clip_range, 1.0 + self.args.clip_range
        )

        adv = sample.normalized_advantage
        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

        start_index = max(sample.prompt_len - 1, 0)
        response_len = sample.response_input_ids.size(1)
        new_values = outputs.values[:, start_index : start_index + response_len]

        value_loss_unclipped = (sample.returns - new_values) ** 2
        value_clipped = sample.old_values + torch.clamp(
            new_values - sample.old_values,
            -self.args.clip_range_value,
            self.args.clip_range_value,
        )
        value_loss_clipped = (sample.returns - value_clipped) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        loss = (
            policy_loss
            + self.args.value_loss_coef * value_loss
            - self.args.entropy_coef * entropy.mean()
        )

        approx_kl = (sample.old_logprobs - new_logprobs).mean()

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach().mean(),
            "approx_kl": approx_kl.detach(),
            "ratio": ratio.detach().mean(),
        }

    def _apply_gradients(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.args.max_grad_norm
        )
        self.optimizer.step()

    def _update(self, rollouts: List[RolloutSample]) -> Dict[str, float]:
        self._prepare_advantages(rollouts)
        metrics = defaultdict(list)

        rewards = torch.cat([sample.reward.view(-1) for sample in rollouts], dim=0)
        returns = torch.cat([sample.returns.view(-1) for sample in rollouts], dim=0)
        metrics["reward_mean"].append(rewards.mean().item())
        metrics["return_mean"].append(returns.mean().item())

        if self.metrics_callback is not None:
            extra_metrics = self.metrics_callback(rollouts)
            if isinstance(extra_metrics, dict):
                for key, value in extra_metrics.items():
                    try:
                        metrics[key].append(float(value))
                    except (TypeError, ValueError):
                        continue

        stop_early = False
        for epoch in range(self.args.ppo_epochs):
            for sample in rollouts:
                step_metrics = self._ppo_step(sample)
                loss = step_metrics.pop("loss")
                if loss is None or not torch.isfinite(loss):
                    continue

                approx_kl_tensor = step_metrics.get("approx_kl")
                approx_kl_scalar: Optional[float] = None
                if approx_kl_tensor is not None:
                    approx_kl_scalar = float(approx_kl_tensor.item())
                    if not math.isfinite(approx_kl_scalar):
                        approx_kl_scalar = None

                self._apply_gradients(loss)

                for key, value in step_metrics.items():
                    metrics[key].append(value.item())

                if approx_kl_scalar is not None and approx_kl_scalar > float(
                    self.args.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        return {k: float(sum(v) / max(len(v), 1)) for k, v in metrics.items()}

    def train(self) -> None:
        dataloader = self.get_train_dataloader()
        total_epochs = int(self.args.num_train_epochs)

        for epoch in range(total_epochs):
            epoch_metrics = defaultdict(list)
            for batch in dataloader:
                for item in batch:
                    rollout = self._collect_rollout(item)
                    self.rollout_buffer.append(rollout)
                    if len(self.rollout_buffer) >= self.args.rollout_buffer_size:
                        metrics = self._update(self.rollout_buffer)
                        self.rollout_buffer.clear()
                        for key, value in metrics.items():
                            epoch_metrics[key].append(value)
                        self.global_step += 1
                        self._handle_logging(metrics)

            if self.rollout_buffer:
                metrics = self._update(self.rollout_buffer)
                self.rollout_buffer.clear()
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
                self.global_step += 1
                self._handle_logging(metrics)

            averaged = {
                k: float(sum(v) / len(v)) for k, v in epoch_metrics.items() if v
            }
            if averaged:
                print(f"Epoch {epoch + 1}/{total_epochs} metrics: {averaged}")

    def _handle_logging(self, metrics: Dict[str, float]) -> None:
        if self.wandb_initialized:
            wandb.log(metrics, step=self.global_step)

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.actor_critic.model.save_pretrained(output_dir)
        torch.save(
            self.actor_critic.value_head.state_dict(),
            os.path.join(output_dir, "ippo_value_head.pt"),
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
