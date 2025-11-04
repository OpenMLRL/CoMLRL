from __future__ import annotations

import inspect
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from comlrl.models.actor_critic import CausalLMWithValueHead

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional at runtime
    wandb = None


RewardFunc = Callable[..., Sequence[float]]
Formatter = Callable[[Dict[str, Any]], str]
MetricsCallback = Callable[[List["RolloutSample"]], Dict[str, float]]


@dataclass
class IPPOConfig:
    """Configuration container for PPO fine-tuning."""

    output_dir: str = "./ippo_output"
    learning_rate: float = 5e-6
    critic_learning_rate: Optional[float] = None
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    rollout_buffer_size: int = 8
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_clip_range: Optional[float] = None
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    advantage_normalization: bool = True
    target_kl: Optional[float] = 0.3
    gamma: float = 1.0
    gae_lambda: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    do_sample: bool = True
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    seed: Optional[int] = None
    use_separate_critic: bool = False
    critic_model_name_or_path: Optional[str] = None
    critic_value_head_hidden_dim: Optional[int] = None
    value_head_hidden_dim: Optional[int] = None
    pad_token_id: Optional[int] = None
    num_agents: int = 1
    num_turns: int = 1
    logging_steps: int = 10
    log_rollouts: bool = False

    def __post_init__(self) -> None:
        if self.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")
        if self.mini_batch_size < 1:
            raise ValueError("mini_batch_size must be >= 1.")
        if self.mini_batch_size > self.rollout_buffer_size:
            self.mini_batch_size = self.rollout_buffer_size
        if self.per_device_train_batch_size != 1:
            raise ValueError("per_device_train_batch_size must be 1 for IPPO.")
        if self.num_agents != 1 or self.num_turns != 1:
            raise ValueError("Independent PPO only supports a single agent/turn.")
        if self.critic_learning_rate is None:
            self.critic_learning_rate = self.learning_rate


@dataclass
class RolloutSample:
    prompt: str
    completion: str
    full_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int
    response_len: int
    old_logprob: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    returns: torch.Tensor
    advantage: torch.Tensor
    normalized_advantage: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IPPOTrainer:
    """Independent PPO trainer with optional separate critic support."""

    def __init__(
        self,
        model: Optional[Union[str, PreTrainedModel]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        reward_func: Optional[RewardFunc] = None,
        reward_processor: Optional[Callable[[float], float]] = None,
        formatters: Optional[Union[Formatter, Sequence[Formatter]]] = None,
        args: Optional[IPPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        metrics_callback: Optional[MetricsCallback] = None,
    ) -> None:
        if reward_func is None or not callable(reward_func):
            raise ValueError("A callable reward_func must be provided.")

        self.args = args if args is not None else IPPOConfig()
        self.reward_func = reward_func
        self.reward_processor = reward_processor or (lambda x: x)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics_callback = metrics_callback
        self.model_config = model_config or {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            # CPU fallback is allowed for experimentation but will be slow.
            print("Warning: CUDA not available. Training will run on CPU.")

        if self.args.seed is not None:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.args.seed)

        self.tokenizer = tokenizer
        self.formatter = self._setup_formatter(formatters)
        self._reward_signature = self._infer_reward_signature(reward_func)

        self.actor_model: CausalLMWithValueHead
        self.critic_model: Optional[CausalLMWithValueHead] = None

        self.tokenizer = self._ensure_tokenizer(model, self.tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must expose pad_token_id.")

        self.args.pad_token_id = (
            self.args.pad_token_id
            if self.args.pad_token_id is not None
            else self.tokenizer.pad_token_id
        )

        self.actor_model = self._load_actor_model(model)
        self.actor_model.to(self.device)

        if self.args.use_separate_critic:
            critic_identifier = self.args.critic_model_name_or_path or model
            if critic_identifier is None:
                raise ValueError(
                    "critic_model_name_or_path must be provided when using a separate critic."
                )
            self.critic_model = self._load_critic_model(critic_identifier)
            self.critic_model.to(self.device)

        self._configure_tokenizer_specials()

        if self.args.use_separate_critic:
            self.actor_optimizer = torch.optim.AdamW(
                self.actor_model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
            self.critic_optimizer = torch.optim.AdamW(
                self.critic_model.parameters(),  # type: ignore[arg-type]
                lr=self.args.critic_learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.actor_model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )

        self.global_step = 0
        self.rollout_buffer: List[RolloutSample] = []

        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if wandb_config is not None:
            self._init_wandb()

    # --------------------------------------------------------------------- #
    # Initialisation helpers
    # --------------------------------------------------------------------- #
    def _ensure_tokenizer(
        self,
        model: Optional[Union[str, PreTrainedModel]],
        tokenizer: Optional[PreTrainedTokenizerBase],
    ) -> PreTrainedTokenizerBase:
        if tokenizer is not None:
            return tokenizer
        if model is None:
            raise ValueError(
                "Tokenizer must be provided when model is a PreTrainedModel instance."
            )
        tokenizer_kwargs = self.model_config.get("tokenizer_kwargs", {})
        return AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)

    def _setup_formatter(
        self,
        formatters: Optional[Union[Formatter, Sequence[Formatter]]],
    ) -> Formatter:
        default_formatter: Formatter = lambda x: x.get("prompt", "")

        if formatters is None:
            return default_formatter
        if callable(formatters):
            return formatters
        raise ValueError(
            "formatters must be None or a single callable for IPPOTrainer."
        )

    def _infer_reward_signature(self, fn: RewardFunc):
        try:
            return inspect.signature(fn)
        except (TypeError, ValueError):
            return None

    def _load_actor_model(
        self, model: Optional[Union[str, PreTrainedModel]]
    ) -> CausalLMWithValueHead:
        if model is None:
            raise ValueError("A policy model identifier or instance is required.")

        if isinstance(model, PreTrainedModel):
            base_model = model
        else:
            model_kwargs = self.model_config.get("model_kwargs", {})
            base_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

        attach_value = not self.args.use_separate_critic
        return CausalLMWithValueHead(
            base_model,
            value_head_hidden_dim=self.args.value_head_hidden_dim,
            attach_value_head=attach_value,
        )

    def _load_critic_model(
        self, model_identifier: Union[str, PreTrainedModel]
    ) -> CausalLMWithValueHead:
        if isinstance(model_identifier, PreTrainedModel):
            base_model = model_identifier
        else:
            model_kwargs = self.model_config.get("critic_model_kwargs", {})
            base_model = AutoModelForCausalLM.from_pretrained(
                model_identifier, **model_kwargs
            )

        return CausalLMWithValueHead(
            base_model,
            value_head_hidden_dim=self.args.critic_value_head_hidden_dim,
            attach_value_head=True,
        )

    def _configure_tokenizer_specials(self) -> None:
        pad_id = self.args.pad_token_id
        self.actor_model.model.config.pad_token_id = pad_id
        self.actor_model.model.config.eos_token_id = getattr(
            self.tokenizer, "eos_token_id", pad_id
        )
        if self.critic_model is not None:
            self.critic_model.model.config.pad_token_id = pad_id
            self.critic_model.model.config.eos_token_id = getattr(
                self.tokenizer, "eos_token_id", pad_id
            )

    def _init_wandb(self) -> None:
        if self.wandb_initialized:
            return
        if wandb is None:
            raise RuntimeError("wandb is not installed but wandb_config was provided.")

        project = self.wandb_config.get("project", "comlrl-ippo")
        entity = self.wandb_config.get("entity")
        name = self.wandb_config.get("name", "ippo-run")
        wandb_dir = self.wandb_config.get("dir")

        init_kwargs: Dict[str, Any] = {
            "project": project,
            "entity": entity,
            "name": name,
            "config": {
                "learning_rate": self.args.learning_rate,
                "rollout_buffer_size": self.args.rollout_buffer_size,
                "mini_batch_size": self.args.mini_batch_size,
                "ppo_epochs": self.args.ppo_epochs,
                "clip_range": self.args.clip_range,
                "entropy_coef": self.args.entropy_coef,
                "value_loss_coef": self.args.value_loss_coef,
                "max_new_tokens": self.args.max_new_tokens,
                "use_separate_critic": self.args.use_separate_critic,
            },
        }

        if wandb_dir is not None:
            os.makedirs(wandb_dir, exist_ok=True)
            init_kwargs["dir"] = wandb_dir

        tags = self.wandb_config.get("tags")
        if isinstance(tags, list):
            init_kwargs["tags"] = tags

        wandb.init(**init_kwargs)
        self.wandb_initialized = True

    # --------------------------------------------------------------------- #
    # Data utilities
    # --------------------------------------------------------------------- #
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Training requires a dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
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
            truncation=True,
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    def _call_reward_func(
        self, prompts: Sequence[str], completions: Sequence[str]
    ) -> List[float]:
        signature = self._reward_signature or inspect.signature(self.reward_func)
        params = signature.parameters
        if len(params) == 1:
            raw = self.reward_func(completions)  # type: ignore[arg-type]
        else:
            raw = self.reward_func(prompts, completions)  # type: ignore[arg-type]

        if isinstance(raw, torch.Tensor):
            rewards = raw.detach().cpu().tolist()
        elif isinstance(raw, (list, tuple)):
            rewards = list(raw)
        else:
            rewards = [float(raw)]
        return [float(self.reward_processor(r)) for r in rewards]

    # --------------------------------------------------------------------- #
    # Rollout collection
    # --------------------------------------------------------------------- #
    def _collect_rollout(self, item: Dict[str, Any]) -> RolloutSample:
        prompt = self._format_prompt(item)
        encoded_prompt = self._encode_prompt(prompt)
        prompt_input_ids = encoded_prompt["input_ids"]
        prompt_attention_mask = encoded_prompt["attention_mask"]

        prompt_len = prompt_input_ids.size(1)

        generation_kwargs: Dict[str, Any] = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": bool(self.args.do_sample),
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "pad_token_id": self.args.pad_token_id,
        }
        if self.args.top_k is not None:
            generation_kwargs["top_k"] = self.args.top_k

        sequences = self.actor_model.generate(**generation_kwargs)
        if sequences.size(1) <= prompt_len:
            raise RuntimeError("Model produced an empty completion during rollout.")

        response_tokens = sequences[:, prompt_len:]
        completion_text = self.tokenizer.decode(
            response_tokens[0], skip_special_tokens=True
        )
        response_len = response_tokens.size(1)
        response_char_length = len(completion_text)

        full_attention_mask = torch.ones_like(sequences, device=self.device)

        with torch.no_grad():
            logprob, entropy, actor_value = self._policy_eval(
                sequences, full_attention_mask, prompt_len, response_len
            )
            if self.args.use_separate_critic:
                value = self._critic_eval(
                    sequences, full_attention_mask, prompt_len, response_len
                )
            else:
                if actor_value is None:
                    raise RuntimeError("Shared value head expected a value prediction.")
                value = actor_value

        rewards = self._call_reward_func([prompt], [completion_text])
        reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        returns = reward_tensor.clone()
        advantage = returns - value

        rollout = RolloutSample(
            prompt=prompt,
            completion=completion_text,
            full_input_ids=sequences.squeeze(0).detach().cpu(),
            attention_mask=full_attention_mask.squeeze(0).detach().cpu(),
            prompt_len=prompt_len,
            response_len=response_len,
            old_logprob=logprob.detach().cpu(),
            old_value=value.detach().cpu(),
            reward=reward_tensor.detach().cpu(),
            returns=returns.detach().cpu(),
            advantage=advantage.detach().cpu(),
            entropy=entropy.detach().cpu() if entropy is not None else None,
            metadata={
                "char_length": response_char_length,
                "token_length": response_len,
            },
        )
        return rollout

    def _policy_eval(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_len: int,
        output_values: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Evaluate the actor to retrieve log-probabilities, entropy, and value prediction.
        """

        outputs = self.actor_model(
            input_ids=sequences,
            attention_mask=attention_mask,
            output_values=output_values,
        )

        logprob, entropy = self._compute_sequence_stats(
            sequences, outputs.logits, prompt_len, response_len
        )

        value = None
        if output_values and outputs.values is not None:
            last_index = prompt_len + response_len - 1
            value = outputs.values[:, last_index]

        return logprob, entropy, value

    def _critic_eval(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_len: int,
    ) -> torch.Tensor:
        if self.critic_model is None:
            raise RuntimeError("Critic model not initialised.")

        outputs = self.critic_model(
            input_ids=sequences,
            attention_mask=attention_mask,
            output_values=True,
        )
        last_index = prompt_len + response_len - 1
        return outputs.values[:, last_index]

    def _compute_sequence_stats(
        self,
        sequences: torch.Tensor,
        logits: torch.Tensor,
        prompt_len: int,
        response_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shifted_logits = logits[:, :-1, :]
        shifted_targets = sequences[:, 1:]

        log_probs = F.log_softmax(shifted_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shifted_targets.unsqueeze(-1)
        ).squeeze(-1)

        start_index = max(prompt_len - 1, 0)
        end_index = start_index + response_len
        response_log_probs = token_log_probs[:, start_index:end_index]

        logprob_sum = response_log_probs.sum(dim=-1)

        response_logits = log_probs[:, start_index:end_index, :]
        entropies = -(response_logits.exp() * response_logits).sum(dim=-1)
        entropy_mean = entropies.mean(dim=-1)

        return logprob_sum, entropy_mean

    # --------------------------------------------------------------------- #
    # PPO update logic
    # --------------------------------------------------------------------- #
    def _prepare_advantages(self, rollouts: List[RolloutSample]) -> None:
        if not rollouts:
            return

        advantages = torch.stack(
            [sample.advantage.to(torch.float32).view(-1)[0] for sample in rollouts]
        )

        if self.args.advantage_normalization and advantages.numel() > 1:
            mean = advantages.mean()
            std = advantages.std(unbiased=False).clamp(min=1e-6)
            for sample in rollouts:
                sample.normalized_advantage = (sample.advantage - mean) / std
        else:
            for sample in rollouts:
                sample.normalized_advantage = sample.advantage.clone()

    def _ppo_step(self, batch: List[RolloutSample]) -> Dict[str, float]:
        actor_losses: List[torch.Tensor] = []
        value_losses: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        approx_kls: List[torch.Tensor] = []
        ratios: List[torch.Tensor] = []

        for sample in batch:
            sequences = sample.full_input_ids.to(self.device).unsqueeze(0)
            attention_mask = sample.attention_mask.to(self.device).unsqueeze(0)

            logprob, entropy, actor_value = self._policy_eval(
                sequences,
                attention_mask,
                sample.prompt_len,
                sample.response_len,
                output_values=not self.args.use_separate_critic,
            )
            if self.args.use_separate_critic:
                value = self._critic_eval(
                    sequences,
                    attention_mask,
                    sample.prompt_len,
                    sample.response_len,
                )
            else:
                if actor_value is None:
                    raise RuntimeError("Value head missing for shared actor-critic.")
                value = actor_value

            old_logprob = sample.old_logprob.to(self.device)
            advantage = sample.normalized_advantage.to(self.device)
            returns = sample.returns.to(self.device)

            ratio = torch.exp(logprob - old_logprob)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.args.clip_range, 1.0 + self.args.clip_range
            )
            surrogate_1 = ratio * advantage
            surrogate_2 = clipped_ratio * advantage
            policy_loss = -torch.min(surrogate_1, surrogate_2)

            value_target = returns
            if (
                self.args.value_clip_range is not None
                and not self.args.use_separate_critic
            ):
                clipped_value = sample.old_value.to(self.device) + torch.clamp(
                    value - sample.old_value.to(self.device),
                    -self.args.value_clip_range,
                    self.args.value_clip_range,
                )
                value_error = torch.max(
                    (value_target - value) ** 2,
                    (value_target - clipped_value) ** 2,
                )
            else:
                value_error = (value_target - value) ** 2

            actor_losses.append(policy_loss)
            value_losses.append(value_error)
            entropies.append(entropy)
            approx_kls.append((old_logprob - logprob).detach())
            ratios.append(ratio.detach())

        actor_loss = torch.stack(actor_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_mean = torch.stack(entropies).mean()
        approx_kl = torch.stack(approx_kls).mean()
        ratio_mean = torch.stack(ratios).mean()

        actor_total = actor_loss - self.args.entropy_coef * entropy_mean
        value_total = self.args.value_loss_coef * value_loss

        if self.args.use_separate_critic:
            self.actor_optimizer.zero_grad()
            actor_total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_model.parameters(), self.args.max_grad_norm
            )
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic_model.parameters(), self.args.max_grad_norm  # type: ignore[arg-type]
            )
            self.critic_optimizer.step()
        else:
            self.optimizer.zero_grad()
            (actor_total + value_total).backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_model.parameters(), self.args.max_grad_norm
            )
            self.optimizer.step()

        return {
            "policy_loss": actor_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
            "entropy": entropy_mean.detach().item(),
            "approx_kl": approx_kl.item(),
            "ratio": ratio_mean.item(),
        }

    def _update(self, rollouts: List[RolloutSample]) -> Dict[str, float]:
        if not rollouts:
            return {}

        self._prepare_advantages(rollouts)
        rewards = torch.stack([sample.reward for sample in rollouts]).float()

        metrics = defaultdict(list)
        metrics["reward_mean"].append(rewards.mean().item())

        if self.metrics_callback is not None:
            try:
                extra = self.metrics_callback(rollouts)
                if isinstance(extra, dict):
                    for key, value in extra.items():
                        metrics[key].append(float(value))
            except Exception:
                pass

        stop_early = False
        for epoch in range(self.args.ppo_epochs):
            random.shuffle(rollouts)
            for start in range(0, len(rollouts), self.args.mini_batch_size):
                batch = rollouts[start : start + self.args.mini_batch_size]
                step_metrics = self._ppo_step(batch)
                for key, value in step_metrics.items():
                    metrics[key].append(value)

                approx_kl = step_metrics.get("approx_kl")
                if (
                    not stop_early
                    and self.args.target_kl is not None
                    and approx_kl is not None
                    and approx_kl > float(self.args.target_kl)
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        averaged = {
            key: float(sum(values) / len(values))
            for key, values in metrics.items()
            if values
        }
        return averaged

    # --------------------------------------------------------------------- #
    # Training loop
    # --------------------------------------------------------------------- #
    def train(self) -> None:
        dataloader = self.get_train_dataloader()
        total_epochs = self.args.num_train_epochs

        for epoch in range(total_epochs):
            epoch_metrics = defaultdict(list)
            for batch in dataloader:
                for item in batch:
                    rollout = self._collect_rollout(item)
                    self.rollout_buffer.append(rollout)
                    if len(self.rollout_buffer) >= self.args.rollout_buffer_size:
                        metrics = self._update(self.rollout_buffer)
                        self.rollout_buffer.clear()
                        self._log_metrics(metrics)
                        self.global_step += 1
                        for key, value in metrics.items():
                            epoch_metrics[key].append(value)

            if self.rollout_buffer:
                metrics = self._update(self.rollout_buffer)
                self.rollout_buffer.clear()
                self._log_metrics(metrics)
                self.global_step += 1
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)

            summary = {
                key: float(sum(values) / len(values))
                for key, values in epoch_metrics.items()
                if values
            }
            if summary:
                print(f"Epoch {epoch + 1}/{total_epochs} metrics: {summary}")

    # --------------------------------------------------------------------- #
    # Logging and persistence
    # --------------------------------------------------------------------- #
    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        if self.wandb_initialized and wandb is not None:
            wandb.log(metrics, step=self.global_step)

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.actor_model.model.save_pretrained(output_dir)
        if self.actor_model.value_head is not None:
            torch.save(
                self.actor_model.value_head.state_dict(),
                os.path.join(output_dir, "value_head.pt"),
            )

        if self.critic_model is not None:
            critic_dir = os.path.join(output_dir, "critic")
            os.makedirs(critic_dir, exist_ok=True)
            self.critic_model.model.save_pretrained(critic_dir)
            if self.critic_model.value_head is not None:
                torch.save(
                    self.critic_model.value_head.state_dict(),
                    os.path.join(critic_dir, "value_head.pt"),
                )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
