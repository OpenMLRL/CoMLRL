from __future__ import annotations

from collections import defaultdict
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import wandb
from tqdm import tqdm  # type: ignore

Formatter = Callable[[Dict[str, Any]], str]


class ActorCriticTrainerBase:
    """Shared training utilities for actor-critic style trainers."""

    def _infer_model_name(self, source: Any) -> Optional[str]:
        if source is None:
            return None
        if isinstance(source, str):
            return source
        base = getattr(source, "model", source)
        config = getattr(base, "config", None)
        if config is not None:
            name = getattr(config, "_name_or_path", None) or getattr(
                config, "model_type", None
            )
            if name:
                return str(name)
        return base.__class__.__name__

    def _resolve_model_sources(
        self,
        *,
        kind: str,
        model: Optional[Any],
        models: Optional[Sequence[Any]],
        expected_count: int,
        expected_label: Optional[str] = None,
    ) -> Tuple[List[Any], Optional[str]]:
        if model is not None and models is not None:
            raise ValueError(f"Cannot provide both model and {kind}.")
        if model is None and models is None:
            raise ValueError(f"Either model or {kind} must be provided.")
        if expected_count < 1:
            raise ValueError("expected_count must be >= 1.")

        if models is not None:
            if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
                raise ValueError(f"{kind} must be a non-empty sequence.")
            sources = list(models)
            if len(sources) != expected_count:
                label = expected_label or f"num_agents ({expected_count})"
                raise ValueError(f"{kind} length ({len(sources)}) must match {label}.")
        else:
            sources = [model] * expected_count

        if any(src is None for src in sources):
            raise ValueError(f"{kind} entries must be non-null.")

        model_name = self._infer_model_name(sources[0]) if sources else None
        return sources, model_name

    def _setup_formatters(
        self, formatters: Optional[Union[Formatter, Sequence[Formatter]]]
    ) -> List[Formatter]:
        def _default_formatter(item: Dict[str, Any], external_prompts=None) -> str:
            if external_prompts is not None:
                return external_prompts
            return item.get("prompt", "")

        def _wrap_formatter(fmt: Formatter) -> Formatter:
            try:
                sig = inspect.signature(fmt)
                if "external_prompts" in sig.parameters:
                    return lambda x, external_prompts=None, f=fmt: f(
                        x, external_prompts=external_prompts
                    )
            except (TypeError, ValueError):
                pass
            return lambda x, external_prompts=None, f=fmt: f(x)

        num_agents = int(self.args.num_agents)
        if formatters is None:
            return [_default_formatter for _ in range(num_agents)]
        if callable(formatters):
            return [_wrap_formatter(formatters) for _ in range(num_agents)]
        if isinstance(formatters, Sequence) and not isinstance(
            formatters, (str, bytes)
        ):
            if len(formatters) != num_agents:
                raise ValueError(
                    "Number of formatters must match num_agents when providing a sequence."
                )
            return [_wrap_formatter(f) for f in list(formatters)]
        raise ValueError(
            "formatters must be None, a callable, or a sequence of callables."
        )

    def _format_prompt(
        self,
        item: Dict[str, Any],
        agent_idx: int,
        external_prompts: Optional[Any] = None,
    ) -> str:
        formatter = self.formatters[agent_idx]
        prompt = formatter(item, external_prompts=external_prompts)
        if not isinstance(prompt, str):
            raise ValueError("Formatter must return a string prompt.")
        return prompt

    def _resolve_turn_prompt(
        self,
        item: Dict[str, Any],
        agent_idx: int,
        external_prompt: Optional[Any] = None,
    ) -> str:
        if external_prompt is None:
            return self._format_prompt(item, agent_idx)
        if not isinstance(external_prompt, str):
            raise ValueError("External prompt must be a string.")
        if getattr(self.args, "external_prompt_passthrough", False):
            return external_prompt
        modified_item = item.copy() if hasattr(item, "copy") else dict(item)
        modified_item["_original_prompt"] = modified_item.get("prompt", "")
        modified_item["prompt"] = external_prompt
        return self._format_prompt(
            modified_item, agent_idx, external_prompts=external_prompt
        )

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

    def _prepare_advantages(self, rollouts: List[Any]) -> None:
        if not rollouts:
            return
        advantages = []
        for sample in rollouts:
            target = sample.metadata.get("adv_target")
            if target is None:
                target = sample.metadata.get("value_target")
            if target is None:
                target = sample.returns
            adv = target.to(torch.float32) - sample.old_value.to(torch.float32)
            sample.advantage = adv.to(sample.returns.dtype)
            advantages.append(adv.view(-1)[0])

        advantages = torch.stack(advantages)
        if self.args.advantage_normalization and advantages.numel() > 1:
            mean = advantages.mean()
            std = advantages.std(unbiased=False).clamp(min=1e-6)
            for sample in rollouts:
                sample.normalized_advantage = (
                    sample.advantage.to(torch.float32) - mean
                ) / std
        else:
            for sample in rollouts:
                sample.normalized_advantage = sample.advantage.clone()

    def _summarize_rollout_metrics(self, rollouts: List[Any]) -> Dict[str, float]:
        if not rollouts:
            return {}

        metrics: Dict[str, float] = {}
        rewards = torch.stack(
            [sample.reward.view(-1)[0] for sample in rollouts]
        ).float()
        if rewards.numel() > 0 and torch.isfinite(rewards).all():
            metrics["reward_mean"] = float(rewards.mean().item())

        returns = torch.stack(
            [sample.returns.view(-1)[0] for sample in rollouts]
        ).float()
        if returns.numel() > 0 and torch.isfinite(returns).all():
            metrics["expected_return"] = float(returns.mean().item())

        values = torch.stack(
            [sample.old_value.view(-1)[0] for sample in rollouts]
        ).float()
        if values.numel() > 0 and torch.isfinite(values).all():
            metrics["value_pred_mean"] = float(values.mean().item())

        targets = [sample.metadata.get("value_target") for sample in rollouts]
        if any(t is not None for t in targets):
            target_vals = torch.stack(
                [
                    (t if t is not None else sample.returns).view(-1)[0]
                    for sample, t in zip(rollouts, targets)
                ]
            ).float()
            if target_vals.numel() > 0 and torch.isfinite(target_vals).all():
                metrics["value_target_mean"] = float(target_vals.mean().item())
        elif (
            self._include_value_target_fallback()
            and returns.numel() > 0
            and torch.isfinite(returns).all()
        ):
            metrics["value_target_mean"] = float(returns.mean().item())

        return metrics

    def _include_value_target_fallback(self) -> bool:
        return True

    def _iter_dataloader(self, dataloader, epoch: int, total_epochs: int):
        if getattr(self, "verbose", True):
            return enumerate(
                tqdm(
                    dataloader,
                    total=len(dataloader),
                    desc=f"Epoch {epoch + 1}/{total_epochs}",
                )
            )
        return enumerate(dataloader)

    def _summarize_epoch_metrics(
        self, epoch_metrics: Dict[str, List[float]]
    ) -> Dict[str, float]:
        return {
            key: float(sum(values) / len(values))
            for key, values in epoch_metrics.items()
            if values
        }

    def _on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        epoch_metrics: Dict[str, List[float]],
    ) -> None:
        summary = self._summarize_epoch_metrics(epoch_metrics)
        if summary and getattr(self, "verbose", True):
            print(f"Epoch {epoch + 1}/{total_epochs} metrics: {summary}")

    def _tag_metrics(
        self, metrics: Dict[str, float], agent_idx: int, turn_idx: int = 0
    ) -> Dict[str, float]:
        prefix = f"turn_{turn_idx + 1}/"
        return {prefix + key: value for key, value in metrics.items()}

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        if self.wandb_initialized and wandb is not None:
            wandb.log(metrics, step=self.env_step)

    def _should_log_train(self) -> bool:
        interval = int(getattr(self.args, "logging_steps", 1))
        if interval <= 1:
            self._last_train_log_step = self.env_step
            return True
        if (
            self._last_train_log_step < 0
            or (self.env_step - self._last_train_log_step) >= interval
        ):
            self._last_train_log_step = self.env_step
            return True
        return False

    def _process_buffer(
        self,
        agent_idx: int,
        buffer: List[Any],
        epoch_metrics: Dict[str, List[float]],
    ) -> None:
        if not buffer:
            return

        has_turn_idx = any(
            "turn_idx" in (getattr(s, "metadata", {}) or {}) for s in buffer
        )
        turn_groups: Dict[int, List[Any]] = {}
        for sample in buffer:
            t_idx = int(sample.metadata.get("turn_idx", 0)) if has_turn_idx else 0
            turn_groups.setdefault(t_idx, []).append(sample)

        buffer.clear()

        combined_log: Dict[str, float] = {}
        for t_idx in sorted(turn_groups.keys()):
            samples = turn_groups[t_idx]
            metrics = self._update(agent_idx, samples)
            tagged = self._tag_metrics(metrics, agent_idx, turn_idx=t_idx)
            combined_log.update(tagged)
            for key, value in tagged.items():
                epoch_metrics[key].append(value)

        if combined_log and self._should_log_train():
            self._log_metrics(combined_log)

    def _run_batch(self, batch, epoch_metrics: Dict[str, List[float]]) -> None:
        for item in batch:
            rollouts = self._collect_rollouts(item)
            for sample in rollouts:
                agent_idx = sample.agent_idx
                buffer = self.rollout_buffers[agent_idx]
                buffer.append(sample)
                if len(buffer) >= self.args.rollout_buffer_size:
                    self._process_buffer(agent_idx, buffer, epoch_metrics)
            if self.args.num_agents > 0:
                # Count joint-action reward evaluations (one per agent group).
                self.env_step += len(rollouts) // self.args.num_agents

    def _flush_buffers(self, epoch_metrics: Dict[str, List[float]]) -> None:
        for agent_idx, buffer in enumerate(self.rollout_buffers):
            if not buffer:
                continue
            self._process_buffer(agent_idx, buffer, epoch_metrics)

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataset is None:
            return {}

        dataloader = self.get_eval_dataloader()
        if dataloader is None:
            return {}

        num_samples = int(self.args.eval_num_samples)
        turn_groups: Dict[int, List[Any]] = {}
        seen = 0

        with torch.no_grad():
            for batch in dataloader:
                for item in batch:
                    rollouts = self._collect_rollouts(item)
                    for sample in rollouts:
                        t_idx = int(sample.metadata.get("turn_idx", 0))
                        turn_groups.setdefault(t_idx, []).append(sample)
                    seen += 1
                    if seen >= num_samples:
                        break
                if seen >= num_samples:
                    break

        eval_log: Dict[str, float] = {}
        for turn_idx, samples in sorted(turn_groups.items()):
            metrics = self._summarize_rollout_metrics(samples)
            for key, value in metrics.items():
                eval_log[f"eval/turn_{turn_idx + 1}/{key}"] = value

        if eval_log:
            self._log_metrics(eval_log)
        return eval_log

    def train(self) -> None:
        total_epochs = self.args.num_train_epochs

        for epoch in range(total_epochs):
            epoch_metrics = defaultdict(list)
            dataloader = self.get_train_dataloader()
            it = self._iter_dataloader(dataloader, epoch, total_epochs)
            for batch_idx, batch in it:
                if (
                    self.eval_dataset is not None
                    and self.args.eval_interval > 0
                    and batch_idx % int(self.args.eval_interval) == 0
                ):
                    self.evaluate()
                self._run_batch(batch, epoch_metrics)

            self._flush_buffers(epoch_metrics)
            self._on_epoch_end(epoch, total_epochs, epoch_metrics)
