from collections import defaultdict
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import wandb
from tqdm import tqdm  # type: ignore
from torch.utils.data import DataLoader


class ActorCriticTrainerBase:
    """Shared training utilities for actor-critic style trainers."""

    def _parallel_agent_mode_enabled(self) -> bool:
        if str(getattr(self, "parallel_training", "")).lower() != "mp":
            return False
        num_agents = int(getattr(getattr(self, "args", None), "num_agents", 0) or 0)
        if num_agents <= 1:
            return False
        devices = getattr(self, "agent_devices", None)
        if not devices:
            return True
        unique = {str(device) for device in devices}
        return len(unique) > 1

    def _run_agent_tasks(
        self,
        fn,
        *,
        agent_indices: Optional[List[int]] = None,
        parallel: Optional[bool] = None,
    ) -> List[Any]:
        num_agents = int(getattr(getattr(self, "args", None), "num_agents", 0) or 0)
        indices = (
            list(agent_indices)
            if agent_indices is not None
            else list(range(max(num_agents, 0)))
        )
        if not indices:
            return []

        use_parallel = (
            self._parallel_agent_mode_enabled() if parallel is None else bool(parallel)
        )
        if not use_parallel or len(indices) <= 1:
            return [fn(agent_idx) for agent_idx in indices]

        results: Dict[int, Any] = {}
        max_workers = len(indices)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fn, agent_idx): agent_idx for agent_idx in indices
            }
            for future in as_completed(futures):
                agent_idx = futures[future]
                results[agent_idx] = future.result()
        return [results[agent_idx] for agent_idx in indices]

    def _filter_model_kwargs(self, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        torch_dtype = None
        if isinstance(cfg, dict):
            torch_dtype = cfg.get("torch_dtype") or cfg.get("dtype")
        if torch_dtype is None:
            model_cfg = getattr(self, "model_config", None)
            if isinstance(model_cfg, dict):
                torch_dtype = model_cfg.get("torch_dtype") or model_cfg.get("dtype")
        return {"torch_dtype": torch_dtype} if torch_dtype is not None else {}

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

    def _get_tokenizer(self, agent_idx: Optional[int] = None):
        tokenizers = getattr(self, "tokenizers", None)
        if isinstance(tokenizers, list) and tokenizers:
            if agent_idx is None:
                return tokenizers[0]
            return tokenizers[agent_idx]
        return self.tokenizer

    def _encode_prompt(
        self,
        prompt: str,
        agent_idx: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        tokenizer = tokenizer or self._get_tokenizer(agent_idx)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        target_device = device or self.device
        return {
            "input_ids": encoded["input_ids"].to(target_device),
            "attention_mask": encoded["attention_mask"].to(target_device),
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

        return metrics

    def _iter_dataloader(self, dataloader, epoch: int, total_epochs: int):
        dist_env = getattr(self, "dist_env", None)
        is_main = bool(getattr(dist_env, "is_main", True))
        if getattr(self, "verbose", True) and is_main:
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
        dist_env = getattr(self, "dist_env", None)
        if (
            summary
            and getattr(self, "verbose", True)
            and getattr(dist_env, "is_main", True)
        ):
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
    ) -> Dict[str, Any]:
        if not buffer:
            return {"metric_values": {}, "log_metrics": {}}

        has_turn_idx = any(
            "turn_idx" in (getattr(s, "metadata", {}) or {}) for s in buffer
        )
        turn_groups: Dict[int, List[Any]] = {}
        for sample in buffer:
            t_idx = int(sample.metadata.get("turn_idx", 0)) if has_turn_idx else 0
            turn_groups.setdefault(t_idx, []).append(sample)

        buffer.clear()

        combined_log: Dict[str, float] = {}
        metric_values: Dict[str, List[float]] = {}
        for t_idx in sorted(turn_groups.keys()):
            samples = turn_groups[t_idx]
            metrics = self._update(agent_idx, samples)
            tagged = self._tag_metrics(metrics, agent_idx, turn_idx=t_idx)
            combined_log.update(tagged)
            for key, value in tagged.items():
                metric_values.setdefault(key, []).append(value)
        return {"metric_values": metric_values, "log_metrics": combined_log}

    def _drain_ready_agent_buffers(
        self,
        ready_agents: List[int],
        epoch_metrics: Dict[str, List[float]],
    ) -> None:
        if not ready_agents:
            return

        unique_ready = sorted({int(idx) for idx in ready_agents})
        run_parallel = bool(
            getattr(
                self,
                "_parallel_update_enabled",
                self._parallel_agent_mode_enabled(),
            )
        )

        def _process(agent_idx: int) -> Dict[str, Any]:
            return self._process_buffer(agent_idx, self.rollout_buffers[agent_idx])

        results = self._run_agent_tasks(
            _process,
            agent_indices=unique_ready,
            parallel=run_parallel,
        )

        combined_log: Dict[str, float] = {}
        for result in results:
            metric_values = result.get("metric_values", {})
            for key, values in metric_values.items():
                for value in values:
                    epoch_metrics[key].append(value)
            combined_log.update(result.get("log_metrics", {}))

        if combined_log and self._should_log_train():
            self._log_metrics(combined_log)

    def _run_batch(self, batch, epoch_metrics: Dict[str, List[float]]) -> None:
        for item in batch:
            rollouts = self._collect_rollouts(item)
            ready_agents: List[int] = []
            for sample in rollouts:
                agent_idx = sample.agent_idx
                buffer = self.rollout_buffers[agent_idx]
                buffer.append(sample)
                if len(buffer) >= self.args.rollout_buffer_size:
                    ready_agents.append(agent_idx)
            if ready_agents:
                self._drain_ready_agent_buffers(ready_agents, epoch_metrics)
            if self.args.num_agents > 0:
                # Count joint-action reward evaluations (one per agent group).
                self.env_step += len(rollouts) // self.args.num_agents

    def _flush_buffers(self, epoch_metrics: Dict[str, List[float]]) -> None:
        ready_agents = [
            agent_idx for agent_idx, buffer in enumerate(self.rollout_buffers) if buffer
        ]
        self._drain_ready_agent_buffers(ready_agents, epoch_metrics)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Training requires a dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        if self.eval_dataset is None:
            return None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataset is None:
            return {}

        dataloader = self.get_eval_dataloader()
        if dataloader is None:
            return {}

        num_samples = int(self.args.eval_num_samples)
        turn_groups: Dict[int, List[Any]] = {}
        seen = 0

        self._in_eval = True
        try:
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
        finally:
            self._in_eval = False

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
