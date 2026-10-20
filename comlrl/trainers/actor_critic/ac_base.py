from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm  # type: ignore
from torch.utils.data import DataLoader

from comlrl.models.actor_critic import CausalLMWithValueHead
from comlrl.utils.checkpoint import TrainingCheckpointMixin
from comlrl.utils.reference_kl import reference_kl_coef, reference_kl_enabled

from ..centralized import CollaborationModeMixin


class ActorCriticTrainerBase(CollaborationModeMixin, TrainingCheckpointMixin):
    """Shared training utilities for actor-critic style trainers."""

    @property
    def num_agents(self) -> int:
        """Actor models trained: ``args.num_agents``, or one joint actor when centralized."""
        actors = self.__dict__.get("_num_actors")
        if actors is None:
            actors = getattr(getattr(self, "args", None), "num_agents", 0)
        return int(actors or 0)

    @num_agents.setter
    def num_agents(self, value: int) -> None:
        self._num_actors = int(value)

    def _kl_shaped_reward(
        self, reward: float, data: Dict[str, Any], index: int
    ) -> Tuple[float, Dict[str, float]]:
        if not reference_kl_enabled(self.args):
            return reward, {}
        reference_kl = 0.0
        raw_kls = data.get("reference_kls") or []
        if index < len(raw_kls):
            reference_kl = float(raw_kls[index])
        penalty = reference_kl_coef(self.args) * reference_kl
        return reward - penalty, {
            "reference_kl": reference_kl,
            "reference_kl_penalty": penalty,
            "environment_reward": reward,
        }

    def _expand_rewards(self, rewards: List[float], num_ret: int) -> List[List[float]]:
        num_agents = self.num_agents
        if len(rewards) == 1:
            return [[rewards[0]] * num_ret for _ in range(num_agents)]
        if len(rewards) == num_ret:
            return [list(rewards) for _ in range(num_agents)]
        if len(rewards) == num_agents:
            return [[rewards[a]] * num_ret for a in range(num_agents)]
        raise ValueError(
            "Reward function must return 1 value, num_generations values, or num_agents values."
        )

    def _value_on_prompt_only(
        self,
        model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute V(h) from prompt/history tokens, excluding actions."""
        prompt_ids = sequences[:, :prompt_len]
        prompt_mask = (
            attention_mask[:, :prompt_len] if attention_mask is not None else None
        )
        outputs = model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            output_values=True,
        )
        if outputs.values is None:
            raise RuntimeError("Value head is missing for value computation.")
        last_index = prompt_len - 1
        return outputs.values[:, last_index]

    def _compute_sequence_stats(
        self,
        sequences: torch.Tensor,
        logits: torch.Tensor,
        prompt_len: int,
        response_len: int,
        completion_loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shifted_logits = logits[:, :-1, :]
        shifted_targets = sequences[:, 1:]

        log_probs = F.log_softmax(shifted_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shifted_targets.unsqueeze(-1)
        ).squeeze(-1)

        start_index = max(prompt_len - 1, 0)
        end_index = start_index + response_len
        response_log_probs = token_log_probs[:, start_index:end_index]
        if completion_loss_mask is not None:
            if (
                completion_loss_mask.ndim != 1
                or completion_loss_mask.numel() != response_len
                or response_log_probs.size(-1) != response_len
            ):
                raise ValueError(
                    "completion_loss_mask must be one-dimensional and match response_len."
                )
            mask = completion_loss_mask.to(
                device=response_log_probs.device, dtype=torch.bool
            ).clone()
            response_log_probs = response_log_probs.masked_fill(~mask, 0.0)

        return response_log_probs.sum(dim=-1)

    def _parallel_agent_mode_enabled(self) -> bool:
        if str(getattr(self, "parallel_training", "")).lower() != "mp":
            return False
        num_agents = self.num_agents
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
        num_agents = self.num_agents
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

        reference_kls = [
            float(sample.metadata.get("reference_kl", 0.0))
            for sample in rollouts
            if "reference_kl" in sample.metadata
        ]
        if reference_kls:
            vals = torch.tensor(reference_kls, dtype=torch.float32)
            if torch.isfinite(vals).all():
                metrics["reference_kl_mean"] = float(vals.mean().item())

        reference_penalties = [
            float(sample.metadata.get("reference_kl_penalty", 0.0))
            for sample in rollouts
            if "reference_kl_penalty" in sample.metadata
        ]
        if reference_penalties:
            vals = torch.tensor(reference_penalties, dtype=torch.float32)
            if torch.isfinite(vals).all():
                metrics["reference_kl_penalty_mean"] = float(vals.mean().item())

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
        if self.wandb_initialized:
            wandb.log(metrics, step=self.env_step)

    def _domain_metrics_from_rollouts(self, rollouts: List[Any]) -> Dict[str, float]:
        callback = getattr(self, "metrics_callback", None)
        if callback is None:
            return {}
        extra = callback(rollouts)
        if not isinstance(extra, dict):
            return {}

        metrics: Dict[str, float] = {}
        for key, value in extra.items():
            try:
                metrics[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return metrics

    def _log_domain_metrics(self, rollouts: List[Any], *, prefix: str = "") -> None:
        metrics = self._domain_metrics_from_rollouts(rollouts)
        if not metrics:
            return
        if prefix:
            metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
        self._log_metrics(metrics)

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
            if self.num_agents > 0:
                # Count joint-action reward evaluations (one per agent group).
                self.env_step += len(rollouts) // self.num_agents
            self._log_domain_metrics(rollouts)

            ready_agents: List[int] = []
            for sample in rollouts:
                agent_idx = sample.agent_idx
                buffer = self.rollout_buffers[agent_idx]
                buffer.append(sample)
                if len(buffer) >= self.args.rollout_buffer_size:
                    ready_agents.append(agent_idx)
            if ready_agents:
                self._drain_ready_agent_buffers(ready_agents, epoch_metrics)

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
        domain_metric_values: Dict[str, List[float]] = defaultdict(list)
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
                        domain_metrics = self._domain_metrics_from_rollouts(rollouts)
                        for key, value in domain_metrics.items():
                            domain_metric_values[key].append(value)
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

        for key, values in domain_metric_values.items():
            if values:
                eval_log[f"eval/{key}"] = float(sum(values) / len(values))

        if eval_log:
            self._log_metrics(eval_log)
        return eval_log

    def train(
        self,
        *,
        resume_from_checkpoint=None,
        checkpoint_dir=None,
        checkpoint_interval=1,
        checkpoint_total_limit=None,
    ) -> None:
        """Train with optional resume and checkpointing at complete epoch boundaries."""
        checkpointing = self._checkpoint_arguments(
            resume_from_checkpoint,
            checkpoint_dir,
            checkpoint_interval,
            checkpoint_total_limit,
        )
        if checkpointing is not None:
            with self._checkpoint_context(**checkpointing):
                return self._train_checkpoint_epochs(
                    self._checkpoint_progress, self._checkpoint_commit
                )

        if not getattr(self, "_checkpoint_context_active", False):
            self._checkpoint_save_blocked = True

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

        if not getattr(self, "_checkpoint_context_active", False):
            self._checkpoint_progress = {
                "kind": "actor_critic",
                "epoch": total_epochs,
            }
            self._checkpoint_save_blocked = False

    def _checkpoint_validate_progress(self, state):
        """Reject incompatible saved RL cursors before restoring trainer state."""
        progress = state["progress"]
        total_epochs = self.args.num_train_epochs
        if not isinstance(progress, dict):
            raise ValueError("Actor-critic checkpoint progress must be a dictionary.")
        if progress and (
            progress.get("kind") != "actor_critic" or "epoch" not in progress
        ):
            raise ValueError(
                "Checkpoint does not contain an actor-critic epoch cursor."
            )
        start_epoch = progress.get("epoch", 0)
        if (
            type(start_epoch) is not int
            or start_epoch < 0
            or start_epoch > total_epochs
        ):
            raise ValueError(
                "Actor-critic checkpoint epoch must be between 0 and num_train_epochs."
            )
        if progress and any(state["runtime"].get("rollout_buffers", [])):
            raise ValueError(
                "Actor-critic resume requires an epoch boundary with empty rollout buffers."
            )
        return start_epoch

    def _on_checkpoint_train_start(self, start_epoch: int, total_epochs: int) -> None:
        """Align subclass state before a checkpoint-aware epoch loop starts.

        Called once after cursor validation and before any dataloader is built,
        including when all configured epochs have already been completed.
        ``start_epoch`` is the number of completed epochs, or zero for a new run.
        """

    def _train_checkpoint_epochs(self, progress, commit) -> None:
        """Run complete RL epochs, optionally inside a parent trainer's context."""
        total_epochs = self.args.num_train_epochs
        start_epoch = ActorCriticTrainerBase._checkpoint_validate_progress(
            self,
            {
                "progress": progress,
                "runtime": {"rollout_buffers": self.rollout_buffers},
            },
        )
        self._on_checkpoint_train_start(start_epoch, total_epochs)
        self._checkpoint_init_wandb()

        for epoch in range(start_epoch, total_epochs):
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
            commit(
                {"kind": "actor_critic", "epoch": epoch + 1},
                force=epoch + 1 == total_epochs,
            )
