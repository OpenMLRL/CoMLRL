"""Single joint-input/joint-output actor optimized with task-reward MAGRPO."""

from dataclasses import dataclass

import torch

from comlrl.utils.distributed import unwrap_model

from .magrpo import MAGRPOConfig, MAGRPOTrainer


@dataclass
class CentralizedMAGRPOConfig(MAGRPOConfig):
    num_turns: int = 1
    collaboration_mode: str = "centralized"

    def __post_init__(self):
        super().__post_init__()
        if self.collaboration_mode != "centralized":
            raise ValueError(
                "CentralizedMAGRPOConfig requires centralized collaboration."
            )
        if self.num_turns != 1:
            raise ValueError("Centralized MAGRPO currently supports num_turns=1 only.")


class CentralizedMAGRPOTrainer(MAGRPOTrainer):
    """Train one joint policy; split role responses only for rewards and evaluation."""

    default_config_cls = CentralizedMAGRPOConfig

    def __init__(
        self,
        *,
        centralized_adapter,
        num_agents=2,
        agents=None,
        tokenizer=None,
        reward_func=None,
        formatters=None,
        args=None,
        **kwargs,
    ):
        # Loaded lazily because preference trainers also use the MAGRPO base.
        from ..preference.collaboration import CentralizedCollaboration

        config = args if args is not None else self.default_config_cls()
        if getattr(config, "collaboration_mode", None) != "centralized":
            raise ValueError(
                "CentralizedMAGRPOTrainer requires centralized configuration."
            )
        if config.num_turns != 1:
            raise ValueError("Centralized MAGRPO currently supports num_turns=1 only.")
        if num_agents != config.num_agents:
            raise ValueError("num_agents must match args.num_agents (the role count).")
        if agents is not None and len(agents) != 1:
            raise ValueError(
                "Centralized collaboration requires exactly one actor model."
            )
        if isinstance(tokenizer, (list, tuple)) and len(tokenizer) != 1:
            raise ValueError("Centralized collaboration requires one actor tokenizer.")
        if not callable(reward_func):
            raise ValueError("reward_func must be callable.")
        for name in ("build_prompt", "parse_completion"):
            if not callable(getattr(centralized_adapter, name, None)):
                raise TypeError(f"centralized_adapter.{name} must be callable.")
        self.centralized_collaboration = CentralizedCollaboration(
            centralized_adapter, formatters, reward_func, num_agents
        )
        self._centralized_eval_items = []
        super().__init__(
            agents=agents,
            tokenizer=tokenizer,
            num_agents=1,
            args=config,
            reward_func=self.centralized_collaboration,
            formatters=self.centralized_collaboration.build_prompt,
            **kwargs,
        )

    def evaluate(self, num_eval_samples=None):
        self._centralized_eval_items = []
        try:
            return super().evaluate(num_eval_samples=num_eval_samples)
        finally:
            self._centralized_eval_items = []

    def _evaluate_sample(self, batch_item, *args, **kwargs):
        super()._evaluate_sample(batch_item, *args, **kwargs)
        self._centralized_eval_items.append(batch_item)

    def _log_eval_metrics(self, all_agent_completions_turns, *args, **kwargs):
        role_outputs = self.centralized_collaboration.split_eval(
            all_agent_completions_turns, self._centralized_eval_items
        )
        return super()._log_eval_metrics(role_outputs, *args, **kwargs)

    def _compute_loss_with_gradients(self, agent, completions_data, returns):
        from ..preference.collaboration import joint_sequence_log_prob

        if len(returns) == 0:
            return next(agent.parameters()).reshape(-1)[0] * 0.0
        device = next(unwrap_model(agent).parameters()).device
        returns_tensor = torch.as_tensor(returns, dtype=torch.float, device=device)
        effective_returns = self._apply_reference_kl_to_returns(
            returns_tensor, completions_data
        )
        advantages = self._compute_advantages(effective_returns)
        if self.args.advantage_normalization and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(
                unbiased=False
            ).clamp(min=1e-6)
        prompt_ids = completions_data["prompt_input_ids"][0]
        sequences = completions_data["completion_input_ids"][0]
        if len(sequences) != len(advantages):
            raise ValueError("Joint completions must align with their returns.")
        losses = [
            -joint_sequence_log_prob(agent, self.tokenizers[0], prompt_ids, tokens)
            * advantage
            for tokens, advantage in zip(sequences, advantages)
        ]
        return torch.stack(losses).mean()
