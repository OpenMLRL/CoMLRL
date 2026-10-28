"""Collaboration modes shared by every trainer family.

``decentralized`` trains one actor per agent. ``centralized`` trains one joint actor that
answers for all ``num_agents`` roles: an adapter builds one prompt from the role prompts and
splits the joint response back into role outputs, which the task reward and the evaluation
logger see unchanged. Trainers opt in through :class:`CollaborationModeMixin`.
"""

from typing import Any, Callable, Optional, Sequence, Tuple

from .collaboration import CentralizedCollaboration

COLLABORATION_MODES = ("decentralized", "centralized")


def validate_collaboration_mode(mode: str) -> None:
    if mode not in COLLABORATION_MODES:
        raise ValueError("collaboration_mode must be decentralized or centralized.")


class CollaborationModeMixin:
    """Hooks a trainer calls to support ``args.collaboration_mode="centralized"``.

    * ``_setup_collaboration`` in ``__init__``: wraps the role formatters and the task
      reward around one joint actor and returns the actor-side arguments.
    * ``_role_completions`` before logging evaluation metrics: splits each joint
      completion into role outputs.
    * ``is_centralized`` wherever the joint actor needs its own objective.
    """

    centralized_collaboration: Optional[CentralizedCollaboration] = None

    @property
    def is_centralized(self) -> bool:
        return self.centralized_collaboration is not None

    def _setup_collaboration(
        self,
        adapter: Any,
        *,
        agents: Optional[Sequence[Any]],
        tokenizer: Any,
        reward_func: Optional[Callable],
        formatters: Any,
        num_agents: int,
    ) -> Tuple[Any, Any, int]:
        """Return ``(reward_func, formatters, num_actors)`` for the configured mode."""
        if getattr(self.args, "collaboration_mode", "decentralized") != "centralized":
            return reward_func, formatters, num_agents
        if getattr(self.args, "num_turns", 1) != 1:
            raise ValueError(
                "Centralized collaboration currently supports num_turns=1 only."
            )
        if num_agents != self.args.num_agents:
            raise ValueError("num_agents must match args.num_agents (the role count).")
        if agents is not None and len(agents) != 1:
            raise ValueError(
                "Centralized collaboration requires exactly one actor model."
            )
        if isinstance(tokenizer, (list, tuple)) and len(tokenizer) != 1:
            raise ValueError("Centralized collaboration requires one actor tokenizer.")
        if not callable(reward_func):
            raise ValueError("reward_func must be a callable.")
        for name in ("build_prompt", "parse_completion"):
            if not callable(getattr(adapter, name, None)):
                raise TypeError(f"centralized_adapter.{name} must be callable.")
        self.centralized_collaboration = CentralizedCollaboration(
            adapter, formatters, reward_func, num_agents
        )
        return (
            self.centralized_collaboration,
            self.centralized_collaboration.build_prompt,
            1,
        )

    def _role_metrics_callback(self, callback):
        """Give a rollout-metrics callback per-role samples (no-op when decentralized)."""
        if callback is None or self.centralized_collaboration is None:
            return callback
        return self.centralized_collaboration.wrap_metrics_callback(callback)

    def _role_completions(self, agent_completions_turns, batch_items):
        """Split joint completions into per-role completions (no-op when decentralized)."""
        if self.centralized_collaboration is None:
            return agent_completions_turns
        return self.centralized_collaboration.split_eval(
            agent_completions_turns, batch_items
        )
