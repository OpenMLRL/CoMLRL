from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@dataclass
class ActorCriticOutput:
    """Container for actor-critic forward passes."""

    logits: torch.Tensor
    values: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]


class CausalLMWithValueHead(nn.Module):
    """
    Wrap a causal language model with a small value head so one parameter
    set can serve both the policy (actor) and critic.
    """

    def __init__(
        self, model: PreTrainedModel, value_head_hidden_dim: Optional[int] = None
    ):
        super().__init__()

        self.model = model
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError(
                "Base model must expose a config with hidden size information."
            )

        hidden_size = getattr(config, "hidden_size", None) or getattr(
            config, "n_embd", None
        )
        if hidden_size is None:
            raise ValueError(
                "Base model config should define `hidden_size` (or `n_embd`). "
                "Unsupported model architecture for value head attachment."
            )

        head_dim = value_head_hidden_dim or hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, head_dim),
            nn.Tanh(),
            nn.Linear(head_dim, 1),
        )
        self._init_value_head()

    def _init_value_head(self) -> None:
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> ActorCriticOutput:
        outputs: CausalLMOutputWithCrossAttentions = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states).squeeze(-1)

        return ActorCriticOutput(
            logits=outputs.logits,
            values=values,
            hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
        )

    def generate(self, *args, **kwargs) -> torch.Tensor:
        """Passthrough generation to the underlying causal LM."""

        return self.model.generate(*args, **kwargs)
