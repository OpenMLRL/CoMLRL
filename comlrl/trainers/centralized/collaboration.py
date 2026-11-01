"""Translate a single joint policy's outputs at the environment boundary."""

from copy import copy
from functools import partial
from typing import Any, Dict, Sequence

import torch

from comlrl.utils.distributed import unwrap_model
from comlrl.utils.formatters import build_formatters
from comlrl.utils.reward_utils import call_reward_function, resolve_reward_range
from comlrl.utils.tokenizer_utils import apply_tokenizer_specials

from .adapters import CentralizedComparatorParseError


def joint_sequence_log_prob(agent, tokenizer, prompt_ids, completion_ids):
    """Score every joint-response token, with earlier roles in the causal context."""
    model = unwrap_model(agent)
    device = next(model.parameters()).device
    apply_tokenizer_specials(tokenizer, [model])
    # Generation runs in inference_mode; autograd needs ordinary index tensors.
    prompt_ids = prompt_ids.to(device).clone()
    completion_ids = completion_ids.to(device).clone()
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        pads = (completion_ids == pad_id).nonzero()
        if pads.numel():
            end = int(pads[0].item())
            if pad_id == tokenizer.eos_token_id:
                end += 1
            completion_ids = completion_ids[:end]
    if not completion_ids.numel():
        return torch.zeros((), device=device, requires_grad=True)
    if not prompt_ids.numel():
        raise ValueError("Centralized training requires a non-empty tokenized prompt.")
    model.train()
    input_ids = torch.cat([prompt_ids, completion_ids[:-1]]).unsqueeze(0)
    logits = model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
    ).logits[0, prompt_ids.numel() - 1 :]
    return logits.float().log_softmax(-1).gather(1, completion_ids[:, None]).sum()


def joint_policy_gradient_loss(
    trainer, agent, completions_data, returns, *, sequence_log_prob=None
):
    """Use the same joint-response objective for task-reward and learned-reward RL."""
    device = next(unwrap_model(agent).parameters()).device
    sequences = completions_data["completion_input_ids"][0]
    if len(sequences) != len(returns):
        raise ValueError("Joint completions must align with their returns.")
    if not len(returns):
        return torch.zeros((), device=device, requires_grad=True)
    if sequence_log_prob is None:
        sequence_log_prob = partial(
            joint_sequence_log_prob, agent, trainer.tokenizers[0]
        )
    returns_tensor = torch.as_tensor(returns, dtype=torch.float, device=device)
    effective_returns = trainer._apply_reference_kl_to_returns(
        returns_tensor, completions_data
    )
    advantages = trainer._compute_advantages(effective_returns)
    if trainer.args.advantage_normalization and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp(min=1e-6)
    prompt_ids = completions_data["prompt_input_ids"][0]
    losses = [
        -sequence_log_prob(prompt_ids, tokens) * advantage
        for tokens, advantage in zip(sequences, advantages)
    ]
    return torch.stack(losses).mean()


class CentralizedCollaboration:
    """Keep joint text for learning, split roles only for task rewards and metrics."""

    def __init__(self, adapter, formatters, reward_func, num_roles: int):
        self.adapter = adapter
        self.num_roles = num_roles
        self.formatters = build_formatters(
            formatters, num_roles, pass_none_external_prompts=False
        )
        self.reward_func = reward_func
        self.reward_range = resolve_reward_range(reward_func)

    def build_prompt(self, item: Dict[str, Any]) -> str:
        prompt = self.adapter.build_prompt(
            item, [formatter(item) for formatter in self.formatters]
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Centralized adapter must build a non-empty prompt.")
        return prompt

    def split(self, completion: str, item: Dict[str, Any]):
        try:
            outputs = self.adapter.parse_completion(completion, item, self.num_roles)
        except CentralizedComparatorParseError:
            outputs = [""] * self.num_roles
        if isinstance(outputs, (str, bytes)):
            raise TypeError("Centralized adapter must return a sequence of strings.")
        outputs = list(outputs)
        if len(outputs) != self.num_roles:
            raise ValueError(
                f"Centralized adapter must return exactly {self.num_roles} outputs."
            )
        if not all(isinstance(output, str) for output in outputs):
            raise TypeError("Centralized adapter outputs must all be strings.")
        return outputs

    def __call__(self, completions, *, prompts=None, batch_items=None):
        if not batch_items:
            raise ValueError("Centralized task rewards require batch_items.")
        if len(batch_items) not in {1, len(completions)}:
            raise ValueError("Centralized completions must align with batch_items.")
        rewards = []
        for idx, completion in enumerate(completions):
            item = batch_items[0] if len(batch_items) == 1 else batch_items[idx]
            outputs = self.split(completion, item)
            values = call_reward_function(
                self.reward_func,
                [self.formatters[0](item)],
                [[output] for output in outputs],
                num_agents=self.num_roles,
                batch_items=[item],
            )
            if len(values) != 1:
                raise ValueError(
                    "Centralized task reward must return one joint reward."
                )
            rewards.append(values[0])
        return rewards

    def split_eval(self, joint_turns, items: Sequence[Dict[str, Any]]):
        if len(joint_turns) != 1 or len(joint_turns[0]) != len(items):
            raise ValueError("Centralized evaluation outputs must align with items.")
        outputs = [[] for _ in range(self.num_roles)]
        for turns, item in zip(joint_turns[0], items):
            parsed_turns = [self.split(completion, item) for completion in turns]
            for role_idx in range(self.num_roles):
                outputs[role_idx].append([row[role_idx] for row in parsed_turns])
        return outputs

    def wrap_metrics_callback(self, callback):
        def role_metrics(rollouts):
            role_samples = []
            for sample in rollouts:
                item = sample.metadata["batch_item"]
                for role_idx, completion in enumerate(
                    self.split(sample.completion, item)
                ):
                    role_sample = copy(sample)
                    role_sample.agent_idx = role_idx
                    role_sample.completion = completion
                    role_samples.append(role_sample)
            return callback(role_samples)

        return role_metrics
