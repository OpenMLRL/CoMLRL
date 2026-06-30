from __future__ import annotations

import copy
from typing import Any, List, Sequence

import torch
import torch.nn.functional as F

from comlrl.schedulers import DeviceScheduler
from comlrl.utils.distributed import unwrap_model


def reference_kl_enabled(args: Any) -> bool:
    return bool(getattr(args, "reference_kl_enabled", False))


def reference_kl_coef(args: Any) -> float:
    return float(getattr(args, "reference_kl_coef", 0.1))


def validate_reference_kl_config(args: Any, expected_count: int) -> None:
    coef = reference_kl_coef(args)
    if coef < 0:
        raise ValueError("reference_kl_coef must be >= 0.")
    reference_devices = getattr(args, "reference_devices", None)
    if reference_kl_enabled(args) and reference_devices is not None:
        DeviceScheduler.resolve_devices(
            reference_devices,
            expected_count,
            kind="reference_devices",
        )


def resolve_reference_devices(
    args: Any,
    fallback_devices: Sequence[torch.device],
    expected_count: int,
) -> List[torch.device]:
    reference_devices = getattr(args, "reference_devices", None)
    if reference_devices is None:
        return list(fallback_devices)
    return DeviceScheduler.resolve_devices(
        reference_devices,
        expected_count,
        kind="reference_devices",
    )


def clone_reference_models(
    policy_models: Sequence[Any],
    *,
    devices: Sequence[torch.device],
) -> List[Any]:
    references: List[Any] = []
    for idx, policy_model in enumerate(policy_models):
        reference_model = copy.deepcopy(policy_model)
        reference_model.to(devices[idx])
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        references.append(reference_model)
    return references


def response_token_logprobs(
    model: Any,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    module = unwrap_model(model)
    try:
        outputs = module(
            input_ids=sequences,
            attention_mask=attention_mask,
            output_values=False,
        )
    except TypeError:
        outputs = module(
            input_ids=sequences,
            attention_mask=attention_mask,
            return_dict=True,
        )
    logits = outputs.logits
    shifted_logits = logits[:, :-1, :]
    shifted_targets = sequences[:, 1:]
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shifted_targets.unsqueeze(-1)
    ).squeeze(-1)
    start_index = max(int(prompt_len) - 1, 0)
    end_index = start_index + int(response_len)
    return token_log_probs[:, start_index:end_index]


def reference_kl_for_sequence(
    policy_model: Any,
    reference_model: Any,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    """
    Return a non-negative sampled KL estimate for generated response tokens.

    Uses Schulman's k3 estimator per token:
    exp(log p_ref - log p_policy) - (log p_ref - log p_policy) - 1.
    """
    policy_module = unwrap_model(policy_model)
    reference_module = unwrap_model(reference_model)
    policy_device = next(policy_module.parameters()).device
    reference_device = next(reference_module.parameters()).device
    policy_seq = sequences.to(policy_device)
    policy_mask = attention_mask.to(policy_device)
    reference_seq = sequences.to(reference_device)
    reference_mask = attention_mask.to(reference_device)
    policy_training = bool(policy_module.training)
    reference_training = bool(reference_module.training)
    policy_module.eval()
    reference_module.eval()
    try:
        with torch.no_grad():
            policy_logps = response_token_logprobs(
                policy_model, policy_seq, policy_mask, prompt_len, response_len
            ).to(reference_device)
            reference_logps = response_token_logprobs(
                reference_model,
                reference_seq,
                reference_mask,
                prompt_len,
                response_len,
            )
            log_ratio_ref_policy = reference_logps - policy_logps
            token_kl = torch.exp(log_ratio_ref_policy) - log_ratio_ref_policy - 1.0
    finally:
        policy_module.train(policy_training)
        reference_module.train(reference_training)
    return token_kl.sum(dim=-1).detach().cpu()
