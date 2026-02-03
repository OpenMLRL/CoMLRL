from __future__ import annotations

"""
Runtime patches to adapt CoMLRL trainer behavior for memory efficiency and
single-agent (GRPO) compatibility without modifying external libraries.

Functions:
- patch_trainer_generation_for_memory(): reduce VRAM usage during MAGRPO generation.
- patch_maac_generation_for_memory(): reduce VRAM usage during MAAC generation.
- patch_iac_generation_for_memory(): reduce VRAM usage during IAC generation.
- patch_single_agent_returns(): provide GRPO flow when num_agents==1 and num_turns==1.
- patch_debug_turn_tracking(): attach turn index to batch items for debugging.
"""

import re

from comlrl.trainers.magrpo import MAGRPOTrainer  # type: ignore


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _run_with_generation_patches(actor_model, fn):
    """Run a callable with a temporary generate() wrapper to reduce VRAM."""
    orig_generate = getattr(actor_model, "generate", None)
    if not callable(orig_generate):
        return fn()

    def generate_wrapper(*args, **kwargs):
        kwargs.setdefault("output_scores", False)
        kwargs.setdefault("use_cache", False)
        return orig_generate(*args, **kwargs)

    try:
        actor_model.generate = generate_wrapper
        return fn()
    finally:
        actor_model.generate = orig_generate


def patch_trainer_generation_for_memory(
    *, use_memory: bool = True, force_sampling: bool = False
) -> None:
    orig = getattr(MAGRPOTrainer, "_generate_completions", None)
    if not callable(orig):
        return

    def wrapped(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=None,
        **kwargs,
    ):
        try:
            if use_memory:
                kwargs.setdefault("output_scores", False)
                kwargs.setdefault("use_cache", False)
            if force_sampling:
                kwargs["do_sample"] = True
                kwargs.setdefault("num_beams", 1)
                if "temperature" not in kwargs:
                    kwargs["temperature"] = _coerce_float(
                        getattr(self.args, "temperature", 1.0), 1.0
                    )
                if "top_p" not in kwargs:
                    kwargs["top_p"] = _coerce_float(
                        getattr(self.args, "top_p", 1.0), 1.0
                    )
                kwargs.setdefault("top_k", 50)

            import torch as _torch  # local import

            eff_max_new = (
                max_new_tokens
                if max_new_tokens is not None
                else _coerce_int(getattr(self.args, "max_new_tokens", 512), 512)
            )
            with _torch.no_grad():
                return orig(
                    self,
                    agent,
                    batch_items,
                    agent_idx=agent_idx,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=eff_max_new,
                    **kwargs,
                )
        except Exception:
            return orig(
                self,
                agent,
                batch_items,
                agent_idx=agent_idx,
                num_return_sequences=num_return_sequences,
                max_new_tokens=(
                    max_new_tokens
                    if max_new_tokens is not None
                    else getattr(self.args, "max_new_tokens", 512)
                ),
                **kwargs,
            )

    MAGRPOTrainer._generate_completions = wrapped  # type: ignore[attr-defined]


def patch_maac_generation_for_memory() -> None:
    from comlrl.trainers.maac import MAACTrainer  # type: ignore

    orig = getattr(MAACTrainer, "_generate", None)
    if not callable(orig):
        return

    def wrapped(self, actor_model, prompt):
        try:
            import torch as _torch  # local import

            with _torch.no_grad():
                return _run_with_generation_patches(
                    actor_model,
                    lambda: orig(self, actor_model, prompt),
                )
        except Exception:
            return orig(self, actor_model, prompt)

    MAACTrainer._generate = wrapped  # type: ignore[attr-defined]


def patch_iac_generation_for_memory() -> None:
    from comlrl.trainers.iac import IACTrainer  # type: ignore

    orig = getattr(IACTrainer, "_generate_rollout", None)
    if not callable(orig):
        return

    def wrapped(self, actor_model, prompt, agent_idx, num_ret):
        try:
            import torch as _torch  # local import

            with _torch.no_grad():
                return _run_with_generation_patches(
                    actor_model,
                    lambda: orig(self, actor_model, prompt, agent_idx, num_ret),
                )
        except Exception:
            return orig(self, actor_model, prompt, agent_idx, num_ret)

    IACTrainer._generate_rollout = wrapped  # type: ignore[attr-defined]


def patch_single_agent_returns() -> None:
    orig = getattr(MAGRPOTrainer, "_train_step_returns", None)
    if not callable(orig):
        return

    def wrapped(self, batch_item, epoch_turn_rewards, epoch_turn_returns, **kwargs):
        n_turns = _coerce_int(getattr(self.args, "num_turns", 1), 1)
        if self.num_agents != 1 or n_turns != 1:
            return orig(
                self, batch_item, epoch_turn_rewards, epoch_turn_returns, **kwargs
            )

        try:
            import numpy as _np  # type: ignore

            num_gens = _coerce_int(getattr(self.args, "num_generations", 2), 2)
            comps = self._generate_completions_with_external_prompts(
                self.agents[0],
                [batch_item],
                agent_idx=0,
                num_return_sequences=num_gens,
                max_new_tokens=getattr(self.args, "max_new_tokens", 128),
                external_prompts=None,
                **kwargs,
            )
            completions0 = comps.get("completions", [[]])[0]
            prompts0 = comps.get("prompts", [""])[0]
            rewards_vec = self._compute_rewards(
                [prompts0], [completions0], batch_items=[batch_item]
            )
            returns_vec = list(map(float, rewards_vec))

            self.optimizers[0].zero_grad()
            agent_loss = self._compute_loss_with_gradients(
                self.agents[0], comps, returns_vec
            )
            agent_loss.backward()
            self.optimizers[0].step()

            if epoch_turn_rewards and len(epoch_turn_rewards) > 0:
                epoch_turn_rewards[0].append(
                    float(_np.mean(rewards_vec)) if rewards_vec else 0.0
                )
            if epoch_turn_returns and len(epoch_turn_returns) > 0:
                epoch_turn_returns[0].append(
                    float(_np.mean(returns_vec)) if returns_vec else 0.0
                )

            batch_loss = float(_np.mean(_np.abs(returns_vec or [0.0])))
            stats = {
                "batch_mean_reward": (
                    float(_np.mean(rewards_vec)) if rewards_vec else 0.0
                ),
                "batch_expected_return": (
                    float(_np.mean(returns_vec)) if returns_vec else 0.0
                ),
            }
            return batch_loss, {0: stats}
        except Exception:
            return orig(
                self, batch_item, epoch_turn_rewards, epoch_turn_returns, **kwargs
            )
        return orig(self, batch_item, epoch_turn_rewards, epoch_turn_returns, **kwargs)

    MAGRPOTrainer._train_step_returns = wrapped  # type: ignore[attr-defined]


def patch_debug_turn_tracking(*, turn_key: str = "_turn_idx") -> None:
    orig = getattr(MAGRPOTrainer, "_generate_completions_with_external_prompts", None)
    if not callable(orig):
        return

    turn_re = re.compile(r"\bturn\s*[:#-]?\s*(\d+)\b", re.IGNORECASE)

    def wrapped(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=128,
        external_prompts=None,
        **kwargs,
    ):
        turn_idx = 1
        if external_prompts is not None:
            m = turn_re.search(str(external_prompts))
            turn_idx = int(m.group(1)) if m else 2
        for item in batch_items or []:
            if isinstance(item, dict):
                item[str(turn_key)] = int(turn_idx)
        return orig(
            self,
            agent,
            batch_items,
            agent_idx=agent_idx,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            external_prompts=external_prompts,
            **kwargs,
        )

    MAGRPOTrainer._generate_completions_with_external_prompts = wrapped  # type: ignore[attr-defined]
