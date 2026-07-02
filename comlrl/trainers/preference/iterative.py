import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import wandb
from tqdm import tqdm  # type: ignore
from transformers import AutoModelForCausalLM

from comlrl.schedulers import DeviceScheduler
from comlrl.utils.distributed import unwrap_model
from comlrl.utils.model_loading import resolve_model_sources
from comlrl.utils.tokenizer_utils import apply_tokenizer_specials

from .madpo import (
    AgentPreferenceTensors,
    MADPOConfig,
    MADPOTrainer,
    PreferencePair,
)
from .marlhf import (
    MARLHFConfig,
    MARLHFTrainer,
    _ACTOR_CRITIC_ALGORITHMS,
    _MAGRPO_ADVANTAGE_MODES,
    _apply_magrpo_family_args,
)
from ..reinforce.magrpo import MAGRPOTrainer


def _normalize_comparator_policy(policy: Optional[str]) -> str:
    mode = str(policy or "current").strip().lower()
    if mode in {"current", "self", "online"}:
        return "current"
    if mode in {"model", "external", "comparator", "reference", "ref"}:
        return "model"
    raise ValueError(
        "comparator_policy must be one of: current, model, external, reference."
    )


def _validate_iterative_config(args: Any) -> None:
    if int(args.num_iterations) < 1:
        raise ValueError("num_iterations must be >= 1.")
    args.comparator_policy = _normalize_comparator_policy(args.comparator_policy)
    if args.comparator_num_candidates is not None:
        if int(args.comparator_num_candidates) < 1:
            raise ValueError("comparator_num_candidates must be >= 1 or null.")
    if args.comparator_policy == "model":
        if args.comparator_model_name is None and args.comparator_agents is None:
            raise ValueError(
                "comparator_model_name or comparator_agents is required when "
                "comparator_policy='model'."
            )


@dataclass
class MADPOIterConfig(MADPOConfig):
    """Configuration for iterative MADPO preference refresh."""

    num_iterations: int = 1
    comparator_policy: str = "current"
    comparator_model_name: Optional[str] = None
    comparator_agents: Optional[Sequence[str]] = None
    comparator_devices: Optional[Union[str, Sequence[str]]] = None
    comparator_num_candidates: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_iterative_config(self)


@dataclass
class MARLHFIterConfig(MARLHFConfig):
    """Configuration for iterative MARLHF preference and reward refresh."""

    num_iterations: int = 1
    comparator_policy: str = "current"
    comparator_model_name: Optional[str] = None
    comparator_agents: Optional[Sequence[str]] = None
    comparator_devices: Optional[Union[str, Sequence[str]]] = None
    comparator_num_candidates: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_iterative_config(self)


class MADPOIterTrainer(MADPOTrainer):
    """
    Iterative MADPO.

    Each iteration refreshes the preference dataset by comparing completions from
    the current policy against a comparator policy, then runs DPO updates on the
    newly generated pairs.
    """

    default_config_cls = MADPOIterConfig
    algorithm_name = "MADPOIter"

    def train(self, **kwargs):
        if int(self.args.num_turns) != 1:
            raise ValueError("MADPOIter currently supports num_turns=1 only.")

        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        for agent_idx, agent in enumerate(self.agents):
            agent.to(self.agent_devices[agent_idx])
            agent.train()

        updates_seen = 0
        total_pairs = 0
        for iteration_idx in range(int(self.args.num_iterations)):
            preference_pairs = self._build_preference_dataset(**kwargs)
            if not preference_pairs:
                if self.verbose:
                    print(
                        "MADPOIter: no non-tied preference pairs were generated "
                        f"for iteration {iteration_idx + 1}."
                    )
                continue

            total_pairs += len(preference_pairs)
            if self.wandb_initialized and wandb.run is not None:
                wandb.log(
                    {
                        "iter/iteration": float(iteration_idx + 1),
                        "iter/preference_pairs": float(len(preference_pairs)),
                    },
                    step=int(self.env_step),
                )

            updates_seen = self._train_preference_pairs(
                preference_pairs,
                iteration_idx=iteration_idx,
                updates_seen=updates_seen,
            )

        if total_pairs == 0 and self.verbose:
            print("MADPOIter: no non-tied preference pairs were generated.")

    def _train_preference_pairs(
        self,
        preference_pairs: List[PreferencePair],
        *,
        iteration_idx: int,
        updates_seen: int,
    ) -> int:
        for epoch in range(int(self.args.num_train_epochs)):
            random.shuffle(preference_pairs)
            batch_size = int(self.args.train_batch_size)
            batches = [
                preference_pairs[start : start + batch_size]
                for start in range(0, len(preference_pairs), batch_size)
            ]
            iterator = batches
            if self.verbose:
                iterator = tqdm(
                    batches,
                    total=len(batches),
                    desc=(
                        f"MADPOIter iteration {iteration_idx + 1}/"
                        f"{int(self.args.num_iterations)} epoch {epoch + 1}/"
                        f"{int(self.args.num_train_epochs)}"
                    ),
                )

            for batch in iterator:
                if int(self.args.eval_interval) > 0 and (
                    updates_seen % int(self.args.eval_interval) == 0
                ):
                    _ = self.evaluate(num_eval_samples=int(self.args.eval_num_samples))

                metrics = self._update_from_preference_batch(batch)
                metrics["iter/iteration"] = float(iteration_idx + 1)
                updates_seen += 1
                if self.args.use_environment_step:
                    self.env_step += int(self.args.environment_steps_per_pair) * len(
                        batch
                    )

                if self.wandb_initialized and wandb.run is not None:
                    if self._should_log_train(int(self.env_step)):
                        wandb.log(metrics, step=int(self.env_step))

        return updates_seen

    def _generate_preference_pairs_for_item(
        self,
        batch_item: Dict[str, Any],
        **kwargs,
    ) -> List[PreferencePair]:
        current_candidates = int(self.args.preference_num_candidates)
        comparator_candidates = int(
            self.args.comparator_num_candidates or current_candidates
        )

        current_outputs = self._generate_policy_outputs_for_item(
            self.agents,
            batch_item,
            num_candidates=current_candidates,
            **kwargs,
        )
        comparator_outputs = self._generate_policy_outputs_for_item(
            self._get_comparator_agents(),
            batch_item,
            num_candidates=comparator_candidates,
            **kwargs,
        )

        current_completions = [
            current_outputs[i]["completions"][0] for i in range(self.num_agents)
        ]
        comparator_completions = [
            comparator_outputs[i]["completions"][0] for i in range(self.num_agents)
        ]
        prompts = [current_outputs[i]["prompts"][0] for i in range(self.num_agents)]

        joint_mode = self.args.joint_mode.lower()
        if joint_mode not in {"align", "aligned"}:
            raise ValueError(
                "MADPOIter preference generation currently supports aligned joint_mode only."
            )

        current_rewards = self._compute_rewards(
            [prompts[0]],
            current_completions,
            batch_items=[batch_item],
        )
        comparator_rewards = self._compute_rewards(
            [prompts[0]],
            comparator_completions,
            batch_items=[batch_item],
        )
        selected_pairs = self._select_policy_comparison_pairs(
            current_rewards,
            comparator_rewards,
        )

        all_rewards = list(current_rewards) + list(comparator_rewards)
        candidate_reward_mean = float(np.mean(all_rewards)) if all_rewards else 0.0
        result: List[PreferencePair] = []

        for winner, loser in selected_pairs:
            winner_source, winner_idx = winner
            loser_source, loser_idx = loser
            winner_completions = self._completion_group(
                winner_source,
                winner_idx,
                current_completions,
                comparator_completions,
            )
            loser_completions = self._completion_group(
                loser_source,
                loser_idx,
                current_completions,
                comparator_completions,
            )
            winner_reward = self._reward_at(
                winner_source,
                winner_idx,
                current_rewards,
                comparator_rewards,
            )
            loser_reward = self._reward_at(
                loser_source,
                loser_idx,
                current_rewards,
                comparator_rewards,
            )

            agent_tensors = [
                self._preference_tensors_from_text(
                    agent_idx,
                    prompts[agent_idx],
                    winner_completions[agent_idx],
                    loser_completions[agent_idx],
                )
                for agent_idx in range(self.num_agents)
            ]
            result.append(
                PreferencePair(
                    prompts=list(prompts),
                    winner_completions=winner_completions,
                    loser_completions=loser_completions,
                    agent_tensors=agent_tensors,
                    winner_reward=float(winner_reward),
                    loser_reward=float(loser_reward),
                    candidate_reward_mean=candidate_reward_mean,
                )
            )

        return result

    def _generate_policy_outputs_for_item(
        self,
        policy_agents: Sequence[Any],
        batch_item: Dict[str, Any],
        *,
        num_candidates: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        def _generate_agent(agent_idx: int) -> Dict[str, Any]:
            return self._generate_completions_with_external_prompts(
                policy_agents[agent_idx],
                [batch_item],
                agent_idx=agent_idx,
                num_return_sequences=num_candidates,
                max_new_tokens=self.args.max_new_tokens,
                **kwargs,
            )

        return self._run_agent_tasks(_generate_agent)

    def _select_policy_comparison_pairs(
        self,
        current_rewards: Sequence[float],
        comparator_rewards: Sequence[float],
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        candidate_pairs: List[Tuple[float, Tuple[str, int], Tuple[str, int]]] = []
        for current_idx, current_reward in enumerate(current_rewards):
            for comparator_idx, comparator_reward in enumerate(comparator_rewards):
                gap = float(current_reward) - float(comparator_reward)
                if gap > 0:
                    candidate_pairs.append(
                        (
                            gap,
                            ("current", current_idx),
                            ("comparator", comparator_idx),
                        )
                    )
                elif gap < 0:
                    candidate_pairs.append(
                        (
                            abs(gap),
                            ("comparator", comparator_idx),
                            ("current", current_idx),
                        )
                    )

        if not candidate_pairs:
            return []

        mode = self.args.pair_selection
        if mode == "random":
            random.shuffle(candidate_pairs)
        elif mode == "all":
            candidate_pairs.sort(key=lambda item: (item[1], item[2]))
        else:
            candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        limit = self.args.preference_pairs_per_sample
        if limit is not None:
            candidate_pairs = candidate_pairs[: int(limit)]
        return [(winner, loser) for _, winner, loser in candidate_pairs]

    def _preference_tensors_from_text(
        self,
        agent_idx: int,
        prompt: str,
        winner_completion: str,
        loser_completion: str,
    ) -> AgentPreferenceTensors:
        tokenizer = self.tokenizers[agent_idx]
        agent_module = unwrap_model(self.agents[agent_idx])
        apply_tokenizer_specials(tokenizer, [agent_module])
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long,
        )
        winner_ids = torch.tensor(
            tokenizer.encode(winner_completion, add_special_tokens=False),
            dtype=torch.long,
        )
        loser_ids = torch.tensor(
            tokenizer.encode(loser_completion, add_special_tokens=False),
            dtype=torch.long,
        )
        return AgentPreferenceTensors(
            prompt_input_ids=prompt_ids.detach().cpu(),
            winner_completion_ids=winner_ids.detach().cpu(),
            loser_completion_ids=loser_ids.detach().cpu(),
        )

    def _get_comparator_agents(self) -> Sequence[Any]:
        if self.args.comparator_policy == "current":
            return self.agents
        if getattr(self, "_comparator_agents", None) is None:
            self._comparator_agents = self._load_comparator_agents()
        return self._comparator_agents

    def _load_comparator_agents(self) -> List[Any]:
        comparator_sources, _ = resolve_model_sources(
            kind="comparator_agents",
            model=self.args.comparator_model_name,
            models=self.args.comparator_agents,
            expected_count=self.num_agents,
            expected_label=f"num_agents ({self.num_agents})",
            model_label="comparator_model_name",
        )
        comparator_devices = DeviceScheduler.resolve_devices(
            self.args.comparator_devices or getattr(self.args, "agent_devices", None),
            self.num_agents,
            kind="comparator_devices",
        )

        model_kwargs: Dict[str, Any] = {}
        torch_dtype = None
        if isinstance(self.model_config, dict):
            torch_dtype = self.model_config.get("torch_dtype") or self.model_config.get(
                "dtype"
            )
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        if comparator_sources and all(
            isinstance(src, str) for src in comparator_sources
        ):
            comparator_agents = [
                AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
                for name in comparator_sources
            ]
        else:
            comparator_agents = list(comparator_sources)

        for agent_idx, comparator_agent in enumerate(comparator_agents):
            comparator_agent.to(comparator_devices[agent_idx])
            comparator_agent.eval()
            for param in comparator_agent.parameters():
                param.requires_grad = False
            apply_tokenizer_specials(self.tokenizers[agent_idx], [comparator_agent])

        return comparator_agents

    @staticmethod
    def _completion_group(
        source: str,
        index: int,
        current_completions: Sequence[Sequence[str]],
        comparator_completions: Sequence[Sequence[str]],
    ) -> List[str]:
        completions = (
            current_completions if source == "current" else comparator_completions
        )
        return [str(agent_completions[index]) for agent_completions in completions]

    @staticmethod
    def _reward_at(
        source: str,
        index: int,
        current_rewards: Sequence[float],
        comparator_rewards: Sequence[float],
    ) -> float:
        rewards = current_rewards if source == "current" else comparator_rewards
        return float(rewards[index])


class MARLHFIterTrainer(MADPOIterTrainer, MARLHFTrainer):
    """
    Iterative MARLHF.

    Each iteration refreshes preference pairs, retrains the joint reward model,
    then runs the configured online RL algorithm against that reward model.
    """

    default_config_cls = MARLHFIterConfig
    algorithm_name = "MARLHFIter"

    def train(self, **kwargs):
        if int(self.args.num_turns) != 1:
            raise ValueError("MARLHFIter currently supports num_turns=1 only.")

        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        total_pairs = 0
        for iteration_idx in range(int(self.args.num_iterations)):
            self._reward_model_active = False
            self._evaluating_with_task_reward = False
            preference_pairs = self._build_preference_dataset(**kwargs)
            if not preference_pairs:
                if self.verbose:
                    print(
                        "MARLHFIter: no non-tied preference pairs were generated "
                        f"for iteration {iteration_idx + 1}."
                    )
                continue

            total_pairs += len(preference_pairs)
            if self.wandb_initialized and wandb.run is not None:
                wandb.log(
                    {
                        "iter/iteration": float(iteration_idx + 1),
                        "iter/preference_pairs": float(len(preference_pairs)),
                    },
                    step=int(self.env_step),
                )

            self.reward_model = None
            self.reward_tokenizer = None
            self.reward_optimizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._init_reward_model()
            self._train_reward_model(preference_pairs)
            self._reward_model_active = True

            if self.args.rl_algorithm in _MAGRPO_ADVANTAGE_MODES:
                _apply_magrpo_family_args(self.args)
                self.advantage_mode = self.args.advantage_mode
                MAGRPOTrainer.train(self, **kwargs)
                continue

            if self.args.rl_algorithm in _ACTOR_CRITIC_ALGORITHMS:
                self._train_actor_critic_rl(**kwargs)
                continue

            raise ValueError(f"Unsupported rl_algorithm: {self.args.rl_algorithm}")

        if total_pairs == 0 and self.verbose:
            print("MARLHFIter: no non-tied preference pairs were generated.")
