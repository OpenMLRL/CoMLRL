import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm  # type: ignore

from comlrl.utils.distributed import unwrap_model
from comlrl.utils.reference_kl import validate_reference_kl_config
from comlrl.utils.tokenizer_utils import apply_tokenizer_specials

from ..reinforce.magrpo import MAGRPOTrainer


@dataclass
class AgentPreferenceTensors:
    prompt_input_ids: torch.Tensor
    winner_completion_ids: torch.Tensor
    loser_completion_ids: torch.Tensor


@dataclass
class PreferencePair:
    prompts: List[str]
    winner_completions: List[str]
    loser_completions: List[str]
    agent_tensors: List[AgentPreferenceTensors]
    winner_reward: float
    loser_reward: float
    candidate_reward_mean: float
    raw_rewards: Optional[List[float]] = None
    target_raw_reward: Optional[float] = None
    comparator_raw_reward: Optional[float] = None


@dataclass
class MADPOConfig:
    """Configuration for multi-agent direct preference optimization."""

    # Core setup
    num_train_epochs: int = 20
    agent_learning_rate: float = 5.0e-6
    logging_steps: int = 50
    num_agents: int = 2
    parallel_training: str = "none"
    agent_devices: Optional[Union[str, Sequence[str]]] = None

    # Sampling/generation
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.6
    top_k: Optional[int] = 50

    # Shared trainer/eval fields used by the MAGRPO utility base.
    num_turns: int = 1
    discount: float = 1.0
    joint_mode: str = "aligned"
    early_termination_threshold: Optional[float] = None
    external_prompt_passthrough: bool = False
    eval_interval: int = 16
    eval_num_samples: int = 4
    eval_batch_size: int = 1
    rollout_buffer_size: int = 2
    train_batch_size: Optional[int] = None
    advantage_normalization: bool = True
    advantage_mode: str = "mean"
    reference_kl_enabled: bool = False
    reference_kl_coef: float = 0.1
    reference_devices: Optional[Union[str, Sequence[str]]] = None

    preference_num_candidates: int = 80
    preference_pairs_per_sample: Optional[int] = 16
    pair_selection: str = "reward_gap"
    dpo_beta: float = 0.1
    use_environment_step: bool = True
    environment_steps_per_pair: int = 2

    def __post_init__(self) -> None:
        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be >= 1.")
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1.")
        if self.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")
        if self.eval_interval < 0:
            raise ValueError("eval_interval must be >= 0.")
        if self.eval_num_samples < 0:
            raise ValueError("eval_num_samples must be >= 0.")
        if self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be >= 1.")
        if self.num_turns < 1:
            raise ValueError("num_turns must be >= 1.")
        if self.logging_steps < 1:
            raise ValueError("logging_steps must be >= 1.")
        if self.train_batch_size is None:
            self.train_batch_size = self.rollout_buffer_size
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1.")
        mode = str(self.parallel_training or "none").strip().lower()
        if mode == "null":
            mode = "none"
        if mode not in {"none", "mp"}:
            raise ValueError("parallel_training only supports: none, mp.")
        if mode == "mp" and self.agent_devices is None:
            raise ValueError("parallel_training='mp' requires explicit agent_devices.")
        self.parallel_training = mode
        validate_reference_kl_config(self, self.num_agents)
        if self.num_turns != 1:
            raise ValueError("MADPO currently supports num_turns=1 only.")
        if self.preference_num_candidates < 2:
            raise ValueError("preference_num_candidates must be >= 2.")
        if (
            self.preference_pairs_per_sample is not None
            and self.preference_pairs_per_sample < 1
        ):
            raise ValueError("preference_pairs_per_sample must be >= 1 or null.")
        if self.dpo_beta <= 0:
            raise ValueError("dpo_beta must be > 0.")
        if self.environment_steps_per_pair < 1:
            raise ValueError("environment_steps_per_pair must be >= 1.")
        mode = str(self.pair_selection or "reward_gap").strip().lower()
        allowed_modes = self._allowed_pair_selection_modes()
        if mode not in allowed_modes:
            raise ValueError(
                "pair_selection must be one of: " f"{', '.join(allowed_modes)}."
            )
        if mode == "all" and self.preference_pairs_per_sample is not None:
            raise ValueError(
                "preference_pairs_per_sample must be null when " "pair_selection='all'."
            )
        self.pair_selection = mode

    def _allowed_pair_selection_modes(self) -> Tuple[str, ...]:
        return ("reward_gap", "all", "random")


class MADPOTrainer(MAGRPOTrainer):
    """
    Multi-Agent Direct Preference Optimization.

    Preference labels are generated from a joint scalar reward. The DPO loss is
    joint-factorized over agents, while each agent receives gradients only
    through its own sequence log-probabilities.
    """

    default_config_cls = MADPOConfig
    algorithm_name = "MADPO"

    def train(self, **kwargs):
        if int(self.args.num_turns) != 1:
            raise ValueError("MADPO currently supports num_turns=1 only.")

        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        for agent_idx, agent in enumerate(self.agents):
            agent.to(self.agent_devices[agent_idx])
            agent.train()

        preference_pairs = self._build_preference_dataset(**kwargs)
        if not preference_pairs:
            if self.verbose:
                print("MADPO: no non-tied preference pairs were generated.")
            return

        updates_seen = 0
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
                    desc=f"MADPO epoch {epoch + 1}/{int(self.args.num_train_epochs)}",
                )

            for batch in iterator:
                if int(self.args.eval_interval) > 0 and (
                    updates_seen % int(self.args.eval_interval) == 0
                ):
                    _ = self.evaluate(num_eval_samples=int(self.args.eval_num_samples))

                metrics = self._update_from_preference_batch(batch)
                updates_seen += 1
                if self.args.use_environment_step:
                    self.env_step += int(self.args.environment_steps_per_pair) * len(
                        batch
                    )

                if self.wandb_initialized and wandb.run is not None:
                    if self._should_log_train(int(self.env_step)):
                        wandb.log(metrics, step=int(self.env_step))

    def _build_preference_dataset(self, **kwargs) -> List[PreferencePair]:
        pairs: List[PreferencePair] = []
        dataloader = self.get_train_dataloader()
        iterator = dataloader
        if self.verbose:
            iterator = tqdm(
                dataloader,
                total=len(dataloader),
                desc="MADPO preference rollout",
            )

        for batch in iterator:
            batch_item = batch[0]
            pairs.extend(self._generate_preference_pairs_for_item(batch_item, **kwargs))

        return pairs

    def _generate_preference_pairs_for_item(
        self,
        batch_item: Dict[str, Any],
        **kwargs,
    ) -> List[PreferencePair]:
        num_candidates = int(self.args.preference_num_candidates)

        def _generate_agent(agent_idx: int) -> Dict[str, Any]:
            return self._generate_completions_with_external_prompts(
                self.agents[agent_idx],
                [batch_item],
                agent_idx=agent_idx,
                num_return_sequences=num_candidates,
                max_new_tokens=self.args.max_new_tokens,
                **kwargs,
            )

        comps_per_agent = self._run_agent_tasks(_generate_agent)
        agent_completions = [
            comps_per_agent[i]["completions"][0] for i in range(self.num_agents)
        ]
        prompts = [comps_per_agent[i]["prompts"][0] for i in range(self.num_agents)]

        joint_mode = self.args.joint_mode.lower()
        if joint_mode not in {"align", "aligned"}:
            raise ValueError(
                "MADPO preference generation currently supports aligned joint_mode only."
            )

        rewards = self._compute_rewards(
            [prompts[0]],
            agent_completions,
            batch_items=[batch_item],
        )
        selected_pairs = self._select_preference_pair_indices(rewards)
        result: List[PreferencePair] = []

        for winner_idx, loser_idx in selected_pairs:
            agent_tensors: List[AgentPreferenceTensors] = []
            winner_texts: List[str] = []
            loser_texts: List[str] = []
            for agent_idx in range(self.num_agents):
                comp_data = comps_per_agent[agent_idx]
                completion_ids = comp_data["completion_input_ids"][0]
                agent_tensors.append(
                    AgentPreferenceTensors(
                        prompt_input_ids=comp_data["prompt_input_ids"][0]
                        .detach()
                        .cpu(),
                        winner_completion_ids=completion_ids[winner_idx].detach().cpu(),
                        loser_completion_ids=completion_ids[loser_idx].detach().cpu(),
                    )
                )
                winner_texts.append(agent_completions[agent_idx][winner_idx])
                loser_texts.append(agent_completions[agent_idx][loser_idx])

            result.append(
                PreferencePair(
                    prompts=list(prompts),
                    winner_completions=winner_texts,
                    loser_completions=loser_texts,
                    agent_tensors=agent_tensors,
                    winner_reward=float(rewards[winner_idx]),
                    loser_reward=float(rewards[loser_idx]),
                    candidate_reward_mean=(float(np.mean(rewards)) if rewards else 0.0),
                )
            )

        return result

    def _select_preference_pair_indices(
        self, rewards: List[float]
    ) -> List[Tuple[int, int]]:
        candidate_pairs: List[Tuple[float, int, int]] = []
        for i, reward_i in enumerate(rewards):
            for j, reward_j in enumerate(rewards):
                if i == j:
                    continue
                gap = float(reward_i) - float(reward_j)
                if gap > 0:
                    candidate_pairs.append((gap, i, j))

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
        if mode != "all" and limit is not None:
            candidate_pairs = candidate_pairs[: int(limit)]
        return [(winner_idx, loser_idx) for _, winner_idx, loser_idx in candidate_pairs]

    def _update_from_preference_batch(
        self, batch: List[PreferencePair]
    ) -> Dict[str, float]:
        if not batch:
            return {}

        detached_deltas = self._detached_agent_deltas(batch)

        for agent_idx in range(self.num_agents):
            self.optimizers[agent_idx].zero_grad()
            agent_loss = self._agent_preference_loss(
                agent_idx,
                batch,
                detached_deltas,
            )
            agent_loss.backward()
            self.optimizers[agent_idx].step()

        return {
            "turn_1/reward_mean": float(
                np.mean([pair.candidate_reward_mean for pair in batch])
            ),
            "turn_1/expected_return": float(
                np.mean([pair.candidate_reward_mean for pair in batch])
            ),
        }

    def _detached_agent_deltas(
        self,
        batch: List[PreferencePair],
    ) -> List[List[float]]:
        rows: List[List[float]] = []
        for pair in batch:
            row: List[float] = []
            for agent_idx in range(self.num_agents):
                with torch.no_grad():
                    delta = self._agent_logprob_delta(agent_idx, pair)
                row.append(float(delta.detach().cpu().item()))
            rows.append(row)
        return rows

    def _agent_preference_loss(
        self,
        agent_idx: int,
        batch: List[PreferencePair],
        detached_deltas: List[List[float]],
    ) -> torch.Tensor:
        device = next(unwrap_model(self.agents[agent_idx]).parameters()).device
        losses: List[torch.Tensor] = []
        beta = float(self.args.dpo_beta)

        for pair_idx, pair in enumerate(batch):
            own_delta = self._agent_logprob_delta(agent_idx, pair)
            other_delta = (
                sum(detached_deltas[pair_idx]) - detached_deltas[pair_idx][agent_idx]
            )
            joint_delta = own_delta + torch.tensor(
                other_delta,
                dtype=own_delta.dtype,
                device=device,
            )
            losses.append(-F.logsigmoid(beta * joint_delta))

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = torch.stack(losses).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.1, device=device, requires_grad=True)
        return loss

    def _agent_logprob_delta(
        self,
        agent_idx: int,
        pair: PreferencePair,
    ) -> torch.Tensor:
        tensors = pair.agent_tensors[agent_idx]
        winner_logprob = self._sequence_log_prob(
            agent_idx,
            tensors.prompt_input_ids,
            tensors.winner_completion_ids,
        )
        loser_logprob = self._sequence_log_prob(
            agent_idx,
            tensors.prompt_input_ids,
            tensors.loser_completion_ids,
        )
        return winner_logprob - loser_logprob

    def _sequence_log_prob(
        self,
        agent_idx: int,
        prompt_input_ids: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        agent = self.agents[agent_idx]
        agent_module = unwrap_model(agent)
        device = next(agent_module.parameters()).device
        tokenizer = self.tokenizers[agent_idx]
        apply_tokenizer_specials(tokenizer, [agent_module])

        prompt_ids = prompt_input_ids.to(device)
        completion_tokens = completion_ids.to(device)
        completion_tokens = self._trim_completion_tokens(
            completion_tokens,
            getattr(tokenizer, "pad_token_id", None),
        )
        if completion_tokens.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        agent.train()
        input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
        attention_mask = torch.ones(len(input_ids), device=device)
        outputs = agent_module(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )
        completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 : -1, :]
        usable = min(completion_tokens.numel(), completion_logits.size(0))
        if usable <= 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        token_logits = completion_logits[:usable]
        target_ids = completion_tokens[:usable]
        token_log_probs = torch.log_softmax(token_logits, dim=-1).gather(
            1,
            target_ids.unsqueeze(1),
        )
        return token_log_probs.sum()

    @staticmethod
    def _trim_completion_tokens(
        completion_tokens: torch.Tensor,
        pad_token_id: Optional[int],
    ) -> torch.Tensor:
        if pad_token_id is None:
            return completion_tokens
        pad_positions = (completion_tokens == int(pad_token_id)).nonzero()
        if pad_positions.numel() == 0:
            return completion_tokens
        first_pad = int(pad_positions[0].item())
        return completion_tokens[:first_pad]
