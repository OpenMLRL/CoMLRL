import inspect
import os
import random
from dataclasses import dataclass
import itertools
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Type, Sequence

import numpy as np
import torch
import wandb
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from comlrl.utils.formatters import build_formatters
from comlrl.utils.model_loading import infer_model_name, resolve_model_sources
from comlrl.utils.reward_utils import call_reward_function
from comlrl.utils.tokenizer_utils import (
    apply_tokenizer_specials,
    resolve_tokenizers,
)


@dataclass
class MAGRPOConfig:
    """Configuration for MAGRPO training."""

    # Core setup
    num_train_epochs: int = 20
    agent_learning_rate: float = 5.0e-6
    logging_steps: int = 50
    num_agents: int = 2

    # Sampling/generation
    num_generations: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.6
    top_k: Optional[int] = 50

    # Multi-turn / tree rollout
    num_turns: int = 2
    discount: float = 0.9
    joint_mode: str = "aligned"
    early_termination_threshold: Optional[float] = -0.2
    external_prompt_passthrough: bool = False

    eval_interval: int = 16
    eval_num_samples: int = 4
    eval_batch_size: int = 1
    rollout_buffer_size: int = 2
    train_batch_size: Optional[int] = None
    advantage_normalization: bool = True
    advantage_mode: str = "mean"

    def __post_init__(self) -> None:
        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be >= 1.")
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1.")
        if self.num_generations < 2:
            raise ValueError(
                "num_generations must be >= 2 (group baseline requires multiple samples)."
            )
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


@dataclass
class NodeSample:
    agent_idx: int
    turn_idx: int
    completions_data: Dict[str, Any]
    returns: List[float]
    node_mean_reward: float
    node_mean_return: float
    node_env_step: int


class MAGRPOTrainer:
    """
    Multi-Agent Group Relative Policy Optimization Trainer (MAGRPO).
    Supports both single-turn and multi-turn training with external transitions.

    When num_turns=1, this trainer behaves as a standard MAGRPO trainer.
    When num_turns>1, it adds multi-turn capabilities with external transitions between turns.

    Args:
        agent_model: The model to be trained for homogeneous agents
        agents: List of agent models (alternative to agent_model)
        num_agents: The number of agents
        reward_func: Single reward function callable
        reward_processor: Optional processor to apply to the reward (e.g., scaling)
        formatters: Formatters to apply to dataset items for each agent
        args: The training arguments
        train_dataset: The training dataset
        eval_dataset: The evaluation dataset
        tokenizer: The tokenizer
        wandb_config: Configuration for Weights & Biases logging
        model_config: Model configuration dict
        eval_logger: Evaluation logger function
        eval_aggregator: Evaluation aggregator function
        external_transition: Function that provides external transitions between turns
        dataset_type: Optional explicit dataset type (e.g., "humaneval")
    """

    default_config_cls: Type[MAGRPOConfig] = MAGRPOConfig
    algorithm_name: str = "MAGRPO"

    def __init__(
        self,
        agent_model: Optional[Union[str, PreTrainedModel]] = None,
        agents: Optional[List[PreTrainedModel]] = None,
        num_agents: int = 2,
        tokenizer: Optional[
            Union[PreTrainedTokenizerBase, Sequence[PreTrainedTokenizerBase]]
        ] = None,
        model_config: Optional[Dict[str, Any]] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        dataset_type: Optional[str] = None,
        reward_func: Optional[Callable] = None,
        reward_processor: Optional[Callable[[float], float]] = None,
        formatters: Optional[Union[Callable, Sequence[Callable]]] = None,
        external_transition: Optional[Callable] = None,
        # Logging/eval
        wandb_config: Optional[Dict[str, Any]] = None,
        eval_logger: Optional[Callable] = None,
        eval_aggregator: Optional[Callable] = None,
        args: Optional[MAGRPOConfig] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args if args is not None else self.default_config_cls()
        if agent_model is None and agents is None:
            raise ValueError("Either agent_model or agents must be provided.")
        if (
            agents is None
            and agent_model is not None
            and not isinstance(agent_model, str)
        ):
            raise ValueError(
                "agent_model should be a string to create homogeneous agents"
            )
        self.env_step = 0
        self._last_train_log_step = -1
        self.advantage_mode = getattr(self.args, "advantage_mode", "mean")
        if str(self.advantage_mode).lower() not in {"mean", "raw", "max", "rloo"}:
            raise ValueError("advantage_mode must be one of: mean, raw, max, rloo.")

        self.formatters = build_formatters(
            formatters, num_agents, pass_none_external_prompts=False
        )
        if reward_func is None or not callable(reward_func):
            raise ValueError("reward_func must be a callable.")
        self.reward_func = reward_func
        self.reward_processor = (
            reward_processor if reward_processor is not None else (lambda x: x)
        )

        self.model_config = model_config if model_config else {}
        expected_count = num_agents
        if agents is not None and agent_model is None:
            expected_count = len(agents)

        actor_sources, model_name = resolve_model_sources(
            kind="agents",
            model=agent_model,
            models=agents,
            expected_count=expected_count,
            expected_label=f"num_agents ({expected_count})",
            model_label="agent_model",
        )
        if actor_sources and not all(isinstance(src, str) for src in actor_sources):
            model_name = infer_model_name(actor_sources[0])

        self.num_agents = expected_count
        self.model_name = model_name
        if actor_sources and all(isinstance(src, str) for src in actor_sources):
            from transformers import AutoModelForCausalLM

            model_kwargs = {}
            torch_dtype = None
            if isinstance(self.model_config, dict):
                torch_dtype = self.model_config.get(
                    "torch_dtype"
                ) or self.model_config.get("dtype")
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            self.agents = [
                AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
                for name in actor_sources
            ]
        else:
            self.agents = list(actor_sources)

        tokenizers = resolve_tokenizers(agent_model, tokenizer, actor_sources)
        if isinstance(tokenizers, list):
            self.tokenizers = tokenizers
            self.tokenizer = tokenizers[0] if tokenizers else None
        else:
            self.tokenizers = [tokenizers] * self.num_agents
            self.tokenizer = tokenizers
        special_tokens = self.model_config.get("special_tokens", {})
        if special_tokens:
            for tok in self.tokenizers:
                tok.add_special_tokens(special_tokens)

        # Allow single-agent as a special case (GRPO)
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self.args.num_generations < 2:
            raise ValueError(
                "num_generations must be >= 2 (group baseline requires multiple samples)."
            )
        if getattr(self.args, "eval_batch_size", 1) < 1:
            raise ValueError("eval_batch_size must be >= 1.")
        if self.args.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")

        self.rollout_buffers: List[List[NodeSample]] = [
            [] for _ in range(self.num_agents)
        ]

        # Check for external_transition requirement in multi-turn training
        if self.args.num_turns > 1 and external_transition is None:
            raise ValueError("Multi-turn MAGRPO requires an external_transition.")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.eval_logger = eval_logger
        self.eval_aggregator = eval_aggregator
        self.external_transition = external_transition

        self.optimizers = [
            torch.optim.AdamW(
                agent.parameters(),
                lr=self.args.agent_learning_rate,
            )
            for agent in self.agents
        ]

        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if self.wandb_config is not None:
            self._init_wandb()

        self.dataset_type = dataset_type or None
        if self.dataset_type is None:
            try:
                if isinstance(self.wandb_config, dict):
                    sections = self.wandb_config.get("config_sections", {})
                    if isinstance(sections, dict):
                        ds = sections.get("dataset", {})
                        if isinstance(ds, dict):
                            self.dataset_type = ds.get("type")
            except Exception:
                self.dataset_type = None

        # Verbosity from config (default True)
        self.verbose = True
        if isinstance(self.wandb_config, dict):
            sections = self.wandb_config.get("config_sections", {})
            if isinstance(sections, dict):
                out = sections.get("output", {})
                if isinstance(out, dict) and "verbose" in out:
                    self.verbose = bool(out.get("verbose"))

    def _init_wandb(self):
        """Initialize Weights & Biases for tracking with multi-turn config."""
        if not self.wandb_initialized:
            if self.wandb_config is None:
                self.wandb_config = {}

            wandb_project = self.wandb_config.get("project", "comlrl")
            wandb_entity = self.wandb_config.get("entity")

            # Use different default names based on num_turns and algorithm
            algo_tag = str(self.algorithm_name or "magrpo").lower()
            wandb_name = (
                self.wandb_config.get("name")
                or self.wandb_config.get("run_name")
                or f"test-{algo_tag}"
            )

            wandb_dir = self.wandb_config.get("dir", None)

            config_dict = {
                "model_name": self.model_name,
                "num_agents": self.num_agents,
                "num_turns": self.args.num_turns,
                "algorithm": self.algorithm_name,
                "advantage_mode": self.advantage_mode,
                "advantage_normalization": self.args.advantage_normalization,
                "agent_learning_rate": self.args.agent_learning_rate,
                "num_train_epochs": self.args.num_train_epochs,
                "num_generations": self.args.num_generations,
                "max_new_tokens": self.args.max_new_tokens,
            }

            # No per-turn weighting or early termination config

            # Incorporate full config sections and derived fields for searchability
            sections = (
                self.wandb_config.get("config_sections")
                if isinstance(self.wandb_config, dict)
                else None
            )
            if isinstance(sections, dict):
                dataset_section = sections.get("dataset") or {}
                output_section = sections.get("output") or {}
                external_section = sections.get("external") or {}
                trainer_section = sections.get("trainer") or {}

                # Attach full sections
                config_dict.update(
                    {
                        "dataset": dataset_section,
                        "output": output_section,
                        "external": external_section,
                        "trainer": trainer_section,
                    }
                )

                # Derived convenience keys
                dataset_name = (
                    dataset_section.get("name")
                    if isinstance(dataset_section, dict)
                    else None
                )
                dataset_type = (
                    dataset_section.get("type")
                    if isinstance(dataset_section, dict)
                    else None
                )
                if dataset_name:
                    config_dict["dataset_name"] = dataset_name
                if dataset_type:
                    config_dict["dataset_type"] = dataset_type

                ext_mode = (
                    external_section.get("mode")
                    if isinstance(external_section, dict)
                    else None
                )
                if ext_mode:
                    config_dict["external_mode"] = ext_mode
                    if "original_prompt" in external_section:
                        config_dict["original_prompt"] = external_section.get(
                            "original_prompt"
                        )
                    if "previous_response" in external_section:
                        config_dict["previous_response"] = external_section.get(
                            "previous_response"
                        )

            init_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": wandb_name,
                "config": config_dict,
            }

            if wandb_dir is not None:
                os.makedirs(wandb_dir, exist_ok=True)
                init_kwargs["dir"] = wandb_dir

            # Optionally support tags if provided by caller
            tags = (
                self.wandb_config.get("tags")
                if isinstance(self.wandb_config, dict)
                else None
            )
            if isinstance(tags, list):
                init_kwargs["tags"] = tags

            wandb.init(**init_kwargs)
            self.wandb_initialized = True

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=1,
            collate_fn=lambda examples: examples,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """Returns the evaluation DataLoader."""
        if self.eval_dataset is None:
            return None

        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

    def evaluate(self, num_eval_samples: int = 4) -> Dict[str, float]:
        """
        Unified evaluation that supports both single-turn and multi-turn.

        Args:
            num_eval_samples: Number of samples to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.eval_dataset is None:
            return {}

        # Storage for completions across turns for all agents
        all_agent_completions_turns = [[] for _ in range(self.num_agents)]
        all_test_cases = []
        all_entry_points = []
        all_prompts = []
        # Collect per-turn immediate rewards across evaluated samples
        eval_turn_rewards: List[List[float]] = [[] for _ in range(self.args.num_turns)]
        # No per-function tracking; single reward function handles composition

        # Get evaluation dataloader
        eval_dataloader = self.get_eval_dataloader()

        # Evaluate on specified number of samples
        seen = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                for batch_item in batch:
                    self._evaluate_sample(
                        batch_item,
                        all_agent_completions_turns,
                        all_test_cases,
                        all_entry_points,
                        all_prompts,
                        eval_turn_rewards,
                    )
                    seen += 1
                    if seen >= num_eval_samples:
                        break
                if seen >= num_eval_samples:
                    break

        extra_eval_metrics: Dict[str, Any] = {}

        n_turns = self.args.num_turns
        if n_turns > 0 and eval_turn_rewards and eval_turn_rewards[0]:
            n_samp = len(eval_turn_rewards[0])
            gamma = float(getattr(self.args, "discount", 0.9))
            sum_returns = [0.0] * n_turns
            for s in range(n_samp):
                rs = [
                    eval_turn_rewards[t][s] if s < len(eval_turn_rewards[t]) else 0.0
                    for t in range(n_turns)
                ]
                ret = [0.0] * n_turns
                ret[-1] = rs[-1]
                for t in range(n_turns - 2, -1, -1):
                    ret[t] = rs[t] + gamma * ret[t + 1]
                for t in range(n_turns):
                    sum_returns[t] += ret[t]
            for t in range(n_turns):
                extra_eval_metrics[f"eval/turn_{t+1}/mean_reward"] = float(
                    np.mean(eval_turn_rewards[t]) if eval_turn_rewards[t] else 0.0
                )
                extra_eval_metrics[f"eval/turn_{t+1}/mean_return"] = float(
                    sum_returns[t] / n_samp if n_samp > 0 else 0.0
                )

        # No per-reward-function logging when using a single reward function

        # Calculate and log metrics (including extra_eval_metrics)
        eval_metrics = self._log_eval_metrics(
            all_agent_completions_turns,
            all_test_cases,
            all_entry_points,
            all_prompts,
            extra_metrics=extra_eval_metrics,
        )
        return eval_metrics

    def _evaluate_sample(
        self,
        batch_item,
        all_agent_completions_turns,
        all_test_cases,
        all_entry_points,
        all_prompts,
        eval_turn_rewards,
        # no per-function component tracking
    ):
        """Evaluate a single sample for any number of turns."""
        # Storage for each agent's completions across turns
        agent_sample_completions = [[] for _ in range(self.num_agents)]

        # Store sample information
        all_test_cases.append(batch_item.get("test", ""))
        all_entry_points.append(batch_item.get("entry_point", ""))
        all_prompts.append(batch_item.get("prompt", ""))

        # Track the selected completions from the previous turn (evaluation traces a single path)
        previous_turn_completions = [None] * self.num_agents
        # Track full history per agent for evaluation path
        eval_prompt_history = [[] for _ in range(self.num_agents)]
        eval_response_history = [[] for _ in range(self.num_agents)]

        # Run episode with configured number of turns
        for turn_idx in range(self.args.num_turns):
            # Prepare external prompts for turns after the first
            agent_external_prompts = [None] * self.num_agents

            if turn_idx > 0 and all(c is not None for c in previous_turn_completions):
                # Use previously selected completions to form next-turn prompts (single eval path)
                selected_prev = list(previous_turn_completions)
                # Get external transitions based on selected prior completions
                if self.external_transition is not None:
                    transition_result = self.external_transition(
                        prompt=batch_item.get("prompt", ""),
                        agent_completions=selected_prev,
                        num_agents=self.num_agents,
                        prompt_history_per_agent=eval_prompt_history,
                        response_history_per_agent=[
                            list(eval_response_history[i])
                            for i in range(self.num_agents)
                        ],
                    )

                    if isinstance(transition_result, (list, tuple)):
                        if len(transition_result) != self.num_agents:
                            raise ValueError(
                                f"External transition returned {len(transition_result)} values but expected {self.num_agents}"
                            )
                        agent_external_prompts = list(transition_result)
                    else:
                        raise ValueError(
                            "External transition must return a list or tuple of external prompts for each agent"
                        )

            # Generate and extract one completion from each agent for evaluation
            for agent_idx in range(self.num_agents):
                agent_completions = self._generate_completions_with_external_prompts(
                    self.agents[agent_idx],
                    [batch_item],
                    agent_idx=agent_idx,
                    num_return_sequences=1,
                    max_new_tokens=self.args.max_new_tokens,
                    external_prompts=agent_external_prompts[agent_idx],
                )
                # Extract the completion directly
                completion = agent_completions["completions"][0][0]
                # Record prompt used this turn
                used_prompt = agent_completions["prompts"][0]
                eval_prompt_history[agent_idx].append(used_prompt)
                agent_sample_completions[agent_idx].append(completion)

            # Compute immediate reward at this turn (single joint sample)
            agent_completions_for_reward = [
                [agent_sample_completions[i][-1]] for i in range(self.num_agents)
            ]
            prompt = self.formatters[0](batch_item)
            rewards = self._compute_rewards(
                [prompt], agent_completions_for_reward, batch_items=[batch_item]
            )
            if rewards:
                # Track per-turn reward across samples
                eval_turn_rewards[turn_idx].append(float(rewards[0]))
                # Update selected previous-turn completions for next-turn prompts
                for agent_idx in range(self.num_agents):
                    chosen = agent_sample_completions[agent_idx][-1]
                    previous_turn_completions[agent_idx] = chosen
                    eval_response_history[agent_idx].append(chosen)

        # Store completions for all agents
        for agent_idx in range(self.num_agents):
            all_agent_completions_turns[agent_idx].append(
                agent_sample_completions[agent_idx]
            )

    def _log_eval_metrics(
        self,
        all_agent_completions_turns,
        all_test_cases,
        all_entry_points,
        all_prompts,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Log evaluation metrics for any number of turns."""
        eval_metrics = {}

        # Detailed logging (if logger is provided), standardized to modern interface
        if (
            self.eval_logger is not None
            and self.eval_aggregator is not None
            and all_agent_completions_turns
            and all(agent_comps for agent_comps in all_agent_completions_turns)
        ):
            detailed_metrics = self.eval_logger(
                agent_completions_turns=all_agent_completions_turns,
                test_cases=all_test_cases,
                entry_points=all_entry_points,
                prompts=all_prompts,
            )

            # Aggregate metrics for logging
            # Aggregate strictly per-turn; aggregator already returns turn_k/* keys only
            aggregated_detailed_metrics = self.eval_aggregator(
                detailed_metrics, num_turns=self.args.num_turns
            )
            for key, value in aggregated_detailed_metrics.items():
                eval_metrics[f"eval/{key}"] = value

        # Merge any extra metrics (already with full key prefixes like 'eval/...')
        if isinstance(extra_metrics, dict) and extra_metrics:
            eval_metrics.update(extra_metrics)

        # Log evaluation metrics
        if self.wandb_initialized and wandb.run is not None:
            wandb.log(eval_metrics, step=self.env_step)

        return eval_metrics

    def train(self, **kwargs):
        """
        Unified train method that supports both single-turn and multi-turn training.
        """
        # Initialize wandb if not already done
        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        device = self.device
        for agent in self.agents:
            agent.to(device)
            agent.train()

        # Create the data pipeline for generating examples
        for epoch in range(0, int(self.args.num_train_epochs)):
            # No per-agent reward tracking in single reward mode

            # Turn tracking for all cases (including single-turn)
            epoch_turn_rewards = [
                [] for _ in range(self.args.num_turns)
            ]  # immediate rewards
            epoch_turn_returns = [[] for _ in range(self.args.num_turns)]  # returns
            dl = self.get_train_dataloader()
            if getattr(self, "verbose", True):
                it = enumerate(
                    tqdm(
                        dl,
                        total=len(dl),
                        desc=f"Epoch {epoch+1}/{int(self.args.num_train_epochs)}",
                    )
                )
            else:
                it = enumerate(dl)
            for batch_idx, batch in it:
                # Periodic evaluation based on configuration
                if int(self.args.eval_interval) > 0 and (
                    batch_idx % int(self.args.eval_interval) == 0
                ):
                    # evaluate() already logs its metrics; avoid duplicate logging here
                    _ = self.evaluate(num_eval_samples=int(self.args.eval_num_samples))

                # Process single batch item (batch_size=1 enforced)
                batch_item = batch[0]
                # Unified training step (returns-based, backward updates)
                batch_loss, _batch_stats = self._train_step_returns(
                    batch_item,
                    epoch_turn_rewards,
                    epoch_turn_returns,
                    **kwargs,
                )

            for agent_idx, buffer in enumerate(self.rollout_buffers):
                if buffer:
                    self._process_buffer(agent_idx, buffer)

            # Log per-turn epoch averages inline (avoid custom system/* metrics)
            if self.wandb_initialized and wandb.run is not None:
                epoch_log: Dict[str, Any] = {}
                n_turns = max(1, int(self.args.num_turns))
                for turn_idx in range(n_turns):
                    if epoch_turn_rewards and epoch_turn_rewards[turn_idx]:
                        epoch_log[f"turn_{turn_idx + 1}/epoch_reward_mean"] = float(
                            np.mean(epoch_turn_rewards[turn_idx])
                        )
                    if epoch_turn_returns and epoch_turn_returns[turn_idx]:
                        epoch_log[f"turn_{turn_idx + 1}/epoch_avg_return"] = float(
                            np.mean(epoch_turn_returns[turn_idx])
                        )
                if epoch_log:
                    wandb.log(epoch_log, step=self.env_step)

    def _train_step_returns(
        self,
        batch_item,
        epoch_turn_rewards,
        epoch_turn_returns,
        **kwargs,
    ):
        """Branching rollout with returns; updates backward from last turn to first.

        Returns an additional per-turn batch summary for logging:
        - batch_mean_reward (immediate reward mean averaged across nodes at the turn)
        - batch_expected_return (expected return averaged across nodes at the turn)
        - no per-function breakdown (single reward function)
        - levels (code-only: mean of level_1/2/3 and bonus across nodes)
        """
        num_turns = int(self.args.num_turns)
        num_gens = int(self.args.num_generations)
        gamma = float(getattr(self.args, "discount", 0.9))

        # Per-turn accumulators for batch-level summaries
        turn_reward_node_means: List[List[float]] = [[] for _ in range(num_turns)]
        turn_return_node_means: List[List[float]] = [[] for _ in range(num_turns)]
        # No per-function accumulation in single reward mode
        turn_node_counts: List[int] = [0 for _ in range(num_turns)]

        def build_node(
            turn_idx: int,
            prompts_per_agent=None,
            prompt_history_per_agent: Optional[List[List[str]]] = None,
            response_history_per_agent: Optional[List[List[str]]] = None,
        ):
            comps_per_agent = []
            for agent_idx in range(self.num_agents):
                comps = self._generate_completions_with_external_prompts(
                    self.agents[agent_idx],
                    [batch_item],
                    agent_idx=agent_idx,
                    num_return_sequences=num_gens,
                    max_new_tokens=self.args.max_new_tokens,
                    external_prompts=(
                        prompts_per_agent[agent_idx] if prompts_per_agent else None
                    ),
                    **kwargs,
                )
                comps_per_agent.append(comps)

            agent_completions_list = [
                comps_per_agent[i]["completions"][0] for i in range(self.num_agents)
            ]
            # Prompts actually used this turn, per agent (may differ across agents)
            prompts_used_this_turn = [
                comps_per_agent[i]["prompts"][0] for i in range(self.num_agents)
            ]
            formatted_prompt = comps_per_agent[0]["prompts"][0]

            # Initialize history containers if not provided
            if prompt_history_per_agent is None:
                prompt_history_per_agent = [[] for _ in range(self.num_agents)]
            if response_history_per_agent is None:
                response_history_per_agent = [[] for _ in range(self.num_agents)]

            # Extend prompt history with this turn's prompts
            next_prompt_history = [
                list(prompt_history_per_agent[i]) + [prompts_used_this_turn[i]]
                for i in range(self.num_agents)
            ]
            # Compute rewards per joint action depending on joint_mode
            joint_mode = str(getattr(self.args, "joint_mode", "aligned")).lower()
            rewards_vec: List[float] = []
            combo_indices: List[Tuple[int, ...]] = []
            if joint_mode in ["cross", "crossed"] and self.num_agents > 1:
                # Cartesian product of per-agent completion indices
                per_agent_ranges = [
                    range(len(agent_completions_list[i]))
                    for i in range(self.num_agents)
                ]
                for idx_tuple in itertools.product(*per_agent_ranges):
                    # Build per-agent single-element lists
                    completion_args = [
                        [agent_completions_list[a][idx_tuple[a]]]
                        for a in range(self.num_agents)
                    ]
                    # Call reward function for this joint action
                    try:
                        sig = inspect.signature(self.reward_func)
                        if "batch_items" in sig.parameters:
                            rlist = self.reward_func(
                                *completion_args, batch_items=[batch_item]
                            )
                        else:
                            rlist = self.reward_func(*completion_args)
                    except TypeError:
                        rlist = self.reward_func(
                            [
                                agent_completions_list[a][idx_tuple[a]]
                                for a in range(self.num_agents)
                            ]
                        )
                    # Apply processor
                    processed = [self.reward_processor(r) for r in rlist]
                    rewards_vec.append(float(processed[0] if processed else 0.0))
                    combo_indices.append(tuple(idx_tuple))
            elif joint_mode in ["align", "aligned"] and self.num_agents > 1:
                # Aligned by index
                rewards_vec = self._compute_rewards(
                    [formatted_prompt], agent_completions_list, batch_items=[batch_item]
                )
                # combo indices: align j with (j,j,...)
                k = len(agent_completions_list[0]) if agent_completions_list else 0
                combo_indices = [tuple([j] * self.num_agents) for j in range(k)]
            elif self.num_agents == 1:
                # Single-agent mode (GRPO)
                rewards_vec = self._compute_rewards(
                    [formatted_prompt], agent_completions_list, batch_items=[batch_item]
                )
                # combo indices: single agent, each completion gets its own index
                k = len(agent_completions_list[0]) if agent_completions_list else 0
                combo_indices = [tuple([j]) for j in range(k)]
            else:
                raise ValueError(f"Unsupported joint_mode: {joint_mode}")

            if 0 <= turn_idx < len(epoch_turn_rewards):
                epoch_turn_rewards[turn_idx].append(
                    np.mean(rewards_vec) if rewards_vec else 0.0
                )

            # Per-node means for batch-level summaries
            self.env_step += len(rewards_vec)
            node_env_step = int(self.env_step)
            node_mean_reward = float(np.mean(rewards_vec)) if rewards_vec else 0.0
            turn_reward_node_means[turn_idx].append(node_mean_reward)

            turn_node_counts[turn_idx] += 1

            # Early termination: stop expanding this branch if mean reward exceeds threshold
            term_threshold = getattr(self.args, "early_termination_threshold", None)
            terminate_here = False
            if term_threshold is not None and rewards_vec:
                try:
                    terminate_here = float(np.mean(rewards_vec)) > float(term_threshold)
                except Exception:
                    terminate_here = False

            node = {
                "turn": turn_idx,
                "completions": comps_per_agent,
                "rewards": rewards_vec,
                "children": [],
                "returns": None,
                "combo_indices": combo_indices,
                "env_step": node_env_step,
            }

            if turn_idx < num_turns - 1 and not terminate_here:
                for j in range(len(rewards_vec)):
                    # Map j to per-agent indices
                    idx_tuple = combo_indices[j]
                    parent_joint = [
                        agent_completions_list[i][idx_tuple[i]]
                        for i in range(self.num_agents)
                    ]
                    # Extend response history with selected completions on this branch
                    next_response_history = [
                        list(response_history_per_agent[i])
                        + [agent_completions_list[i][idx_tuple[i]]]
                        for i in range(self.num_agents)
                    ]

                    child_prompts = self.external_transition(
                        prompt=batch_item.get("prompt", ""),
                        agent_completions=parent_joint,
                        num_agents=self.num_agents,
                        # Full history along this branch up to (and including) this turn
                        prompt_history_per_agent=next_prompt_history,
                        response_history_per_agent=next_response_history,
                    )
                    if (
                        not isinstance(child_prompts, (list, tuple))
                        or len(child_prompts) != self.num_agents
                    ):
                        raise ValueError(
                            "External transition must return per-agent prompts"
                        )
                    child = build_node(
                        turn_idx + 1,
                        prompts_per_agent=list(child_prompts),
                        prompt_history_per_agent=next_prompt_history,
                        response_history_per_agent=next_response_history,
                    )
                    node["children"].append(child)
            return node

        root = build_node(
            0,
            prompts_per_agent=None,
            prompt_history_per_agent=[[] for _ in range(self.num_agents)],
            response_history_per_agent=[[] for _ in range(self.num_agents)],
        )

        def compute_returns(node):
            if not node["children"]:
                node["returns"] = list(node["rewards"]) if node["rewards"] else []
                return node["returns"]
            parent_returns = []
            for j, rj in enumerate(node["rewards"] or []):
                child_node = node["children"][j]
                child_returns = compute_returns(child_node)
                mean_child = float(np.mean(child_returns)) if child_returns else 0.0
                parent_returns.append(rj + gamma * mean_child)
            node["returns"] = parent_returns
            return parent_returns

        compute_returns(root)

        # After returns computed, record per-turn mean returns
        def record_turn_returns(node):
            t = node["turn"]
            if 0 <= t < len(epoch_turn_returns):
                vals = node.get("returns") or []
                if vals:
                    mean_ret = float(np.mean(vals))
                    epoch_turn_returns[t].append(mean_ret)
                    turn_return_node_means[t].append(mean_ret)
            for ch in node["children"]:
                record_turn_returns(ch)

        record_turn_returns(root)

        pending_samples: List[List[NodeSample]] = [[] for _ in range(self.num_agents)]

        def post_order_update(node):
            for child in node["children"]:
                post_order_update(child)
            returns_vec = node.get("returns") or []
            comps_per_agent = node["completions"]
            if not returns_vec:
                return
            # If cross mode, build per-agent joint reward sums (accumulate joint returns
            # for each completion across all joint actions it participates in)
            joint_mode_local = str(getattr(self.args, "joint_mode", "aligned")).lower()
            combo_idx_list = node.get("combo_indices") or []
            per_agent_joint_sums: List[List[float]] = []
            if joint_mode_local == "cross" and combo_idx_list:
                # Determine K per agent
                k = len(comps_per_agent[0]["completions"][0]) if comps_per_agent else 0
                for a in range(self.num_agents):
                    sums = [0.0] * k
                    counts = [0] * k
                    for j, ret in enumerate(returns_vec):
                        idx_a = combo_idx_list[j][a]
                        sums[idx_a] += float(ret)
                        counts[idx_a] += 1
                    # Use joint reward sum per completion (no averaging)
                    per_agent_joint_sums.append(sums)
            else:
                # Aligned: returns already length K
                k = len(returns_vec)
                per_agent_joint_sums = [
                    list(map(float, returns_vec)) for _ in range(self.num_agents)
                ]
            node_env_step = int(node.get("env_step", self.env_step))
            for agent_idx in range(self.num_agents):
                node_rewards = node.get("rewards") or []
                node_mean_reward = float(np.mean(node_rewards)) if node_rewards else 0.0
                node_mean_return = float(np.mean(returns_vec)) if returns_vec else 0.0
                sample = NodeSample(
                    agent_idx=agent_idx,
                    turn_idx=int(node.get("turn", 0)),
                    completions_data=self._pack_completions_for_buffer(
                        comps_per_agent[agent_idx]
                    ),
                    returns=[float(r) for r in per_agent_joint_sums[agent_idx]],
                    node_mean_reward=node_mean_reward,
                    node_mean_return=node_mean_return,
                    node_env_step=node_env_step,
                )
                pending_samples[agent_idx].append(sample)

        post_order_update(root)

        for agent_idx, samples in enumerate(pending_samples):
            samples.sort(key=lambda s: s.node_env_step)
            for sample in samples:
                self._append_to_buffer(agent_idx, sample)

        # Build per-turn batch summary
        batch_loss = float(np.mean(np.abs(root.get("returns") or [0.0])))
        batch_stats: Dict[int, Dict[str, Any]] = {}
        for t in range(num_turns):
            stats: Dict[str, Any] = {}
            if turn_reward_node_means[t]:
                stats["batch_mean_reward"] = float(np.mean(turn_reward_node_means[t]))
            if turn_return_node_means[t]:
                stats["batch_expected_return"] = float(
                    np.mean(turn_return_node_means[t])
                )
            # No per-reward-function means; use a single reward function
            batch_stats[t] = stats

        return batch_loss, batch_stats

    def _generate_completions(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=128,
        prompts_override: Optional[List[str]] = None,
        external_prompts: Optional[Any] = None,
        **kwargs,
    ):
        """
        Generate completions from an agent given prompts, preserving model state.

        Args:
            agent: The agent model to generate completions
            batch_items: List of data items (dictionaries from dataset)
            agent_idx: Index of the agent (used to select the appropriate formatter)
            num_return_sequences: Number of completions to generate per prompt
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional arguments to pass to the model during generation

        Returns:
            Dict: A dictionary containing generated completions and associated data
        """
        device = agent.device

        # Apply the appropriate formatter to create prompts from batch items
        if prompts_override is not None:
            if len(prompts_override) != len(batch_items):
                raise ValueError(
                    "prompts_override must have the same length as batch_items"
                )
            prompts = prompts_override
        else:
            format_func = self.formatters[agent_idx]
            if external_prompts is None:
                prompts = [format_func(item) for item in batch_items]
            else:
                if isinstance(external_prompts, (list, tuple)):
                    if len(external_prompts) != len(batch_items):
                        raise ValueError(
                            "external_prompts must have the same length as batch_items"
                        )
                    external_list = list(external_prompts)
                else:
                    external_list = [external_prompts for _ in batch_items]
                prompts = [
                    format_func(item, external_prompts=ext)
                    for item, ext in zip(batch_items, external_list)
                ]
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating completions")
        tokenizer = self.tokenizers[agent_idx]
        apply_tokenizer_specials(tokenizer, [agent])
        pad_id = tokenizer.pad_token_id

        prompt_encodings = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        prompt_input_ids = prompt_encodings.input_ids
        prompt_attention_mask = prompt_encodings.attention_mask

        training_mode = agent.training
        original_requires_grad = {}

        for name, param in agent.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        agent.eval()

        generation_output = None
        try:
            generation_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "max_new_tokens": max_new_tokens,
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            top_k = getattr(self.args, "top_k", None)
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "num_beams": 1,
                    "num_return_sequences": num_return_sequences,
                }
            )
            if top_k is not None:
                generation_kwargs["top_k"] = top_k

            kwargs.pop("do_sample", None)
            generation_kwargs.update(kwargs)
            generation_output = agent.generate(**generation_kwargs)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

        agent.train(training_mode)
        for name, param in agent.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        completion_input_ids = generation_output.sequences

        prompt_len = prompt_input_ids[0].shape[0]
        pad_positions = (prompt_input_ids[0] == tokenizer.pad_token_id).nonzero()
        if pad_positions.shape[0] > 0:
            prompt_len = pad_positions[0].item()

        completions = []
        completion_tokens_list = []

        total_sequences = completion_input_ids.shape[0]

        batch_completions = []
        batch_completion_tokens = []

        end_idx = min(num_return_sequences, total_sequences)

        for s in range(end_idx):
            completion_tokens = completion_input_ids[s, prompt_len:]
            batch_completion_tokens.append(completion_tokens)

            completion_text = tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )
            batch_completions.append(completion_text)

        completions.append(batch_completions)
        completion_tokens_list.append(batch_completion_tokens)

        completion_attention_masks = []
        batch_masks = []
        for tokens in completion_tokens_list[0]:  # Only one batch
            mask = torch.ones(len(tokens), device=device)
            batch_masks.append(mask)
        completion_attention_masks.append(batch_masks)

        logits = (
            generation_output.scores if hasattr(generation_output, "scores") else []
        )

        return {
            "prompts": prompts,
            "batch_items": batch_items,  # Store original batch items for reference
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "completions": completions,
            "completion_input_ids": completion_tokens_list,
            "completion_attention_mask": completion_attention_masks,
            "logits": logits,
        }

    def _generate_completions_with_external_prompts(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=128,
        external_prompts=None,
        **kwargs,
    ):
        """
        Generate completions with optional external prompts.
        This wraps the _generate_completions method to handle external transitions.

        When num_turns=1 or external_prompts is None, behaves like _generate_completions.
        """
        # If single-turn or no external prompts, use standard method directly
        if self.args.num_turns == 1 or external_prompts is None:
            return self._generate_completions(
                agent,
                batch_items,
                agent_idx=agent_idx,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                external_prompts=external_prompts,
                **kwargs,
            )

        # Multi-turn with external prompts: external modes return next-turn prompts.

        prompts = [external_prompts for _ in batch_items]
        if getattr(self.args, "external_prompt_passthrough", False):
            return self._generate_completions(
                agent,
                batch_items,
                agent_idx=agent_idx,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                prompts_override=prompts,
                external_prompts=external_prompts,
                **kwargs,
            )

        # Temporarily replace prompts in batch_items
        modified_items = []
        for item, prompt in zip(batch_items, prompts):
            modified_item = item.copy() if hasattr(item, "copy") else dict(item)
            modified_item["_original_prompt"] = modified_item.get("prompt", "")
            modified_item["prompt"] = prompt
            modified_items.append(modified_item)

        # Use _generate_completions with modified items
        completions_data = self._generate_completions(
            agent,
            modified_items,
            agent_idx=agent_idx,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            external_prompts=external_prompts,
            **kwargs,
        )

        # Restore original prompts in batch_items
        for i, item in enumerate(completions_data["batch_items"]):
            if "_original_prompt" in item:
                item["prompt"] = item["_original_prompt"]
                del item["_original_prompt"]

        # Update prompts in completions_data to reflect the formatted prompts
        completions_data["prompts"] = prompts

        return completions_data

    def _compute_rewards(
        self, prompts, completions_list, batch_items=None
    ) -> List[float]:
        """
        Compute rewards using a single reward function and optional processor.

        Args:
            prompts: List of prompts (passed via batch_items to reward_fn if needed)
            completions_list: List of completions from each agent

        Returns:
            List of final processed rewards
        """
        # Initialize list to store rewards
        all_rewards = []

        for i in range(self.num_agents):
            if not isinstance(completions_list[i], list):
                completions_list[i] = (
                    [completions_list[i]]
                    if not isinstance(completions_list[i], list)
                    else completions_list[i]
                )

        min_completions = min(len(completions_list[i]) for i in range(self.num_agents))
        try:
            reward_signature = inspect.signature(self.reward_func)
        except (TypeError, ValueError):
            reward_signature = None

        for completion_idx in range(min_completions):
            agent_completions = [
                completions_list[agent_idx][completion_idx]
                for agent_idx in range(self.num_agents)
            ]

            completion_args = [[comp] for comp in agent_completions]
            func_rewards = call_reward_function(
                self.reward_func,
                prompts,
                completion_args,
                num_agents=self.num_agents,
                batch_items=batch_items,
                signature=reward_signature,
            )

            processed_rewards = [self.reward_processor(r) for r in func_rewards]

            all_rewards.append(processed_rewards[0])

        return all_rewards

    def _pack_completions_for_buffer(
        self, completions_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt_ids = completions_data["prompt_input_ids"].detach().cpu()
        completion_ids = completions_data["completion_input_ids"]
        if completion_ids and isinstance(completion_ids[0], list):
            packed_completion_ids = [[t.detach().cpu() for t in completion_ids[0]]]
        else:
            packed_completion_ids = [[t.detach().cpu() for t in completion_ids]]
        return {
            "prompt_input_ids": prompt_ids,
            "completion_input_ids": packed_completion_ids,
        }

    def _append_to_buffer(self, agent_idx: int, sample: NodeSample) -> None:
        buffer = self.rollout_buffers[agent_idx]
        buffer.append(sample)
        if len(buffer) >= int(self.args.rollout_buffer_size):
            self._process_buffer(agent_idx, buffer)

    def _should_log_train(self, step: int) -> bool:
        interval = int(getattr(self.args, "logging_steps", 1))
        if interval <= 1:
            self._last_train_log_step = step
            return True
        if (
            self._last_train_log_step < 0
            or (step - self._last_train_log_step) >= interval
        ):
            self._last_train_log_step = step
            return True
        return False

    def _process_buffer(self, agent_idx: int, buffer: List[NodeSample]) -> None:
        if not buffer:
            return
        turn_groups: Dict[int, List[NodeSample]] = {}
        for sample in buffer:
            t_idx = int(sample.turn_idx)
            turn_groups.setdefault(t_idx, []).append(sample)
        buffer.clear()
        for t_idx in sorted(turn_groups.keys()):
            samples = turn_groups[t_idx]
            self._update_from_samples(agent_idx, samples)
            if self.wandb_initialized and wandb.run is not None and samples:
                batch_log: Dict[str, Any] = {}
                prefix = f"turn_{t_idx + 1}/"
                batch_log[prefix + "reward_mean"] = float(
                    np.mean([s.node_mean_reward for s in samples])
                )
                batch_log[prefix + "expected_return"] = float(
                    np.mean([s.node_mean_return for s in samples])
                )
                step = max(s.node_env_step for s in samples)
                if self._should_log_train(step):
                    wandb.log(batch_log, step=step)

    def _update_from_samples(self, agent_idx: int, samples: List[NodeSample]) -> None:
        if not samples:
            return
        random.shuffle(samples)
        batch_size = int(getattr(self.args, "train_batch_size", len(samples)) or 1)
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            if not batch:
                continue
            self.optimizers[agent_idx].zero_grad()
            scale = 1.0 / len(batch)
            for sample in batch:
                loss = self._compute_loss_with_gradients(
                    self.agents[agent_idx],
                    sample.completions_data,
                    sample.returns,
                )
                (loss * scale).backward()
            self.optimizers[agent_idx].step()

    def _compute_advantages(self, returns_tensor: torch.Tensor) -> torch.Tensor:
        mode = str(self.advantage_mode or "mean").lower()
        if mode == "mean":
            return returns_tensor - returns_tensor.mean()
        if mode == "raw":
            return returns_tensor
        if mode == "max":
            return returns_tensor - returns_tensor.max()
        if mode == "rloo":
            G = returns_tensor.numel()
            if G <= 1:
                return returns_tensor - returns_tensor.mean()
            sum_ret = returns_tensor.sum()
            return (returns_tensor * G - sum_ret) / (G - 1)
        raise ValueError(f"Unknown advantage_mode: {self.advantage_mode}")

    def _compute_loss_with_gradients(self, agent, completions_data, returns):
        """
        Compute loss with proper gradient tracking by performing a new forward pass.

        Args:
            agent: The agent model
            completions_data: The completions data from _generate_completions
            returns: The returns for each completion (not immediate rewards)

        Returns:
            torch.Tensor: The computed loss with gradients attached
        """
        device = agent.device

        # Make sure we have the correct number of rewards
        if len(returns) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Convert returns to tensor
        returns_tensor = torch.tensor(returns, dtype=torch.float, device=device)

        advantages = self._compute_advantages(returns_tensor)
        if self.args.advantage_normalization and advantages.numel() > 1:
            mean = advantages.mean()
            std = advantages.std(unbiased=False).clamp(min=1e-6)
            advantages = (advantages - mean) / std

        # Set agent to train mode to ensure gradients are tracked
        agent.train()

        prompt_input_ids = completions_data["prompt_input_ids"].to(device)
        completion_input_ids = completions_data["completion_input_ids"]
        if completion_input_ids and isinstance(completion_input_ids[0], list):
            completion_input_ids = [[t.to(device) for t in completion_input_ids[0]]]
        else:
            completion_input_ids = [[t.to(device) for t in completion_input_ids]]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        # Process single prompt (batch_size=1)
        prompt_ids = prompt_input_ids[0]

        # Token-based loss: concatenate prompt + completion
        for seq_idx, completion_tokens in enumerate(completion_input_ids[0]):
            # Break if we've processed enough completions for the available rewards
            if seq_idx >= len(advantages):
                break

            advantage = advantages[seq_idx]

            # Create input sequence by concatenating prompt with all but last token of completion
            # (we'll predict the next token at each step)
            if len(completion_tokens) > 0:
                input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])

                # Target is the completion tokens
                target_ids = completion_tokens

                # Create attention mask for the full sequence
                attention_mask = torch.ones(len(input_ids), device=device)

                # Forward pass with gradients enabled
                outputs = agent(
                    input_ids=input_ids.unsqueeze(0),  # Add batch dimension
                    attention_mask=attention_mask.unsqueeze(0),  # Add batch dimension
                )

                # Get logits for the completion part (excluding prompt)
                completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 : -1, :]

                # Calculate log probabilities
                log_probs = []
                for i, token_id in enumerate(target_ids):
                    if i < completion_logits.size(
                        0
                    ):  # Check if we have logits for this position
                        token_logits = completion_logits[i]
                        token_log_prob = torch.log_softmax(token_logits, dim=-1)[
                            token_id
                        ]
                        log_probs.append(token_log_prob)

                if log_probs:
                    sequence_log_prob = torch.stack(log_probs).sum()
                    # Policy gradient loss: -log_prob * advantage
                    loss = -sequence_log_prob * advantage
                    total_loss = total_loss + loss
                    num_samples += 1

        # Average the loss over all processed samples
        if num_samples > 0:
            total_loss = total_loss / num_samples

        # Safety check for invalid loss values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)

        return total_loss

    def save_model(self, output_dir):
        """
        Save the final trained models.

        Args:
            output_dir: Directory to save the models to
        """
        os.makedirs(output_dir, exist_ok=True)

        for agent_idx, agent in enumerate(self.agents):
            agent_dir = f"{output_dir}/agent_{agent_idx}"
            os.makedirs(agent_dir, exist_ok=True)

            agent.save_pretrained(agent_dir)

            if self.tokenizers:
                self.tokenizers[agent_idx].save_pretrained(agent_dir)

        # Log final model saving to wandb
        if self.wandb_initialized and wandb.run is not None:
            wandb.log({"final_model_saved": output_dir})
            wandb.finish()
