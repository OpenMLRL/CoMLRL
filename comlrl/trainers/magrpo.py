import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments


RewardFunc = Union[PreTrainedModel, Callable[[List[str]], float]]


@dataclass
class MAGRPOConfig(TrainingArguments):
    """
    Configuration for MAGRPO training, inheriting from TrainingArguments.
    Supports both single-turn and multi-turn training modes.
    """

    # Core MAGRPO parameters
    num_agents: int = field(
        default=2,
        metadata={"help": "Number of agents; set to 1 for single-agent GRPO."},
    )
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of generations to sample per prompt for each agent."},
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate after the prompt."},
    )

    # Generation parameters (Note: these are set but not currently used by the trainer)
    temperature: float = field(
        default=0.7,
        metadata={
            "help": "Temperature for sampling (currently set but not used - uses model_config instead)."
        },
    )
    top_p: float = field(
        default=0.9,
        metadata={
            "help": "Top-p for sampling (currently set but not used - uses model_config instead)."
        },
    )

    # Multi-turn specific parameters (optional, for MT-MAGRPO)
    num_turns: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of turns per episode. Default is 1 for single-turn training. "
            "Set > 1 to enable multi-turn training with external transitions between turns."
        },
    )
    turn_gradient_weights: Optional[List[float]] = field(
        default_factory=lambda: [1.0, 1.0],
        metadata={
            "help": "Gradient weights for each turn in multi-turn training. "
            "List length should match num_turns. Defaults to equal weights."
        },
    )
    early_termination_weight: Optional[float] = field(
        default=2.0,
        metadata={
            "help": "Weight multiplier applied when early termination occurs (perfect reward achieved). "
            "Only used in multi-turn training."
        },
    )


class MAGRPOTrainer:
    """
    Multi-Agent Group Relative Policy Optimization Trainer (MAGRPO).
    Supports both single-turn and multi-turn training with external transitions.

    When num_turns=1, this trainer behaves as a standard MAGRPO trainer.
    When num_turns>1, it adds multi-turn capabilities with external transitions between turns.

    Args:
        model: The model to be trained for homogeneous agents
        agents: List of agent models (alternative to model)
        num_agents: The number of agents
        reward_funcs: The reward functions for all agents
        reward_weights: The weights for each reward function
        reward_processors: Processors to apply to rewards (e.g., scaling)
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
    """

    def __init__(
        self,
        model: Optional[Union[str, PreTrainedModel]] = None,
        agents: Optional[List[PreTrainedModel]] = None,
        num_agents: int = 2,
        reward_funcs: Union[RewardFunc, List[RewardFunc]] = None,
        reward_weights: Optional[List[float]] = None,
        reward_processors: Optional[List[Callable]] = None,
        formatters: Optional[Union[Callable, List[Callable]]] = None,
        args: Optional[MAGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        external_transition: Optional[Callable] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        eval_logger: Optional[Callable] = None,
        eval_aggregator: Optional[Callable] = None,
    ):
        # Check for GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU not found. MAGRPOTrainer requires GPU for training."
            )

        if model is None and agents is None:
            raise ValueError("Either model or agents must be provided")
        if model is not None and agents is not None:
            raise ValueError("Cannot provide both model and agents parameters")

        self.args = args if args is not None else MAGRPOConfig()

        # Setup formatters (unified for both single-turn and multi-turn)
        self._setup_formatters(formatters, num_agents)

        self._setup_reward_functions(reward_funcs, reward_weights, reward_processors)

        if agents is not None:
            self.agents = agents
            self.num_agents = len(agents)
            if (
                hasattr(agents[0], "base_model")
                and hasattr(agents[0].base_model, "config")
                and hasattr(agents[0].base_model.config, "model_type")
            ):
                self.model_name = agents[0].base_model.config.model_type
            elif hasattr(agents[0], "config") and hasattr(
                agents[0].config, "_name_or_path"
            ):
                self.model_name = agents[0].config._name_or_path
            else:
                self.model_name = agents[0].__class__.__name__

            self.model_config = model_config if model_config else {}
        else:
            self.model_config = model_config if model_config else {}
            self.num_agents = num_agents
            if isinstance(model, str):
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.agents = [
                    AutoModelForCausalLM.from_pretrained(
                        model, **self.model_config.get("model_kwargs", {})
                    )
                    for _ in range(num_agents)
                ]
                self.model_name = model

                if tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model, **self.model_config.get("tokenizer_kwargs", {})
                    )
                    special_tokens = self.model_config.get("special_tokens", {})
                    if special_tokens:
                        self.tokenizer.add_special_tokens(special_tokens)
            else:
                raise ValueError(
                    "Model should be a string to create homogeneous agents"
                )

        # Allow single-agent as a special case (GRPO)
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self.args.num_generations < 2:
            raise ValueError(
                "num_generations must be >= 2 (group baseline requires multiple samples)."
            )
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("MAGRPO requires per_device_train_batch_size to be 1. ")

        # Check for external_transition requirement in multi-turn training
        if self.args.num_turns > 1 and external_transition is None:
            raise ValueError(
                "Multi-turn training requires an external_transition function."
            )

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.eval_logger = eval_logger
        self.eval_aggregator = eval_aggregator
        self.external_transition = external_transition

        self.optimizers = [
            torch.optim.AdamW(
                agent.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            for agent in self.agents
        ]

        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if self.wandb_config is not None:
            self._init_wandb()

    def _setup_formatters(self, formatters, num_agents):
        """Set up format functions for each agent that can handle external transitions."""
        # Use multi-turn compatible default formatter that accepts external prompts
        default_format_func = lambda x, external_prompts=None: x.get("prompt", "")

        if formatters is None:
            self.formatters = [default_format_func] * num_agents
        elif callable(formatters) and not isinstance(formatters, list):
            # Wrap the formatter to accept external_prompts parameter
            original_formatter = formatters
            sig = inspect.signature(original_formatter)
            if "external_prompts" in sig.parameters:
                wrapped_formatter = lambda x, external_prompts=None: (
                    original_formatter(x, external_prompts=external_prompts)
                    if external_prompts is not None
                    else original_formatter(x)
                )
            else:
                wrapped_formatter = lambda x, external_prompts=None: original_formatter(
                    x
                )
            self.formatters = [wrapped_formatter] * num_agents
        elif isinstance(formatters, list):
            if len(formatters) != num_agents:
                raise ValueError(
                    f"Number of formatters ({len(formatters)}) must match "
                    f"number of agents ({num_agents})"
                )
            # Ensure all formatters can accept external_prompts
            wrapped_formatters = []
            for formatter in formatters:
                sig = inspect.signature(formatter)
                if "external_prompts" in sig.parameters:

                    def make_wrapper(f):
                        def wrapped(x, external_prompts=None):
                            return f(x, external_prompts=external_prompts)

                        return wrapped

                    wrapped_formatters.append(make_wrapper(formatter))
                else:
                    # Wrap to accept but ignore parameter
                    wrapped = lambda x, external_prompts=None, f=formatter: f(x)
                    wrapped_formatters.append(wrapped)
            self.formatters = wrapped_formatters
        else:
            raise ValueError(
                f"formatters must be a callable, a list of callables, or None. "
                f"Got {type(formatters)}"
            )

    def _setup_reward_functions(
        self, reward_funcs, reward_weights=None, reward_processors=None
    ):
        """Set up reward functions with weights and processors."""
        if not isinstance(reward_funcs, list):
            self.reward_funcs = [reward_funcs]
        else:
            self.reward_funcs = reward_funcs

        if reward_weights is None:
            self.reward_weights = [1.0 / len(self.reward_funcs)] * len(
                self.reward_funcs
            )
        else:
            if len(reward_weights) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(reward_weights)}) must match "
                    f"number of reward functions ({len(self.reward_funcs)})"
                )
            total = sum(reward_weights)
            self.reward_weights = [w / total for w in reward_weights]

        if reward_processors is None:
            self.reward_processors = [lambda x: x] * len(self.reward_funcs)
        elif not isinstance(reward_processors, list):
            self.reward_processors = [reward_processors] * len(self.reward_funcs)
        else:
            if len(reward_processors) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward processors ({len(reward_processors)}) must match "
                    f"number of reward functions ({len(self.reward_funcs)})"
                )

            self.reward_processors = []
            for processor in reward_processors:
                if processor is None:
                    self.reward_processors.append(lambda x: x)
                else:
                    self.reward_processors.append(processor)

    def _init_wandb(self):
        """Initialize Weights & Biases for tracking with multi-turn config."""
        if not self.wandb_initialized:
            if self.wandb_config is None:
                self.wandb_config = {}

            wandb_project = self.wandb_config.get("project", "mlrl")
            wandb_entity = self.wandb_config.get("entity", "nu-llpr")

            # Use different default names based on num_turns
            if self.args.num_turns == 1:
                wandb_name = self.wandb_config.get("name", "test-magrpo")
            else:
                wandb_name = self.wandb_config.get("name", "test-mt-magrpo")

            wandb_dir = self.wandb_config.get("dir", None)

            config_dict = {
                "model_name": self.model_name,
                "num_agents": self.num_agents,
                "num_turns": self.args.num_turns,
                "num_reward_functions": len(self.reward_funcs),
                "reward_weights": self.reward_weights,
                "learning_rate": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "num_train_epochs": self.args.num_train_epochs,
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "num_generations": self.args.num_generations,
                "max_new_tokens": self.args.max_new_tokens,
            }

            # Only add multi-turn specific config if num_turns > 1
            if self.args.num_turns > 1:
                config_dict.update(
                    {
                        "turn_gradient_weights": self.args.turn_gradient_weights,
                        "early_termination_weight": self.args.early_termination_weight,
                        # External model configuration is handled by the calling code
                    }
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

            wandb.init(**init_kwargs)
            self.wandb_initialized = True

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=False,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """Returns the evaluation DataLoader."""
        if self.eval_dataset is None:
            return None

        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
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

        # Get evaluation dataloader
        eval_dataloader = self.get_eval_dataloader()

        # Evaluate on specified number of samples
        with torch.no_grad():
            for eval_idx, batch in enumerate(eval_dataloader):
                if eval_idx >= num_eval_samples:
                    break

                # Process each batch item
                for batch_item in batch:
                    self._evaluate_sample(
                        batch_item,
                        all_agent_completions_turns,
                        all_test_cases,
                        all_entry_points,
                        all_prompts,
                    )

        # Calculate and log metrics
        return self._log_eval_metrics(
            all_agent_completions_turns,
            all_test_cases,
            all_entry_points,
            all_prompts,
        )

    def _evaluate_sample(
        self,
        batch_item,
        all_agent_completions_turns,
        all_test_cases,
        all_entry_points,
        all_prompts,
    ):
        """Evaluate a single sample for any number of turns."""
        # Storage for each agent's completions across turns
        agent_sample_completions = [[] for _ in range(self.num_agents)]

        # Store sample information
        all_test_cases.append(batch_item.get("test", ""))
        all_entry_points.append(batch_item.get("entry_point", ""))
        all_prompts.append(batch_item.get("prompt", ""))

        # Store best completions from previous turn for external transitions
        previous_best_completions = [None] * self.num_agents

        # Run episode with configured number of turns
        for turn_idx in range(self.args.num_turns):
            # Prepare external prompts for turns after the first
            agent_external_prompts = [None] * self.num_agents

            if turn_idx > 0 and all(c is not None for c in previous_best_completions):
                # Get external transitions based on previous turn's best result
                if self.external_transition is not None:
                    transition_result = self.external_transition(
                        prompt=batch_item.get("prompt", ""),
                        agent_completions=previous_best_completions,
                        num_agents=self.num_agents,
                    )

                    # External transition should return prompts for each agent
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
                agent_sample_completions[agent_idx].append(completion)

            # Check for early termination (only relevant for multi-turn)
            if self.args.num_turns > 1:
                agent_completions_for_reward = [
                    [agent_sample_completions[i][-1]] for i in range(self.num_agents)
                ]
                # Get the prompt from the first agent's completion data
                # Since all agents use the same batch_item, we can use any agent's prompt
                prompt = self.formatters[0](batch_item)
                rewards, _ = self._compute_rewards(
                    [prompt],
                    agent_completions_for_reward,
                    batch_items=[batch_item],
                )

                if rewards:
                    for agent_idx in range(self.num_agents):
                        previous_best_completions[agent_idx] = agent_sample_completions[
                            agent_idx
                        ][-1]

                if rewards[0] == 4.0:
                    # Early termination
                    break

        # Store completions for all agents
        for agent_idx in range(self.num_agents):
            all_agent_completions_turns[agent_idx].append(
                agent_sample_completions[agent_idx]
            )

    def _log_eval_metrics(
        self, all_agent_completions_turns, all_test_cases, all_entry_points, all_prompts
    ) -> Dict[str, float]:
        """Log evaluation metrics for any number of turns."""
        eval_metrics = {}

        # Detailed logging (if logger is provided)
        if (
            self.eval_logger is not None
            and self.eval_aggregator is not None
            and all_agent_completions_turns
            and all(agent_comps for agent_comps in all_agent_completions_turns)
        ):
            # Dynamically call eval_logger based on its signature
            sig = inspect.signature(self.eval_logger)
            params = sig.parameters

            # Check if logger accepts a generic list of agent completions
            if "agent_completions_turns" in params or "agent_completions" in params:
                # Modern N-agent logger interface
                if "agent_completions_turns" in params:
                    detailed_metrics = self.eval_logger(
                        agent_completions_turns=all_agent_completions_turns,
                        test_cases=all_test_cases,
                        entry_points=all_entry_points,
                        prompts=all_prompts,
                    )
                else:
                    # For single-turn, flatten the turns dimension
                    if self.args.num_turns == 1:
                        flattened_completions = [
                            [sample[0] for sample in agent_comps]
                            for agent_comps in all_agent_completions_turns
                        ]
                        detailed_metrics = self.eval_logger(
                            agent_completions=flattened_completions,
                            test_cases=all_test_cases,
                            entry_points=all_entry_points,
                            prompts=all_prompts,
                        )
                    else:
                        detailed_metrics = self.eval_logger(
                            agent_completions=all_agent_completions_turns,
                            test_cases=all_test_cases,
                            entry_points=all_entry_points,
                            prompts=all_prompts,
                        )
            else:
                # Legacy interface expecting individual agent arguments
                # Build arguments based on number of agents
                args = []

                # Add agent completions up to the number expected by the logger
                param_list = list(params.keys())
                for i, param_name in enumerate(param_list):
                    if param_name in ["test_cases", "all_test_cases"]:
                        break
                    if i < len(all_agent_completions_turns):
                        # For single-turn, flatten the turns dimension
                        if self.args.num_turns == 1:
                            args.append(
                                [sample[0] for sample in all_agent_completions_turns[i]]
                            )
                        else:
                            args.append(all_agent_completions_turns[i])
                    else:
                        args.append([])

                # Check if this is a code logger or text logger
                if "test_cases" in params or "all_test_cases" in params:
                    # Code logger needs test cases, entry points, and prompts
                    args.extend([all_test_cases, all_entry_points, all_prompts])

                detailed_metrics = self.eval_logger(*args)

            # Aggregate metrics for logging
            if self.args.num_turns > 1:
                aggregated_detailed_metrics = self.eval_aggregator(
                    detailed_metrics, num_turns=self.args.num_turns
                )
            else:
                aggregated_detailed_metrics = self.eval_aggregator(detailed_metrics)

            # Add to eval_metrics - using unified format
            for key, value in aggregated_detailed_metrics.items():
                eval_metrics[f"eval/{key}"] = value

        # Log evaluation metrics
        if self.wandb_initialized:
            wandb.log(eval_metrics)

        return eval_metrics

    def train(self, **kwargs):
        """
        Unified train method that supports both single-turn and multi-turn training.
        """
        # Initialize wandb if not already done
        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        # Setup devices for training (GPU is required)
        device = torch.device("cuda")
        for agent in self.agents:
            agent.to(device)
            agent.train()

        # Track epoch rewards for conditional saving
        epoch_rewards_history = []

        # Create the data pipeline for generating examples
        for epoch in range(0, int(self.args.num_train_epochs)):
            epoch_loss = 0.0
            epoch_rewards = []
            epoch_agent_rewards = [[] for _ in range(self.num_agents)]
            # Track individual reward components
            epoch_reward_components = [[] for _ in range(len(self.reward_funcs))]

            # Turn tracking for all cases (including single-turn)
            epoch_turn_rewards = [[] for _ in range(self.args.num_turns)]
            epoch_early_terminations = 0

            for batch_idx, batch in enumerate(self.get_train_dataloader()):
                # evaluate every 4 batches
                if batch_idx % 4 == 0:
                    eval_metrics = self.evaluate(num_eval_samples=4)
                    if self.wandb_initialized:
                        wandb.log(eval_metrics)

                # Process single batch item (batch_size=1 enforced)
                batch_item = batch[0]
                # Unified training step
                batch_loss, rewards, turn_data, early_termination = self._train_step(
                    batch_item,
                    epoch_rewards,
                    epoch_turn_rewards,
                    **kwargs,
                )
                if early_termination:
                    epoch_early_terminations += 1

                epoch_loss += batch_loss

            # Log epoch summary
            self._log_epoch_summary(
                epoch,
                epoch_loss,
                epoch_rewards,
                epoch_agent_rewards,
                epoch_reward_components,
                epoch_turn_rewards,
                epoch_early_terminations,
                epoch_rewards_history,
            )

    def _train_step(
        self,
        batch_item,
        epoch_rewards,
        epoch_turn_rewards,
        **kwargs,
    ):
        """Execute a unified training step that handles any number of turns."""
        # Store turn data for sequential updates
        turn_data = []
        early_termination = False

        # Store best completions from previous turn for external transitions
        previous_best_completions = [None] * self.num_agents

        # Execute multi-turn episode
        for turn_idx in range(self.args.num_turns):
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            # Prepare external prompts for turns after the first
            agent_external_prompts = [None] * self.num_agents
            if turn_idx > 0 and all(c is not None for c in previous_best_completions):
                transition_result = self.external_transition(
                    prompt=batch_item.get("prompt", ""),
                    agent_completions=previous_best_completions,
                    num_agents=self.num_agents,
                )

                # External transition should return prompts for each agent
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

            # Generate completions from each agent
            all_completions = []
            for agent_idx in range(self.num_agents):
                agent_completions = self._generate_completions_with_external_prompts(
                    self.agents[agent_idx],
                    [batch_item],
                    agent_idx=agent_idx,
                    num_return_sequences=self.args.num_generations,
                    max_new_tokens=self.args.max_new_tokens,
                    external_prompts=agent_external_prompts[agent_idx],
                    **kwargs,
                )
                all_completions.append(agent_completions)

            # Extract completions for reward calculation
            agent_completions_list = []
            for agent_idx in range(self.num_agents):
                agent_completions_list.append(
                    all_completions[agent_idx]["completions"][0]
                )

            # Get formatted prompt
            formatted_prompt = all_completions[0]["prompts"][0]

            # Compute rewards
            rewards, reward_components = self._compute_rewards(
                [formatted_prompt],
                agent_completions_list,
                batch_items=[batch_item],
            )

            # Find the best completion set (highest reward)
            if rewards:
                best_idx = rewards.index(max(rewards))
                for agent_idx in range(self.num_agents):
                    previous_best_completions[agent_idx] = agent_completions_list[
                        agent_idx
                    ][best_idx]

            # Calculate turn mean reward
            turn_mean_reward = np.mean(rewards) if rewards else 0
            epoch_turn_rewards[turn_idx].append(turn_mean_reward)

            # Store turn data
            turn_data.append(
                {
                    "completions": all_completions,
                    "rewards": rewards,
                    "reward_components": reward_components,
                    "mean_reward": turn_mean_reward,
                }
            )

            # Log turn metrics
            if self.wandb_initialized:
                turn_log_data = {
                    f"turn_{turn_idx + 1}/batch_rewards_mean": turn_mean_reward,
                }

                # Log reward components
                for i, component in enumerate(reward_components):
                    component_mean = np.mean(component) if component else 0
                    turn_log_data[f"turn_{turn_idx + 1}/reward_{i + 1}_mean"] = (
                        component_mean
                    )

                # Check for early termination
                if turn_mean_reward == 4.0:
                    early_termination = True
                    turn_log_data[f"turn_{turn_idx + 1}/early_termination"] = True
                    wandb.log(turn_log_data)

                    # Fill remaining turns with max values
                    for future_turn in range(turn_idx + 1, self.args.num_turns):
                        # Log perfect metrics for skipped turns
                        perfect_metrics = {
                            f"turn_{future_turn + 1}/batch_rewards_mean": 4.0,
                            f"turn_{future_turn + 1}/reward_1_mean": 4.0,
                        }
                        wandb.log(perfect_metrics)
                        epoch_turn_rewards[future_turn].append(4.0)

                        # Log improvement as 0
                        if future_turn > 0:
                            wandb.log(
                                {
                                    f"turn_{future_turn + 1}/improvement_from_turn_{future_turn}": 0.0
                                }
                            )

                    break

                wandb.log(turn_log_data)

        # Log turn-to-turn improvements
        if self.wandb_initialized:
            if len(turn_data) >= 2:
                for i in range(1, len(turn_data)):
                    improvement = (
                        turn_data[i]["mean_reward"] - turn_data[i - 1]["mean_reward"]
                    )
                    wandb.log({f"turn_{i + 1}/improvement_from_turn_{i}": improvement})
            elif early_termination and self.args.num_turns >= 2:
                # If early terminated at turn 1, still log 0 improvement for turn 2
                wandb.log({"turn_2/improvement_from_turn_1": 0.0})

        # Sequential model updates after episode ends
        batch_loss = 0.0

        # Use turn-specific gradient weights from config
        turn_weights = self.args.turn_gradient_weights

        for turn_idx, turn_info in enumerate(turn_data):
            # Get turn-specific weight
            turn_gradient_weight = (
                turn_weights[turn_idx] if turn_idx < len(turn_weights) else 1.0
            )

            # Apply termination weight if needed (for early termination bonus)
            if early_termination and turn_idx == len(turn_data) - 1:
                # Combine turn weight with early termination weight
                final_weight = turn_gradient_weight * self.args.early_termination_weight
            else:
                final_weight = turn_gradient_weight

            # Update each agent for this turn
            turn_loss = 0.0
            agent_losses = []

            for agent_idx in range(self.num_agents):
                # Compute loss with gradients
                agent_loss = self._compute_loss_with_gradients(
                    self.agents[agent_idx],
                    turn_info["completions"][agent_idx],
                    turn_info["rewards"],
                )

                # Apply weight
                weighted_loss = agent_loss * final_weight

                # Backward pass and optimization
                weighted_loss.backward()
                self.optimizers[agent_idx].step()
                self.optimizers[agent_idx].zero_grad()

                turn_loss += agent_loss.detach().item()
                agent_losses.append(agent_loss.detach().item())

            batch_loss += turn_loss

            # Log turn update info
            if self.wandb_initialized:
                wandb.log(
                    {
                        f"turn_{turn_idx + 1}/update_loss": turn_loss,
                        f"turn_{turn_idx + 1}/gradient_weight": turn_gradient_weight,
                        f"turn_{turn_idx + 1}/final_weight": final_weight,
                    }
                )

        # Collect all rewards for epoch tracking
        for turn_info in turn_data:
            epoch_rewards.extend(turn_info["rewards"])

        # Log episode summary
        if self.wandb_initialized:
            log_data = {
                "system/episode_loss": batch_loss,
                "system/episode_num_turns": len(turn_data),
            }
            # Only log early termination if it happened
            if early_termination:
                log_data["system/episode_early_termination"] = True
            wandb.log(log_data)

        return batch_loss, epoch_rewards, turn_data, early_termination

    def _log_epoch_summary(
        self,
        epoch,
        epoch_loss,
        epoch_rewards,
        epoch_agent_rewards,
        epoch_reward_components,
        epoch_turn_rewards,
        epoch_early_terminations,
        epoch_rewards_history,
    ):
        """Log epoch summary metrics in unified format."""
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        epoch_rewards_history.append(avg_reward)

        if not self.wandb_initialized:
            return

        epoch_log = {
            "system/epoch": epoch,
            "system/epoch_loss": (
                epoch_loss / len(self.get_train_dataloader()) if epoch_loss else 0
            ),
            "system/epoch_avg_reward": avg_reward,
        }

        # Add agent-specific reward tracking
        avg_agent_rewards = [
            sum(rewards) / len(rewards) if rewards else 0
            for rewards in epoch_agent_rewards
        ]
        for i, avg_agent_reward in enumerate(avg_agent_rewards):
            epoch_log[f"system/agent{i + 1}_avg_reward"] = avg_agent_reward

        # Add component-specific reward tracking
        avg_reward_components = [
            sum(comp) / len(comp) if comp else 0 for comp in epoch_reward_components
        ]
        for i, avg_component in enumerate(avg_reward_components):
            epoch_log[f"system/reward_{i + 1}_avg"] = avg_component

        # Multi-turn specific metrics
        if self.args.num_turns > 1 and epoch_turn_rewards:
            epoch_log["system/epoch_early_termination_rate"] = (
                epoch_early_terminations / len(self.get_train_dataloader())
            )

            # Log average rewards per turn
            for turn_idx in range(self.args.num_turns):
                if epoch_turn_rewards[turn_idx]:
                    epoch_log[f"system/epoch_turn_{turn_idx + 1}_avg_reward"] = np.mean(
                        epoch_turn_rewards[turn_idx]
                    )

        wandb.log(epoch_log)

    def _generate_completions(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=128,
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
        format_func = self.formatters[agent_idx]
        prompts = [format_func(item) for item in batch_items]
        # batch_size is always 1 due to enforced constraint

        # Ensure tokenizer exists
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating completions")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        prompt_input_ids = prompt_encodings.input_ids
        prompt_attention_mask = prompt_encodings.attention_mask

        # Store original model state and gradient settings
        training_mode = agent.training
        original_requires_grad = {}

        # Save original requires_grad states
        for name, param in agent.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False  # Temporarily disable gradients for generation

        agent.eval()  # Set to eval mode for generation

        # Generate completions without gradients
        generation_output = None
        try:
            # Use max_new_tokens instead of max_length
            generation_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "max_new_tokens": max_new_tokens,  # Changed from max_length
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            # If requesting multiple sequences, use sampling for diversity
            if num_return_sequences > 1:
                # Use generation parameters from config
                generation_update = {
                    "do_sample": True,  # Enable sampling for randomness
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "top_k": 50,  # Default top_k value
                    "num_beams": 1,  # Disable beam search when sampling
                    "num_return_sequences": num_return_sequences,
                }
                generation_kwargs.update(generation_update)

            # Set pad_token_id from tokenizer if not set
            if (
                "pad_token_id" not in generation_kwargs
                or generation_kwargs["pad_token_id"] is None
            ):
                generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            # Add any additional user-provided kwargs (these override model defaults)
            generation_kwargs.update(kwargs)
            generation_output = agent.generate(**generation_kwargs)
        except Exception as e:
            raise ValueError(f"Generation failed: {str(e)}")

        # Restore original model state and gradients
        agent.train(training_mode)
        for name, param in agent.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        # Extract completion tokens (excluding prompt tokens)
        completion_input_ids = generation_output.sequences

        # For single prompt, find its actual length in tokens
        # to properly extract just the completion part
        prompt_len = prompt_input_ids[0].shape[0]
        # Find where padding token starts if any
        pad_positions = (prompt_input_ids[0] == self.tokenizer.pad_token_id).nonzero()
        if pad_positions.shape[0] > 0:
            prompt_len = pad_positions[0].item()

        # Extract completion text for single prompt
        completions = []
        completion_tokens_list = []

        # Calculate total sequence count
        total_sequences = completion_input_ids.shape[0]

        # Process single prompt and its multiple completions
        batch_completions = []
        batch_completion_tokens = []

        # Get all sequences for this prompt (start_idx=0, end_idx=num_return_sequences)
        end_idx = min(num_return_sequences, total_sequences)

        for s in range(end_idx):
            # Get only the completion part (exclude the prompt tokens)
            completion_tokens = completion_input_ids[s, prompt_len:]
            batch_completion_tokens.append(completion_tokens)

            # Decode to text
            completion_text = self.tokenizer.decode(
                completion_tokens, skip_special_tokens=True
            )
            batch_completions.append(completion_text)

        completions.append(batch_completions)
        completion_tokens_list.append(batch_completion_tokens)

        # Create attention masks for completions (single batch)
        completion_attention_masks = []
        batch_masks = []
        for tokens in completion_tokens_list[0]:  # Only one batch
            mask = torch.ones(len(tokens), device=device)
            batch_masks.append(mask)
        completion_attention_masks.append(batch_masks)

        # Extract logit for computing loss
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
                **kwargs,
            )

        # Multi-turn with external prompts: external modes return full prompts.
        # Use them directly for this turn without invoking first-turn formatters.
        prompts = [external_prompts for _ in batch_items]

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
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Compute combined rewards based on multiple reward functions, with weights.

        Args:
            prompts: List of prompts
            completions_list: List of completions from each agent

        Returns:
            Tuple containing:
            - List of final weighted rewards
            - List of individual reward components (for logging)
        """
        # Initialize lists to store rewards
        all_rewards = []
        all_reward_components = [[] for _ in range(len(self.reward_funcs))]

        # Single prompt case (batch_size=1 enforced)
        # Ensure correct structure for all agents
        for i in range(self.num_agents):
            if not isinstance(completions_list[i], list):
                completions_list[i] = (
                    [completions_list[i]]
                    if not isinstance(completions_list[i], list)
                    else completions_list[i]
                )

        # Find minimum number of completions across all agents
        min_completions = min(len(completions_list[i]) for i in range(self.num_agents))

        for completion_idx in range(min_completions):
            # Extract one completion from each agent
            agent_completions = [
                completions_list[agent_idx][completion_idx]
                for agent_idx in range(self.num_agents)
            ]

            # Calculate rewards from each function and apply weights
            weighted_reward = 0.0
            reward_components = []

            for func_idx, (reward_func, weight, processor) in enumerate(
                zip(self.reward_funcs, self.reward_weights, self.reward_processors)
            ):
                # Call reward function with all agent completions
                try:
                    completion_args = [[comp] for comp in agent_completions]

                    # Check if reward function accepts batch_items parameter
                    sig = inspect.signature(reward_func)
                    if "batch_items" in sig.parameters:
                        func_rewards = reward_func(
                            *completion_args, batch_items=batch_items
                        )
                    else:
                        func_rewards = reward_func(*completion_args)
                except TypeError:
                    func_rewards = reward_func(agent_completions)

                # Apply processor to rewards
                processed_rewards = [processor(r) for r in func_rewards]

                # Store the raw component rewards for logging
                reward_components.append(processed_rewards[0])
                all_reward_components[func_idx].extend(processed_rewards)

                # Add weighted component to total reward
                weighted_reward += weight * processed_rewards[0]

            all_rewards.append(weighted_reward)

        return all_rewards, all_reward_components

    def _compute_loss_with_gradients(self, agent, completions_data, rewards):
        """
        Compute loss with proper gradient tracking by performing a new forward pass.

        Args:
            agent: The agent model
            completions_data: The completions data from _generate_completions
            rewards: The rewards for each completion

        Returns:
            torch.Tensor: The computed loss with gradients attached
        """
        device = agent.device

        # Make sure we have the correct number of rewards
        if len(rewards) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)

        # Use baseline approach
        rewards_baseline = rewards_tensor.mean()  # Use mean as baseline
        advantages = rewards_tensor - rewards_baseline  # Compute advantages

        # Clip advantages to reasonable range to prevent numerical instability
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        # Set agent to train mode to ensure gradients are tracked
        agent.train()

        prompt_input_ids = completions_data["prompt_input_ids"]
        completion_input_ids = completions_data["completion_input_ids"]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        # Process single prompt (batch_size=1)
        prompt_ids = prompt_input_ids[0]

        # Process each generated completion for this prompt
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

            if self.tokenizer:
                self.tokenizer.save_pretrained(agent_dir)

        # Log final model saving to wandb
        if self.wandb_initialized:
            wandb.log({"final_model_saved": output_dir})
            wandb.finish()
