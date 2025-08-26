import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from comlrl.trainers.magrpo import MAGRPOConfig

RewardFunc = Union[PreTrainedModel, Callable[[List[str]], float]]


class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer (GRPO) - Single Agent Version.

    Args:
        model: The model to be trained
        reward_funcs: The reward functions
        reward_weights: The weights for each reward function
        reward_processors: Processors to apply to rewards (e.g., scaling)
        formatter: Formatter to apply to dataset items
        args: The training arguments
        train_dataset: The training dataset
        eval_dataset: The evaluation dataset
        tokenizer: The tokenizer
        wandb_config: Configuration for Weights & Biases logging
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, List[RewardFunc]] = None,
        reward_weights: Optional[List[float]] = None,
        reward_processors: Optional[List[Callable]] = None,
        formatter: Optional[Callable] = None,
        args: Optional[MAGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        self._setup_formatter(formatter)
        self._setup_reward_functions(reward_funcs, reward_weights, reward_processors)

        self.args = args if args is not None else MAGRPOConfig()

        if isinstance(model, str):
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(model)
            self.model_name = model
        else:
            self.model = model
            self.model_name = model.__class__.__name__

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if self.wandb_config is not None:
            self._init_wandb()

    def _setup_formatter(self, formatter):
        """Set up format function."""
        default_format_func = lambda x: x.get("prompt", "")

        if formatter is None:
            self.formatter = default_format_func
        elif callable(formatter):
            self.formatter = formatter
        else:
            raise ValueError(
                f"formatter must be a callable or None. Got {type(formatter)}"
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
            self.reward_processors = [
                p if p is not None else lambda x: x for p in reward_processors
            ]

    def _init_wandb(self):
        """Initialize Weights & Biases for tracking."""
        if not self.wandb_initialized:
            wandb_project = self.wandb_config.get("project", "trl")
            wandb_entity = self.wandb_config.get("entity", "2478906339-null")
            wandb_name = self.wandb_config.get("name", "test-grpo")
            wandb_dir = self.wandb_config.get("dir", None)

            config_dict = {
                "model_name": self.model_name,
                "num_reward_functions": len(self.reward_funcs),
                "reward_weights": self.reward_weights,
                "learning_rate": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "num_train_epochs": self.args.num_train_epochs,
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "num_generations": self.args.num_generations,
                "max_new_tokens": self.args.max_new_tokens,
            }

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
        """Evaluate the model on a subset of the evaluation dataset."""
        if self.eval_dataset is None:
            return {}

        eval_rewards = []
        eval_reward_components = [[] for _ in range(len(self.reward_funcs))]

        eval_dataloader = self.get_eval_dataloader()

        with torch.no_grad():
            for eval_idx, batch in enumerate(eval_dataloader):
                if eval_idx >= num_eval_samples:
                    break

                for batch_item in batch:
                    completions_data = self._generate_completions(
                        [batch_item],
                        num_return_sequences=1,
                        max_new_tokens=self.args.max_new_tokens,
                    )

                    formatted_prompt = completions_data["prompts"][0]
                    completion = completions_data["completions"][0][0]

                    rewards, reward_components = self._compute_rewards(
                        [formatted_prompt],
                        [[completion]],
                        batch_items=[batch_item],
                    )

                    eval_rewards.extend(rewards)

                    for i, component in enumerate(reward_components):
                        eval_reward_components[i].extend(component)

        eval_metrics = {
            "eval/avg_reward": np.mean(eval_rewards) if eval_rewards else 0,
            "eval/num_samples": len(eval_rewards),
        }

        for i, component_rewards in enumerate(eval_reward_components):
            if component_rewards:
                eval_metrics[f"eval/reward_{i + 1}_avg"] = np.mean(component_rewards)

        if self.wandb_initialized:
            wandb.log(eval_metrics)

        return eval_metrics

    def train(self, **kwargs):
        """Train the model."""
        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        epoch_rewards_history = []

        for epoch in range(0, int(self.args.num_train_epochs)):
            epoch_loss = 0.0
            epoch_rewards = []
            epoch_reward_components = [[] for _ in range(len(self.reward_funcs))]

            for batch_idx, batch in enumerate(self.get_train_dataloader()):
                if batch_idx % 4 == 0:
                    eval_metrics = self.evaluate(num_eval_samples=4)
                    wandb.log(eval_metrics)

                for item_idx, batch_item in enumerate(batch):
                    self.optimizer.zero_grad()

                    completions_data = self._generate_completions(
                        [batch_item],
                        num_return_sequences=self.args.num_generations,
                        max_new_tokens=self.args.max_new_tokens,
                        **kwargs,
                    )

                    formatted_prompt = completions_data["prompts"][0]
                    completions = completions_data["completions"][0]

                    rewards, reward_components = self._compute_rewards(
                        [formatted_prompt],
                        [completions],
                        batch_items=[batch_item],
                    )

                    epoch_rewards.extend(rewards)

                    for i, component in enumerate(reward_components):
                        epoch_reward_components[i].extend(component)

                    loss = self._compute_loss_with_gradients(completions_data, rewards)

                    loss.backward()
                    self.optimizer.step()

                    batch_loss = loss.detach().item()
                    epoch_loss += batch_loss

                    log_data = {
                        "batch_loss": batch_loss,
                        "batch_rewards_mean": np.mean(rewards) if rewards else 0,
                    }

                    for i, component in enumerate(reward_components):
                        component_mean = np.mean(component) if component else 0
                        log_data[f"reward_{i + 1}_mean"] = component_mean

                    wandb.log(log_data)

            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            avg_reward_components = [
                sum(comp) / len(comp) if comp else 0 for comp in epoch_reward_components
            ]

            epoch_rewards_history.append(avg_reward)

            epoch_log = {
                "epoch": epoch,
                "epoch_loss": (
                    epoch_loss / len(self.get_train_dataloader()) if epoch_loss else 0
                ),
                "epoch_avg_reward": avg_reward,
            }

            for i, avg_component in enumerate(avg_reward_components):
                epoch_log[f"reward_{i + 1}_avg"] = avg_component

            wandb.log(epoch_log)

    def _generate_completions(
        self,
        batch_items,
        num_return_sequences=1,
        max_new_tokens=128,
        **kwargs,
    ):
        """Generate completions from the model."""
        device = self.model.device

        prompts = [self.formatter(item) for item in batch_items]
        batch_size = len(prompts)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating completions")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt_encodings = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        prompt_input_ids = prompt_encodings.input_ids
        prompt_attention_mask = prompt_encodings.attention_mask

        training_mode = self.model.training
        original_requires_grad = {}

        for name, param in self.model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        self.model.eval()

        generation_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": max_new_tokens,
            "output_scores": True,
            "return_dict_in_generate": True,
        }

        if num_return_sequences > 1:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "num_beams": 1,
                    "num_return_sequences": num_return_sequences,
                }
            )

        generation_kwargs.update(kwargs)

        generation_output = self.model.generate(**generation_kwargs)

        self.model.train(training_mode)
        for name, param in self.model.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        completion_input_ids = generation_output.sequences

        prompt_lengths = []
        for b in range(batch_size):
            prompt_len = prompt_input_ids[b].shape[0]
            pad_positions = (
                prompt_input_ids[b] == self.tokenizer.pad_token_id
            ).nonzero()
            if pad_positions.shape[0] > 0:
                prompt_len = pad_positions[0].item()
            prompt_lengths.append(prompt_len)

        completions = []
        completion_tokens_list = []

        total_sequences = completion_input_ids.shape[0]

        for b in range(batch_size):
            prompt_len = prompt_lengths[b]
            batch_completions = []
            batch_completion_tokens = []

            start_idx = b * num_return_sequences
            end_idx = min(start_idx + num_return_sequences, total_sequences)

            for s in range(start_idx, end_idx):
                completion_tokens = completion_input_ids[s, prompt_len:]
                batch_completion_tokens.append(completion_tokens)

                completion_text = self.tokenizer.decode(
                    completion_tokens, skip_special_tokens=True
                )
                batch_completions.append(completion_text)

            completions.append(batch_completions)
            completion_tokens_list.append(batch_completion_tokens)

        completion_attention_masks = []
        for batch_tokens in completion_tokens_list:
            batch_masks = []
            for tokens in batch_tokens:
                mask = torch.ones(len(tokens), device=device)
                batch_masks.append(mask)
            completion_attention_masks.append(batch_masks)

        logits = (
            generation_output.scores if hasattr(generation_output, "scores") else []
        )

        return {
            "prompts": prompts,
            "batch_items": batch_items,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "completions": completions,
            "completion_input_ids": completion_tokens_list,
            "completion_attention_mask": completion_attention_masks,
            "logits": logits,
        }

    def _compute_rewards(
        self, prompts, completions_list, batch_items=None
    ) -> Tuple[List[float], List[List[float]]]:
        """Compute combined rewards based on multiple reward functions."""
        all_rewards = []
        all_reward_components = [[] for _ in range(len(self.reward_funcs))]

        if len(prompts) == 1:
            completions = completions_list[0]

            for completion in completions:
                weighted_reward = 0.0
                reward_components = []

                for func_idx, (reward_func, weight, processor) in enumerate(
                    zip(self.reward_funcs, self.reward_weights, self.reward_processors)
                ):
                    import inspect

                    sig = inspect.signature(reward_func)

                    if "batch_items" in sig.parameters:
                        func_rewards = reward_func(
                            [completion], batch_items=batch_items
                        )
                    else:
                        func_rewards = reward_func([completion])

                    processed_rewards = [processor(r) for r in func_rewards]

                    reward_components.append(processed_rewards[0])
                    all_reward_components[func_idx].extend(processed_rewards)

                    weighted_reward += weight * processed_rewards[0]

                all_rewards.append(weighted_reward)

            return all_rewards, all_reward_components
        else:
            raise NotImplementedError(
                "Batch processing not implemented for single agent"
            )

    def _compute_loss_with_gradients(self, completions_data, rewards):
        """Compute loss with proper gradient tracking."""
        device = self.model.device

        if len(rewards) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)

        rewards_baseline = rewards_tensor.mean()
        advantages = rewards_tensor - rewards_baseline

        advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        self.model.train()

        prompt_input_ids = completions_data["prompt_input_ids"]
        prompt_attention_mask = completions_data["prompt_attention_mask"]
        completion_input_ids = completions_data["completion_input_ids"]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        for batch_idx in range(len(prompt_input_ids)):
            prompt_ids = prompt_input_ids[batch_idx]
            prompt_mask = prompt_attention_mask[batch_idx]

            for seq_idx, completion_tokens in enumerate(
                completion_input_ids[batch_idx]
            ):
                if seq_idx >= len(advantages):
                    break

                advantage = advantages[seq_idx]

                if len(completion_tokens) > 0:
                    input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
                    target_ids = completion_tokens
                    attention_mask = torch.ones(len(input_ids), device=device)

                    outputs = self.model(
                        input_ids=input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                    )

                    completion_logits = outputs.logits[
                        0, prompt_ids.size(0) - 1 : -1, :
                    ]

                    log_probs = []
                    for i, token_id in enumerate(target_ids):
                        if i < completion_logits.size(0):
                            token_logits = completion_logits[i]
                            token_log_prob = torch.log_softmax(token_logits, dim=-1)[
                                token_id
                            ]
                            log_probs.append(token_log_prob)

                    if log_probs:
                        sequence_log_prob = torch.stack(log_probs).sum()
                        loss = -sequence_log_prob * advantage
                        total_loss = total_loss + loss
                        num_samples += 1

        if num_samples > 0:
            total_loss = total_loss / num_samples

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)

        return total_loss

    def save_model(self, output_dir):
        """Save the model."""
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)

        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        if self.wandb_initialized:
            wandb.log({"final_model_saved": output_dir})
            wandb.finish()
