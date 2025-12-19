from __future__ import annotations

import inspect
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import wandb
from comlrl.models.actor_critic import CausalLMWithValueHead
from comlrl.trainers.iac import RolloutSample

RewardFunc = Callable[..., Sequence[float]]
Formatter = Callable[[Dict[str, Any]], str]
MetricsCallback = Callable[[List["RolloutSample"]], Dict[str, float]]


@dataclass
class MAACConfig:
    """Configuration container for Multi-Agent Actor-Critic with shared critic."""

    output_dir: str = "./maac_output"
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    rollout_buffer_size: int = 8
    mini_batch_size: int = 4
    value_loss_coef: float = 0.5
    advantage_normalization: bool = True
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    do_sample: bool = True
    num_train_epochs: int = 8
    per_device_train_batch_size: int = 1
    pad_token_id: Optional[int] = None
    num_agents: int = 2
    reward_norm_eps: float = 1e-3
    num_return_sequences: int = 1
    critic_model_name_or_path: Optional[Union[str, PreTrainedModel]] = None
    num_turns: int = 1
    discount: float = 0.9
    critic_type: str = "v"  # "v" (V(s)) or "q" (Q(s,a))
    early_termination_threshold: Optional[float] = None
    eval_interval: int = 4
    eval_num_samples: int = 4

    def __post_init__(self) -> None:
        if self.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")
        if self.mini_batch_size < 1:
            raise ValueError("mini_batch_size must be >= 1.")
        if self.mini_batch_size > self.rollout_buffer_size:
            self.mini_batch_size = self.rollout_buffer_size
        if self.per_device_train_batch_size != 1:
            raise ValueError("per_device_train_batch_size must be 1 for MAAC.")
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1.")
        if self.num_return_sequences < 1:
            raise ValueError("num_return_sequences must be >= 1.")
        if self.critic_model_name_or_path is None:
            raise ValueError("critic_model_name_or_path must be provided for MAAC.")
        if self.num_turns < 1:
            raise ValueError("num_turns must be >= 1.")
        if self.num_turns > 1 and self.num_return_sequences != 1:
            raise ValueError(
                "Multi-turn MAAC currently supports num_return_sequences == 1."
            )
        if self.num_turns > 1 and self.rollout_buffer_size % self.num_turns != 0:
            raise ValueError(
                "For multi-turn MAAC, rollout_buffer_size must be a multiple of num_turns "
                "so per-turn metrics align (e.g., num_turns=2 => buffer_size even)."
            )
        critic_type = (self.critic_type or "v").lower()
        if critic_type not in ("v", "q"):
            raise ValueError("critic_type must be one of: 'v', 'q'.")
        if self.eval_interval < 0:
            raise ValueError("eval_interval must be >= 0.")
        if self.eval_num_samples < 1:
            raise ValueError("eval_num_samples must be >= 1.")


class MAACTrainer:
    """Multi-Agent Actor-Critic with a shared critic conditioned on joint prompts."""

    def __init__(
        self,
        model: Optional[Union[str, PreTrainedModel]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        reward_func: Optional[RewardFunc] = None,
        reward_processor: Optional[Callable[[float], float]] = None,
        formatters: Optional[Union[Formatter, Sequence[Formatter]]] = None,
        args: Optional[MAACConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        metrics_callback: Optional[MetricsCallback] = None,
        external_transition: Optional[Callable] = None,
    ) -> None:
        if reward_func is None or not callable(reward_func):
            raise ValueError("A callable reward_func must be provided.")
        self.args = args if args is not None else MAACConfig()
        self.reward_func = reward_func
        self.reward_processor = reward_processor or (lambda x: x)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics_callback = metrics_callback
        self.model_config = model_config or {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Training will run on CPU.")

        self.tokenizer = self._ensure_tokenizer(model, tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must expose pad_token_id.")

        self.args.pad_token_id = (
            self.args.pad_token_id
            if self.args.pad_token_id is not None
            else self.tokenizer.pad_token_id
        )

        if self.args.num_agents > 1 and isinstance(model, PreTrainedModel):
            raise ValueError(
                "Multi-agent MAAC requires `model` to be a pretrained identifier string."
            )

        if self.args.num_turns > 1 and external_transition is None:
            raise ValueError("Multi-turn MAAC requires an external_transition.")
        self.external_transition = external_transition

        self.actor_models: List[CausalLMWithValueHead] = []
        for _ in range(self.args.num_agents):
            actor_model = self._load_actor_model(model)
            actor_model.to(self.device)
            self.actor_models.append(actor_model)

        critic_identifier = self.args.critic_model_name_or_path
        if critic_identifier is None:
            raise ValueError("critic_model_name_or_path must be provided.")
        self.critic_model = self._load_critic_model(critic_identifier)
        self.critic_model.to(self.device)

        self._configure_tokenizer_specials()
        self.formatters = self._setup_formatter(formatters)
        self._reward_signature = self._infer_reward_signature(reward_func)

        self.actor_optimizers: List[torch.optim.Optimizer] = []
        for actor_model in self.actor_models:
            optimizer = torch.optim.AdamW(
                actor_model.parameters(),
                lr=self.args.actor_learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
            self.actor_optimizers.append(optimizer)

        self.critic_optimizer = torch.optim.AdamW(
            self.critic_model.parameters(),
            lr=self.args.critic_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

        self.global_step = 0
        self.rollout_buffers: List[List[RolloutSample]] = [
            [] for _ in range(self.args.num_agents)
        ]

        self.wandb_config = wandb_config
        self.wandb_initialized = False
        self.data_step = 0
        if wandb_config is not None:
            self._init_wandb()

    # ------------------------------------------------------------------ #
    # Initialisation helpers
    # ------------------------------------------------------------------ #
    def _ensure_tokenizer(
        self,
        model: Optional[Union[str, PreTrainedModel]],
        tokenizer: Optional[PreTrainedTokenizerBase],
    ) -> PreTrainedTokenizerBase:
        if tokenizer is not None:
            return tokenizer
        if model is None:
            raise ValueError(
                "Tokenizer must be provided when model is a PreTrainedModel instance."
            )
        tokenizer_kwargs = self.model_config.get("tokenizer_kwargs", {})
        return AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)

    def _load_actor_model(
        self, model: Optional[Union[str, PreTrainedModel]]
    ) -> CausalLMWithValueHead:
        if model is None:
            raise ValueError("model must be provided for MAAC.")
        model_kwargs = self.model_config.get("model_kwargs", {})
        base = (
            model
            if isinstance(model, PreTrainedModel)
            else AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        )
        return CausalLMWithValueHead(
            base_model=base, attach_value_head=False, value_head_hidden_dim=None
        )

    def _load_critic_model(
        self, model: Union[str, PreTrainedModel]
    ) -> CausalLMWithValueHead:
        model_kwargs = self.model_config.get("critic_model_kwargs", {})
        base = (
            model
            if isinstance(model, PreTrainedModel)
            else AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        )
        return CausalLMWithValueHead(
            base_model=base,
            attach_value_head=True,
            value_head_hidden_dim=self.model_config.get("critic_value_head_hidden_dim"),
        )

    def _configure_tokenizer_specials(self) -> None:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must expose pad_token_id.")

    def _setup_formatter(
        self, formatters: Optional[Union[Formatter, Sequence[Formatter]]]
    ) -> List[Formatter]:
        if formatters is None:
            raise ValueError("Formatters are required for MAAC.")
        if callable(formatters):
            return [formatters for _ in range(self.args.num_agents)]
        if isinstance(formatters, Sequence):
            if len(formatters) != self.args.num_agents:
                raise ValueError("Number of formatters must match num_agents.")
            return list(formatters)
        raise ValueError("formatters must be a callable or sequence of callables.")

    def _init_wandb(self) -> None:
        wandb_project = self.wandb_config.get("project", "maac")
        wandb_entity = self.wandb_config.get("entity")
        wandb_run_name = self.wandb_config.get("name")
        wandb_dir = self.wandb_config.get("dir")

        init_kwargs = {
            "project": wandb_project,
            "name": wandb_run_name,
            "entity": wandb_entity,
            "config": {
                "num_agents": self.args.num_agents,
                "actor_learning_rate": self.args.actor_learning_rate,
                "critic_learning_rate": self.args.critic_learning_rate,
                "max_new_tokens": self.args.max_new_tokens,
                "num_return_sequences": self.args.num_return_sequences,
            },
        }
        if wandb_dir is not None:
            os.makedirs(wandb_dir, exist_ok=True)
            init_kwargs["dir"] = wandb_dir
        tags = self.wandb_config.get("tags")
        if isinstance(tags, list):
            init_kwargs["tags"] = tags
        wandb.init(**init_kwargs)
        self.wandb_initialized = True

    # ------------------------------------------------------------------ #
    # Data utilities
    # ------------------------------------------------------------------ #
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Training requires a dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

    def get_eval_dataloader(self) -> DataLoader:
        if self.eval_dataset is None:
            raise ValueError("Evaluation requires a dataset.")
        return DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

    def _format_prompt(self, item: Dict[str, Any], agent_idx: int) -> str:
        formatter = self.formatters[agent_idx]
        prompt = formatter(item)
        if not isinstance(prompt, str):
            raise ValueError("Formatter must return a string prompt.")
        return prompt

    def _encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    def _build_joint_prompt(self, prompts: Sequence[str]) -> str:
        pieces = [f"[Agent {idx}] {p}" for idx, p in enumerate(prompts)]
        return "\n\n".join(pieces)

    def _build_critic_input(
        self, prompts: Sequence[str], action_completions: Optional[Sequence[str]] = None
    ) -> str:
        """Build centralized critic conditioning input.

        - critic_type='v': V(s) conditioned on joint prompt only.
        - critic_type='q': Q(s,a) conditioned on joint prompt + joint action text.
        """
        base = self._build_joint_prompt(prompts)
        if (self.args.critic_type or "v").lower() == "v":
            return base

        action_completions = list(action_completions or [])
        action_lines: List[str] = ["[Joint Action]"]
        for idx, comp in enumerate(action_completions):
            action_lines.append(f"[Agent {idx} action]\n{comp}")
        return base + "\n\n" + "\n\n".join(action_lines)

    def _critic_value_from_text(self, critic_input: str) -> Dict[str, Any]:
        encoded = self._encode_prompt(critic_input)
        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        prompt_len = ids.size(1)
        value = self._value_on_prompt_only(self.critic_model, ids, mask, prompt_len)
        return {
            "critic_input": critic_input,
            "input_ids": ids,
            "attention_mask": mask,
            "prompt_len": prompt_len,
            "value": value,
        }

    def _infer_reward_signature(self, reward_func: Callable) -> inspect.Signature:
        try:
            return inspect.signature(reward_func)
        except (TypeError, ValueError):
            return inspect.Signature()

    # ------------------------------------------------------------------ #
    # Rollout collection
    # ------------------------------------------------------------------ #
    def _call_reward_func(
        self, prompts: Sequence[str], agent_completions: Sequence[Sequence[str]]
    ) -> List[float]:
        signature = self._reward_signature or inspect.signature(self.reward_func)
        params = signature.parameters
        num_agents = self.args.num_agents

        def _call_with_args():
            param_count = len(params)
            if num_agents == 1:
                if param_count == 1:
                    return self.reward_func(agent_completions[0])  # type: ignore[arg-type]
                return self.reward_func(prompts, agent_completions[0])  # type: ignore[arg-type]

            if param_count == num_agents:
                return self.reward_func(*agent_completions)  # type: ignore[arg-type]
            if param_count == num_agents + 1:
                return self.reward_func(prompts, *agent_completions)  # type: ignore[arg-type]
            if param_count == 1:
                return self.reward_func(agent_completions)  # type: ignore[arg-type]
            return self.reward_func(*agent_completions)  # type: ignore[arg-type]

        try:
            raw = _call_with_args()
        except TypeError:
            raw = self.reward_func(*agent_completions)  # type: ignore[arg-type]

        if isinstance(raw, torch.Tensor):
            rewards = raw.detach().cpu().tolist()
        elif isinstance(raw, (list, tuple)):
            rewards = list(raw)
        else:
            rewards = [float(raw)]

        return [float(self.reward_processor(r)) for r in rewards]

    def _generate(self, actor_model, prompt: str) -> Dict[str, Any]:
        encoded_prompt = self._encode_prompt(prompt)
        prompt_input_ids = encoded_prompt["input_ids"]
        prompt_attention_mask = encoded_prompt["attention_mask"]
        prompt_len = prompt_input_ids.size(1)

        num_ret = int(self.args.num_return_sequences)
        generation_kwargs: Dict[str, Any] = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": bool(self.args.do_sample),
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "pad_token_id": self.args.pad_token_id,
            "num_return_sequences": num_ret,
            "num_beams": 1,
        }
        if self.args.top_k is not None:
            generation_kwargs["top_k"] = self.args.top_k

        sequences = actor_model.generate(**generation_kwargs)
        if sequences.size(1) <= prompt_len:
            raise RuntimeError("Model produced an empty completion during rollout.")

        response_tokens = sequences[:, prompt_len:]
        pad_id = self.args.pad_token_id
        response_lens: List[int] = []
        completion_texts: List[str] = []
        for seq in response_tokens:
            if pad_id is not None:
                pad_positions = (seq == pad_id).nonzero(as_tuple=False)
                resp_len = (
                    pad_positions[0].item()
                    if pad_positions.numel() > 0
                    else seq.size(0)
                )
            else:
                resp_len = seq.size(0)
            response_lens.append(resp_len)
            completion_texts.append(
                self.tokenizer.decode(seq[:resp_len], skip_special_tokens=True)
            )

        return {
            "prompt": prompt,
            "prompt_len": prompt_len,
            "sequences": sequences,
            "attention_mask": torch.ones_like(sequences, device=self.device),
            "response_lens": response_lens,
            "completions": completion_texts,
        }

    def _collect_rollouts(self, item: Dict[str, Any]) -> List[RolloutSample]:
        num_turns = max(1, int(getattr(self.args, "num_turns", 1)))
        if num_turns > 1:
            return self._collect_rollouts_multi_turn(item, num_turns)

        prompts: List[str] = []
        completions_per_agent: List[List[str]] = []
        rollout_data: List[Dict[str, Any]] = []
        num_ret = int(self.args.num_return_sequences)

        for agent_idx, actor_model in enumerate(self.actor_models):
            prompt = self._format_prompt(item, agent_idx)
            gen = self._generate(actor_model, prompt)
            prompts.append(prompt)
            completions_per_agent.append(gen["completions"])
            rollout_data.append(
                {
                    "agent_idx": agent_idx,
                    "prompt": prompt,
                    "prompt_len": gen["prompt_len"],
                    "sequences": gen["sequences"],
                    "attention_mask": gen["attention_mask"],
                    "response_lens": gen["response_lens"],
                }
            )

        rewards = self._call_reward_func(prompts, completions_per_agent)
        num_agents = self.args.num_agents
        if len(rewards) == 1:
            rewards_matrix = [[rewards[0]] * num_ret for _ in range(num_agents)]
        elif len(rewards) == num_ret:
            rewards_matrix = [list(rewards) for _ in range(num_agents)]
        elif len(rewards) == num_agents:
            rewards_matrix = [[rewards[a]] * num_ret for a in range(num_agents)]
        else:
            raise ValueError(
                "Reward function must return 1 value, num_return_sequences values, "
                "or num_agents values."
            )

        rollouts: List[RolloutSample] = []
        critic_type = (self.args.critic_type or "v").lower()
        critic_values_by_i: List[Dict[str, Any]] = []
        if critic_type == "v":
            critic_input = self._build_critic_input(prompts)
            with torch.no_grad():
                critic_values_by_i = [self._critic_value_from_text(critic_input)]
        else:
            for i in range(num_ret):
                joint_action = [
                    completions_per_agent[a][i] for a in range(self.args.num_agents)
                ]
                critic_input = self._build_critic_input(prompts, joint_action)
                with torch.no_grad():
                    critic_values_by_i.append(
                        self._critic_value_from_text(critic_input)
                    )

        for data in rollout_data:
            agent_idx = data["agent_idx"]
            for i in range(num_ret):
                seq = data["sequences"][i]
                attn = data["attention_mask"][i]
                resp_len = data["response_lens"][i]
                reward = float(rewards_matrix[agent_idx][i])
                reward_tensor = torch.tensor([reward], device=self.device)

                logprob, _ = self._policy_eval(
                    self.actor_models[agent_idx],
                    seq.unsqueeze(0),
                    attn.unsqueeze(0),
                    data["prompt_len"],
                    resp_len,
                    output_values=False,
                )

                critic_pack = (
                    critic_values_by_i[0]
                    if critic_type == "v"
                    else critic_values_by_i[i]
                )
                joint_ids = critic_pack["input_ids"]
                joint_mask = critic_pack["attention_mask"]
                joint_len = int(critic_pack["prompt_len"])
                value = critic_pack["value"].detach().cpu()
                rollouts.append(
                    RolloutSample(
                        agent_idx=agent_idx,
                        prompt=data["prompt"],
                        completion=self.tokenizer.decode(
                            seq[data["prompt_len"] : data["prompt_len"] + resp_len],
                            skip_special_tokens=True,
                        ),
                        full_input_ids=seq.detach().cpu(),
                        attention_mask=attn.detach().cpu(),
                        prompt_len=data["prompt_len"],
                        response_len=resp_len,
                        old_logprob=logprob.detach().cpu(),
                        old_value=value.detach().cpu(),
                        reward=reward_tensor.detach().cpu(),
                        returns=reward_tensor.detach().cpu(),
                        advantage=torch.zeros_like(reward_tensor).detach().cpu(),
                        normalized_advantage=None,
                        metadata={
                            "joint_input_ids": joint_ids.detach().cpu(),
                            "joint_attention_mask": joint_mask.detach().cpu(),
                            "joint_prompt_len": joint_len,
                            "turn_idx": 0,
                            "adv_target": reward_tensor.detach().cpu(),
                        },
                    )
                )

        for sample in rollouts:
            r = float(sample.reward.view(-1)[0].item())
            sample.metadata["value_target"] = torch.tensor([r]).detach().cpu()

        if self.metrics_callback is not None:
            try:
                extra = self.metrics_callback(rollouts)
                if isinstance(extra, dict):
                    self._log_metrics(extra)
            except Exception:
                pass

        return rollouts

    def _collect_rollouts_multi_turn(
        self, item: Dict[str, Any], num_turns: int
    ) -> List[RolloutSample]:
        if self.args.num_return_sequences != 1:
            raise ValueError(
                "Multi-turn MAAC currently supports num_return_sequences == 1."
            )

        prompt_history = [[] for _ in range(self.args.num_agents)]
        response_history = [[] for _ in range(self.args.num_agents)]
        previous_completions: List[Optional[str]] = [None] * self.args.num_agents
        per_agent_samples: List[List[RolloutSample]] = [
            [] for _ in range(self.args.num_agents)
        ]
        rollouts: List[RolloutSample] = []
        gamma = float(getattr(self.args, "discount", 0.9))

        for turn_idx in range(num_turns):
            if turn_idx == 0:
                turn_prompts = [
                    self._format_prompt(item, agent_idx)
                    for agent_idx in range(self.args.num_agents)
                ]
            else:
                if self.external_transition is None:
                    raise ValueError("external_transition is required for multi-turn.")
                transition_result = self.external_transition(
                    prompt=item.get("prompt", ""),
                    agent_completions=previous_completions,
                    num_agents=self.args.num_agents,
                    prompt_history_per_agent=prompt_history,
                    response_history_per_agent=response_history,
                )
                if (
                    not isinstance(transition_result, (list, tuple))
                    or len(transition_result) != self.args.num_agents
                ):
                    raise ValueError(
                        "External transition must return per-agent prompts"
                    )
                turn_prompts = list(transition_result)

            completions_per_agent: List[List[str]] = []
            rollout_data: List[Dict[str, Any]] = []
            for agent_idx, actor_model in enumerate(self.actor_models):
                prompt = turn_prompts[agent_idx]
                gen = self._generate(actor_model, prompt)
                completions_per_agent.append(gen["completions"])
                rollout_data.append(
                    {
                        "agent_idx": agent_idx,
                        "prompt": prompt,
                        "prompt_len": gen["prompt_len"],
                        "sequences": gen["sequences"],
                        "attention_mask": gen["attention_mask"],
                        "response_lens": gen["response_lens"],
                        "completion_texts": gen["completions"],
                    }
                )
                prompt_history[agent_idx].append(prompt)

            rewards = self._call_reward_func(turn_prompts, completions_per_agent)
            rewards_matrix = self._expand_rewards(rewards, num_ret=1)
            critic_input = self._build_critic_input(
                turn_prompts,
                action_completions=[c[0] for c in completions_per_agent],
            )
            with torch.no_grad():
                critic_pack = self._critic_value_from_text(critic_input)
            joint_ids = critic_pack["input_ids"]
            joint_mask = critic_pack["attention_mask"]
            joint_len = int(critic_pack["prompt_len"])
            joint_value = critic_pack["value"]

            for data in rollout_data:
                agent_idx = data["agent_idx"]
                seq = data["sequences"][0]
                attn = data["attention_mask"][0]
                resp_len = data["response_lens"][0]
                reward_val = float(rewards_matrix[agent_idx][0])
                reward_tensor = torch.tensor([reward_val], device=self.device)

                logprob, _ = self._policy_eval(
                    self.actor_models[agent_idx],
                    seq.unsqueeze(0),
                    attn.unsqueeze(0),
                    data["prompt_len"],
                    resp_len,
                    output_values=False,
                )

                value = joint_value.detach().cpu()
                completion_text = data["completion_texts"][0]
                sample = RolloutSample(
                    agent_idx=agent_idx,
                    prompt=data["prompt"],
                    completion=completion_text,
                    full_input_ids=seq.detach().cpu(),
                    attention_mask=attn.detach().cpu(),
                    prompt_len=data["prompt_len"],
                    response_len=resp_len,
                    old_logprob=logprob.detach().cpu(),
                    old_value=value.detach().cpu(),
                    reward=reward_tensor.detach().cpu(),
                    returns=reward_tensor.detach().cpu(),
                    advantage=torch.zeros_like(reward_tensor).detach().cpu(),
                    normalized_advantage=None,
                    metadata={
                        "joint_input_ids": joint_ids.detach().cpu(),
                        "joint_attention_mask": joint_mask.detach().cpu(),
                        "joint_prompt_len": joint_len,
                        "turn_idx": turn_idx,
                    },
                )
                rollouts.append(sample)
                per_agent_samples[agent_idx].append(sample)
                response_history[agent_idx].append(completion_text)
                previous_completions[agent_idx] = completion_text

            term_threshold = getattr(self.args, "early_termination_threshold", None)
            if term_threshold is not None:
                mean_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
                if mean_reward > float(term_threshold):
                    break

        for agent_idx in range(self.args.num_agents):
            traj = per_agent_samples[agent_idx]
            for t, sample in enumerate(traj):
                r = float(sample.reward.view(-1)[0].item())
                if t < len(traj) - 1:
                    next_v = float(traj[t + 1].old_value.view(-1)[0].item())
                    target = r + gamma * next_v
                else:
                    target = r
                sample.metadata["adv_target"] = torch.tensor([target]).detach().cpu()
                sample.metadata["value_target"] = torch.tensor([target]).detach().cpu()

        for agent_idx in range(self.args.num_agents):
            future = 0.0
            for sample in reversed(per_agent_samples[agent_idx]):
                immediate = float(sample.reward.view(-1)[0].item())
                future = immediate + gamma * future
                sample.returns = (
                    torch.tensor([future], device=self.device).detach().cpu()
                )
                sample.advantage = torch.zeros_like(sample.returns)
                sample.normalized_advantage = None

        if self.metrics_callback is not None:
            try:
                extra = self.metrics_callback(rollouts)
                if isinstance(extra, dict):
                    self._log_metrics(extra)
            except Exception:
                pass

        return rollouts

    def _expand_rewards(self, rewards: List[float], num_ret: int) -> List[List[float]]:
        """Map reward list to [num_agents x num_ret] matrix."""
        num_agents = self.args.num_agents
        if len(rewards) == 1:
            return [[rewards[0]] * num_ret for _ in range(num_agents)]
        if len(rewards) == num_ret:
            return [list(rewards) for _ in range(num_agents)]
        if len(rewards) == num_agents:
            return [[rewards[a]] * num_ret for a in range(num_agents)]
        raise ValueError(
            "Reward function must return 1 value, num_return_sequences values, or num_agents values."
        )

    # ------------------------------------------------------------------ #
    # Advantage prep
    # ------------------------------------------------------------------ #
    def _normalize_returns(self, rollouts: List[RolloutSample]) -> None:
        returns = torch.stack([sample.returns for sample in rollouts]).float()
        returns = returns.view(len(rollouts), -1)
        flat = returns.view(-1)
        if flat.numel() < 2:
            return

        mean = flat.mean()
        std = flat.std(unbiased=False)
        if std < self.args.reward_norm_eps:
            return
        normalized = (returns - mean) / std

        for sample, norm_value in zip(rollouts, normalized):
            norm_tensor = (
                norm_value.view_as(sample.returns)
                .to(sample.returns.dtype)
                .detach()
                .clone()
            )
            sample.returns = norm_tensor
            sample.advantage = norm_tensor - sample.old_value.to(norm_tensor.dtype)
            sample.normalized_advantage = None

    def _prepare_advantages(self, rollouts: List[RolloutSample]) -> None:
        if not rollouts:
            return
        advantages = []
        for sample in rollouts:
            target = sample.metadata.get("adv_target") or sample.metadata.get(
                "value_target"
            )
            if target is None:
                target = sample.returns
            adv = target.to(torch.float32) - sample.old_value.to(torch.float32)
            sample.advantage = adv.to(sample.returns.dtype)
            advantages.append(adv.view(-1)[0])
        advantages = torch.stack(advantages)

        if self.args.advantage_normalization and advantages.numel() > 1:
            mean = advantages.mean()
            std = advantages.std(unbiased=False).clamp(min=1e-6)
            for sample in rollouts:
                sample.normalized_advantage = (
                    sample.advantage.to(torch.float32) - mean
                ) / std
        else:
            for sample in rollouts:
                sample.normalized_advantage = sample.advantage.clone()

    # ------------------------------------------------------------------ #
    # Losses
    # ------------------------------------------------------------------ #
    def _policy_eval(
        self,
        actor_model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_len: int,
        output_values: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = actor_model(
            input_ids=sequences,
            attention_mask=attention_mask,
            output_values=output_values,
        )

        logprob = self._compute_sequence_stats(
            sequences, outputs.logits, prompt_len, response_len
        )

        value = None
        if output_values:
            value = self._value_on_prompt_only(
                actor_model, sequences, attention_mask, prompt_len
            )

        return logprob, value

    def _value_on_prompt_only(
        self,
        model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        prompt_ids = sequences[:, :prompt_len]
        prompt_mask = (
            attention_mask[:, :prompt_len] if attention_mask is not None else None
        )
        outputs = model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            output_values=True,
        )
        if outputs.values is None:
            raise RuntimeError("Value head is missing for value computation.")
        last_index = prompt_len - 1
        return outputs.values[:, last_index]

    def _compute_sequence_stats(
        self,
        sequences: torch.Tensor,
        logits: torch.Tensor,
        prompt_len: int,
        response_len: int,
    ) -> torch.Tensor:
        shifted_logits = logits[:, :-1, :]
        shifted_targets = sequences[:, 1:]

        log_probs = F.log_softmax(shifted_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shifted_targets.unsqueeze(-1)
        ).squeeze(-1)

        start_index = max(prompt_len - 1, 0)
        end_index = start_index + response_len
        response_log_probs = token_log_probs[:, start_index:end_index]

        return response_log_probs.sum(dim=-1)

    def _ac_step(self, agent_idx: int, batch: List[RolloutSample]) -> Dict[str, float]:
        actor_model = self.actor_models[agent_idx]
        actor_optimizer = self.actor_optimizers[agent_idx]

        actor_losses: List[torch.Tensor] = []
        value_losses: List[torch.Tensor] = []

        for sample in batch:
            sequences = sample.full_input_ids.to(self.device).unsqueeze(0)
            attention_mask = sample.attention_mask.to(self.device).unsqueeze(0)

            logprob, _ = self._policy_eval(
                actor_model,
                sequences,
                attention_mask,
                sample.prompt_len,
                sample.response_len,
                output_values=False,
            )

            joint_ids = sample.metadata["joint_input_ids"].to(self.device)
            joint_mask = sample.metadata["joint_attention_mask"].to(self.device)
            joint_len = int(sample.metadata["joint_prompt_len"])
            value = self._value_on_prompt_only(
                self.critic_model, joint_ids, joint_mask, joint_len
            )

            old_value = sample.old_value.to(self.device, dtype=value.dtype)
            advantage = sample.normalized_advantage.to(self.device, dtype=value.dtype)
            value_target = sample.metadata.get("value_target")
            if value_target is None:
                returns = sample.returns.to(self.device, dtype=value.dtype)
            else:
                returns = value_target.to(self.device, dtype=value.dtype)

            if not torch.isfinite(logprob).all():
                raise FloatingPointError(
                    "Encountered non-finite logprob during AC step."
                )
            if not torch.isfinite(advantage).all():
                raise FloatingPointError("Advantage contains non-finite values.")
            if not torch.isfinite(returns).all():
                raise FloatingPointError("Returns contain non-finite values.")

            policy_loss = -(logprob * advantage)
            value_error = (returns - value) ** 2

            actor_losses.append(policy_loss)
            value_losses.append(value_error)

        actor_loss = torch.stack(actor_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        value_total = self.args.value_loss_coef * value_loss
        if not torch.isfinite(actor_loss) or not torch.isfinite(value_loss):
            raise FloatingPointError("Non-finite loss detected.")

        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            actor_model.parameters(), self.args.max_grad_norm
        )
        actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_model.parameters(), self.args.max_grad_norm
        )
        self.critic_optimizer.step()

        return {
            "policy_loss": actor_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
        }

    def _update(
        self, agent_idx: int, rollouts: List[RolloutSample]
    ) -> Dict[str, float]:
        if not rollouts:
            return {}
        metrics = self._summarize_rollout_metrics(rollouts)

        self._prepare_advantages(rollouts)
        random.shuffle(rollouts)

        loss_metrics = defaultdict(list)
        for start in range(0, len(rollouts), self.args.mini_batch_size):
            batch = rollouts[start : start + self.args.mini_batch_size]
            step_metrics = self._ac_step(agent_idx, batch)
            for key, value in step_metrics.items():
                loss_metrics[key].append(value)
        averaged_losses = {
            key: float(sum(values) / len(values))
            for key, values in loss_metrics.items()
            if values
        }
        metrics.update(averaged_losses)
        return metrics

    def _summarize_rollout_metrics(
        self, rollouts: List[RolloutSample]
    ) -> Dict[str, float]:
        if not rollouts:
            return {}

        metrics: Dict[str, float] = {}
        rewards = torch.stack(
            [sample.reward.view(-1)[0] for sample in rollouts]
        ).float()
        if rewards.numel() > 0 and torch.isfinite(rewards).all():
            metrics["reward_mean"] = float(rewards.mean().item())

        returns = torch.stack(
            [sample.returns.view(-1)[0] for sample in rollouts]
        ).float()
        if returns.numel() > 0 and torch.isfinite(returns).all():
            metrics["expected_return"] = float(returns.mean().item())

        values = torch.stack(
            [sample.old_value.view(-1)[0] for sample in rollouts]
        ).float()
        if values.numel() > 0 and torch.isfinite(values).all():
            metrics["value_pred_mean"] = float(values.mean().item())

        targets = [sample.metadata.get("value_target") for sample in rollouts]
        if any(t is not None for t in targets):
            target_vals = torch.stack(
                [
                    (t if t is not None else sample.returns).view(-1)[0]
                    for sample, t in zip(rollouts, targets)
                ]
            ).float()
            if target_vals.numel() > 0 and torch.isfinite(target_vals).all():
                metrics["value_target_mean"] = float(target_vals.mean().item())

        return metrics

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataset is None:
            return {}

        dataloader = self.get_eval_dataloader()
        num_samples = int(self.args.eval_num_samples)
        turn_groups: Dict[int, List[RolloutSample]] = {}
        seen = 0

        with torch.no_grad():
            for batch in dataloader:
                for item in batch:
                    rollouts = self._collect_rollouts(item)
                    for sample in rollouts:
                        t_idx = int(sample.metadata.get("turn_idx", 0))
                        turn_groups.setdefault(t_idx, []).append(sample)
                    seen += 1
                    if seen >= num_samples:
                        break
                if seen >= num_samples:
                    break

        eval_log: Dict[str, float] = {}
        for turn_idx, samples in sorted(turn_groups.items()):
            metrics = self._summarize_rollout_metrics(samples)
            for key, value in metrics.items():
                eval_log[f"eval/turn_{turn_idx + 1}/{key}"] = value

        if eval_log:
            self._log_metrics(eval_log)
        return eval_log

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        dataloader = self.get_train_dataloader()
        total_epochs = self.args.num_train_epochs

        for epoch in range(total_epochs):
            epoch_metrics = defaultdict(list)
            for batch_idx, batch in enumerate(dataloader):
                if (
                    self.eval_dataset is not None
                    and self.args.eval_interval > 0
                    and batch_idx % int(self.args.eval_interval) == 0
                ):
                    self.evaluate()
                for item in batch:
                    rollouts = self._collect_rollouts(item)
                    for sample in rollouts:
                        agent_idx = sample.agent_idx
                        buffer = self.rollout_buffers[agent_idx]
                        buffer.append(sample)
                        if len(buffer) >= self.args.rollout_buffer_size:
                            self._process_buffer(agent_idx, buffer, epoch_metrics)
                    self.data_step += 1

            for agent_idx, buffer in enumerate(self.rollout_buffers):
                if not buffer:
                    continue
                self._process_buffer(agent_idx, buffer, epoch_metrics)

            summary = {
                key: float(sum(values) / len(values))
                for key, values in epoch_metrics.items()
                if values
            }
            num_turns = max(1, int(getattr(self.args, "num_turns", 1)))
            epoch_log: Dict[str, float] = {}
            for turn_idx in range(num_turns):
                prefix = f"turn_{turn_idx + 1}/"

                def _maybe_log(metric_key: str, epoch_key: str) -> None:
                    values = epoch_metrics.get(prefix + metric_key)
                    if values:
                        epoch_log[prefix + epoch_key] = float(sum(values) / len(values))

                _maybe_log("reward_mean", "epoch_reward_mean")
                _maybe_log("expected_return", "epoch_avg_return")
                _maybe_log("value_pred_mean", "epoch_value_pred_mean")
                _maybe_log("value_target_mean", "epoch_value_target_mean")
                _maybe_log("policy_loss", "epoch_policy_loss")
                _maybe_log("value_loss", "epoch_value_loss")

            if epoch_log:
                self._log_metrics(epoch_log)

            if summary:
                to_print = epoch_log if epoch_log else summary
                print(f"Epoch {epoch + 1}/{total_epochs} metrics: {to_print}")

    # ------------------------------------------------------------------ #
    # Logging and persistence
    # ------------------------------------------------------------------ #
    def _tag_metrics(
        self, metrics: Dict[str, float], agent_idx: int, turn_idx: int = 0
    ) -> Dict[str, float]:
        prefix = f"turn_{turn_idx + 1}/"
        return {prefix + key: value for key, value in metrics.items()}

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        if self.wandb_initialized and wandb is not None:
            wandb.log(metrics, step=self.data_step)

    def _process_buffer(
        self,
        agent_idx: int,
        buffer: List[RolloutSample],
        epoch_metrics: Dict[str, List[float]],
    ) -> None:
        if not buffer:
            return

        has_turn_idx = any(
            "turn_idx" in (getattr(s, "metadata", {}) or {}) for s in buffer
        )
        turn_groups: Dict[int, List[RolloutSample]] = {}
        for sample in buffer:
            t_idx = int(sample.metadata.get("turn_idx", 0)) if has_turn_idx else 0
            turn_groups.setdefault(t_idx, []).append(sample)

        buffer.clear()

        combined_log: Dict[str, float] = {}
        for t_idx in sorted(turn_groups.keys()):
            samples = turn_groups[t_idx]
            metrics = self._update(agent_idx, samples)
            tagged = self._tag_metrics(metrics, agent_idx, turn_idx=t_idx)
            combined_log.update(tagged)
            for key, value in tagged.items():
                epoch_metrics[key].append(value)

        self._log_metrics(combined_log)

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for agent_idx, actor in enumerate(self.actor_models):
            agent_dir = os.path.join(output_dir, f"agent_{agent_idx}")
            os.makedirs(agent_dir, exist_ok=True)
            actor.model.save_pretrained(agent_dir)
        critic_dir = os.path.join(output_dir, "critic")
        os.makedirs(critic_dir, exist_ok=True)
        self.critic_model.model.save_pretrained(critic_dir)
        if self.critic_model.value_head is not None:
            torch.save(
                self.critic_model.value_head.state_dict(),
                os.path.join(critic_dir, "value_head.pt"),
            )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
