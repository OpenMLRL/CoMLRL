import inspect
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from comlrl.models.actor_critic import CausalLMWithValueHead
from comlrl.schedulers import DeviceScheduler
from comlrl.utils.distributed import local_context, unwrap_model
from comlrl.utils.formatters import build_formatters
from comlrl.utils.model_loading import resolve_model_sources
from comlrl.utils.reference_kl import (
    clone_reference_models,
    reference_kl_coef,
    reference_kl_enabled,
    reference_kl_for_sequence,
    validate_reference_kl_config,
)
from comlrl.utils.reward_utils import call_reward_function, normalize_reward_lengths
from comlrl.utils.tokenizer_utils import apply_tokenizer_specials, resolve_tokenizers
from .ac_base import ActorCriticTrainerBase
import wandb


RewardFunc = Callable[..., Sequence[float]]
Formatter = Callable[[Dict[str, Any]], str]
MetricsCallback = Callable[[List["RolloutSample"]], Dict[str, float]]


@dataclass
class IACConfig:
    """Configuration container for Independent Actor-Critic fine-tuning."""

    agent_learning_rate: float = 5e-6
    critic_learning_rate: Optional[float] = 5e-6
    rollout_buffer_size: int = 8
    train_batch_size: Optional[int] = None
    value_clip_range: Optional[float] = 0.2
    value_loss_coef: float = 0.6
    advantage_normalization: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.6
    top_k: Optional[int] = None
    num_train_epochs: int = 40
    use_separate_critic: bool = True
    critic_type: str = "v"  # "v" (V(h)) or "q" (Q(h,a))
    critic_value_head_hidden_dim: Optional[int] = None
    value_head_hidden_dim: Optional[int] = None
    num_agents: int = 2
    num_turns: int = 2
    parallel_training: str = "none"
    agent_devices: Optional[Union[str, Sequence[str]]] = None
    critic_devices: Optional[Union[str, Sequence[str]]] = None
    external_prompt_passthrough: bool = False
    discount: float = 0.9
    num_generations: int = 1
    eval_interval: int = 16
    eval_num_samples: int = 4
    eval_batch_size: int = 1
    early_termination_threshold: Optional[float] = -0.2
    logging_steps: int = 1
    reference_kl_enabled: bool = False
    reference_kl_coef: float = 0.1

    def __post_init__(self) -> None:
        if self.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")
        if self.train_batch_size is None:
            self.train_batch_size = self.rollout_buffer_size
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1.")
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1.")
        if self.num_turns < 1:
            raise ValueError("num_turns must be >= 1.")
        if self.num_turns > 1 and self.num_generations != 1:
            raise ValueError("Multi-turn IAC currently supports num_generations == 1.")
        critic_type = (self.critic_type or "v").lower()
        if critic_type not in ("v", "q"):
            raise ValueError("critic_type must be one of: 'v', 'q'.")
        if self.critic_learning_rate is None:
            self.critic_learning_rate = self.agent_learning_rate
        if self.num_generations < 1:
            raise ValueError("num_generations must be >= 1.")
        if self.eval_interval < 0:
            raise ValueError("eval_interval must be >= 0.")
        if self.eval_num_samples < 0:
            raise ValueError("eval_num_samples must be >= 0.")
        if self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be >= 1.")
        if self.logging_steps < 1:
            raise ValueError("logging_steps must be >= 1.")
        mode = str(self.parallel_training or "none").strip().lower()
        if mode == "null":
            mode = "none"
        if mode not in {"none", "mp"}:
            raise ValueError("parallel_training only supports: none, mp.")
        if mode == "mp":
            if self.agent_devices is None:
                raise ValueError(
                    "parallel_training='mp' requires explicit agent_devices."
                )
            if self.use_separate_critic and self.critic_devices is None:
                raise ValueError(
                    "parallel_training='mp' requires explicit critic_devices when "
                    "use_separate_critic=True."
                )
        self.parallel_training = mode
        validate_reference_kl_config(self, self.num_agents)


@dataclass
class RolloutSample:
    agent_idx: int
    prompt: str
    completion: str
    full_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int
    response_len: int
    old_logprob: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    returns: torch.Tensor
    advantage: torch.Tensor
    normalized_advantage: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IACTrainer(ActorCriticTrainerBase):
    """Independent Actor-Critic trainer with optional separate critic support."""

    algorithm_name: str = "IAC"

    def __init__(
        self,
        agent_model: Optional[Union[str, PreTrainedModel]] = None,
        critic_model: Optional[Union[str, PreTrainedModel]] = None,
        tokenizer: Optional[
            Union[PreTrainedTokenizerBase, Sequence[PreTrainedTokenizerBase]]
        ] = None,
        reward_func: Optional[RewardFunc] = None,
        reward_processor: Optional[Callable[[float], float]] = None,
        formatters: Optional[Union[Formatter, Sequence[Formatter]]] = None,
        args: Optional[IACConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        metrics_callback: Optional[MetricsCallback] = None,
        external_transition: Optional[Callable] = None,
        agents: Optional[
            Sequence[Union[str, PreTrainedModel, CausalLMWithValueHead]]
        ] = None,
        critics: Optional[
            Sequence[Union[str, PreTrainedModel, CausalLMWithValueHead]]
        ] = None,
    ) -> None:
        self.args = args if args is not None else IACConfig()
        if reward_func is None or not callable(reward_func):
            raise ValueError("reward_func must be a callable.")
        if agent_model is None and agents is None:
            raise ValueError("Either agent_model or agents must be provided.")
        if not self.args.use_separate_critic and (
            critics is not None or critic_model is not None
        ):
            raise ValueError(
                "critics can only be provided when use_separate_critic=True."
            )
        if (
            agents is None
            and self.args.num_agents > 1
            and isinstance(agent_model, PreTrainedModel)
        ):
            raise ValueError(
                "Multi-agent IAC requires `agent_model` to be a pretrained identifier string."
            )
        if agents is not None and tokenizer is None:
            raise ValueError("Tokenizer must be provided when using agents.")
        if self.args.num_turns > 1 and external_transition is None:
            raise ValueError("Multi-turn IAC requires an external_transition.")
        self.reward_func = reward_func
        self.reward_processor = reward_processor or (lambda x: x)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics_callback = metrics_callback
        self.model_config = model_config or {}
        self.critic_type = (self.args.critic_type or "v").lower()
        self.parallel_training = (
            str(getattr(self.args, "parallel_training", "none")).strip().lower()
        )
        if self.parallel_training == "null":
            self.parallel_training = "none"
        if self.parallel_training not in {"none", "mp"}:
            raise ValueError("parallel_training only supports: none, mp.")
        if self.parallel_training == "mp":
            self.agent_devices, self.critic_devices = DeviceScheduler.assign_devices(
                self.args.num_agents,
                getattr(self.args, "agent_devices", None),
                getattr(self.args, "critic_devices", None),
                use_separate_critic=self.args.use_separate_critic,
            )
        else:
            single_device = DeviceScheduler.resolve_single_device(
                getattr(self.args, "agent_devices", None),
                getattr(self.args, "critic_devices", None),
            )
            self.agent_devices = [single_device] * self.args.num_agents
            self.critic_devices = [single_device] * self.args.num_agents
        self.device = self.agent_devices[0]
        self.dist_env = local_context(self.device)

        self.agents: List[CausalLMWithValueHead] = []
        self.critics: List[CausalLMWithValueHead] = []

        tokenizers = resolve_tokenizers(agent_model, tokenizer, agents)
        if isinstance(tokenizers, list):
            self.tokenizers = tokenizers
            self.tokenizer = tokenizers[0] if tokenizers else None
        else:
            self.tokenizers = [tokenizers] * self.args.num_agents
            self.tokenizer = tokenizers
        self.formatters = build_formatters(formatters, self.args.num_agents)
        try:
            self._reward_signature = inspect.signature(reward_func)
        except (TypeError, ValueError):
            self._reward_signature = None
        self.external_transition = external_transition

        actor_sources, _agent_name = resolve_model_sources(
            kind="agents",
            model=agent_model,
            models=agents,
            expected_count=self.args.num_agents,
            model_label="agent_model",
        )
        for idx, actor_source in enumerate(actor_sources):
            if actor_source is None:
                raise ValueError("A policy model identifier or instance is required.")
            if isinstance(actor_source, CausalLMWithValueHead):
                agent_model = actor_source
            elif isinstance(actor_source, PreTrainedModel):
                base_model = actor_source
                attach_value = not self.args.use_separate_critic
                agent_model = CausalLMWithValueHead(
                    base_model,
                    value_head_hidden_dim=self.args.value_head_hidden_dim,
                    attach_value_head=attach_value,
                )
            else:
                model_kwargs = self._filter_model_kwargs(
                    self.model_config.get("model_kwargs", {})
                )
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        actor_source, **model_kwargs
                    )
                except (OSError, ValueError) as exc:
                    raise ValueError(
                        f"Failed to load actor model from identifier '{actor_source}'."
                    ) from exc
                attach_value = not self.args.use_separate_critic
                agent_model = CausalLMWithValueHead(
                    base_model,
                    value_head_hidden_dim=self.args.value_head_hidden_dim,
                    attach_value_head=attach_value,
                )
            agent_model.to(self.agent_devices[idx])
            self.agents.append(agent_model)

        if self.args.use_separate_critic:
            if critics is None and critic_model is None:
                raise ValueError(
                    "Either critic_model or critics must be provided when use_separate_critic=True."
                )
            critic_sources, _critic_name = resolve_model_sources(
                kind="critics",
                model=critic_model,
                models=critics,
                expected_count=self.args.num_agents,
                model_label="critic_model",
            )
            for idx, critic_source in enumerate(critic_sources):
                if isinstance(critic_source, CausalLMWithValueHead):
                    critic_model = critic_source
                elif isinstance(critic_source, PreTrainedModel):
                    base_model = critic_source
                    critic_model = CausalLMWithValueHead(
                        base_model,
                        value_head_hidden_dim=self.args.critic_value_head_hidden_dim,
                        attach_value_head=True,
                    )
                else:
                    model_kwargs = self._filter_model_kwargs(
                        self.model_config.get("critic_model_kwargs", {})
                    )
                    try:
                        base_model = AutoModelForCausalLM.from_pretrained(
                            critic_source, **model_kwargs
                        )
                    except (OSError, ValueError) as exc:
                        raise ValueError(
                            f"Failed to load critic model from identifier '{critic_source}'."
                        ) from exc
                    critic_model = CausalLMWithValueHead(
                        base_model,
                        value_head_hidden_dim=self.args.critic_value_head_hidden_dim,
                        attach_value_head=True,
                    )
                critic_model.to(self.critic_devices[idx])
                self.critics.append(critic_model)
        else:
            self.critics = []

        self.reference_models: List[Any] = []
        if reference_kl_enabled(self.args):
            self.reference_models = clone_reference_models(
                self.agents,
                devices=self.agent_devices,
            )

        if self.tokenizers and len(self.tokenizers) == len(self.agents):
            for idx, tok in enumerate(self.tokenizers):
                models = [self.agents[idx]]
                if idx < len(self.critics):
                    models.append(self.critics[idx])
                if idx < len(self.reference_models):
                    models.append(self.reference_models[idx])
                apply_tokenizer_specials(tok, models)
        else:
            apply_tokenizer_specials(
                self.tokenizer, [*self.agents, *self.critics, *self.reference_models]
            )

        self.agent_optimizers = []
        self.critic_optimizers = []

        for agent_model in self.agents:
            optimizer = torch.optim.AdamW(
                agent_model.parameters(),
                lr=self.args.agent_learning_rate,
            )
            self.agent_optimizers.append(optimizer)

        if self.args.use_separate_critic:
            for critic_model in self.critics:
                optimizer = torch.optim.AdamW(
                    critic_model.parameters(),
                    lr=self.args.critic_learning_rate,
                )
                self.critic_optimizers.append(optimizer)

        self.env_step = 0
        self.rollout_buffers = [[] for _ in range(self.args.num_agents)]
        self.wandb_config = wandb_config
        self.wandb_initialized = False
        self._last_train_log_step = -1
        if wandb_config is not None and self.dist_env.is_main:
            self._init_wandb()
        self.verbose = True
        if isinstance(self.wandb_config, dict):
            sections = self.wandb_config.get("config_sections", {})
            if isinstance(sections, dict):
                out = sections.get("output", {})
                if isinstance(out, dict) and "verbose" in out:
                    self.verbose = bool(out.get("verbose"))

    def _agent_device(self, agent_idx: int) -> torch.device:
        return self.agent_devices[agent_idx]

    def _critic_device(self, agent_idx: int) -> torch.device:
        if self.args.use_separate_critic:
            return self.critic_devices[agent_idx]
        return self.agent_devices[agent_idx]

    def _init_wandb(self) -> None:
        if not self.dist_env.is_main:
            return
        if self.wandb_initialized:
            return
        if wandb is None:
            raise RuntimeError("wandb is not installed but wandb_config was provided.")
        if self.wandb_config is None:
            self.wandb_config = {}

        wandb_project = self.wandb_config.get("project", "comlrl")
        wandb_entity = self.wandb_config.get("entity")
        algo_tag = str(self.algorithm_name or "iac").lower()
        wandb_name = (
            self.wandb_config.get("name")
            or self.wandb_config.get("run_name")
            or f"test-{algo_tag}"
        )
        wandb_dir = self.wandb_config.get("dir")

        config_dict: Dict[str, Any] = {
            "algorithm": self.algorithm_name,
            "num_agents": self.args.num_agents,
            "num_turns": self.args.num_turns,
            "agent_learning_rate": self.args.agent_learning_rate,
            "critic_learning_rate": self.args.critic_learning_rate,
            "rollout_buffer_size": self.args.rollout_buffer_size,
            "train_batch_size": self.args.train_batch_size,
            "value_loss_coef": self.args.value_loss_coef,
            "max_new_tokens": self.args.max_new_tokens,
            "use_separate_critic": self.args.use_separate_critic,
            "critic_type": self.args.critic_type,
            "reference_kl_enabled": reference_kl_enabled(self.args),
            "reference_kl_coef": reference_kl_coef(self.args),
        }

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

            config_dict.update(
                {
                    "dataset": dataset_section,
                    "output": output_section,
                    "external": external_section,
                    "trainer": trainer_section,
                }
            )

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

        init_kwargs: Dict[str, Any] = {
            "project": wandb_project,
            "entity": wandb_entity,
            "name": wandb_name,
            "config": config_dict,
        }

        if wandb_dir is not None:
            os.makedirs(wandb_dir, exist_ok=True)
            init_kwargs["dir"] = wandb_dir

        tags = self.wandb_config.get("tags")
        if isinstance(tags, list):
            init_kwargs["tags"] = tags

        wandb.init(**init_kwargs)
        self.wandb_initialized = True

    # Rollout collection
    def _generate_rollout(
        self,
        agent_model: CausalLMWithValueHead,
        prompt: str,
        agent_idx: int,
        num_ret: int,
    ) -> Dict[str, Any]:
        agent_device = self._agent_device(agent_idx)
        encoded_prompt = self._encode_prompt(
            prompt, agent_idx=agent_idx, device=agent_device
        )
        prompt_input_ids = encoded_prompt["input_ids"]
        prompt_attention_mask = encoded_prompt["attention_mask"]
        prompt_len = prompt_input_ids.size(1)

        generation_kwargs: Dict[str, Any] = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": True,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "num_return_sequences": (
                1 if bool(getattr(self, "_in_eval", False)) else num_ret
            ),
            "num_beams": 1,
        }
        if self.args.top_k is not None:
            generation_kwargs["top_k"] = self.args.top_k

        generation_model = unwrap_model(agent_model)
        sequences = generation_model.generate(**generation_kwargs)
        if sequences.size(1) <= prompt_len:
            raise RuntimeError("Model produced an empty completion during rollout.")

        response_tokens = sequences[:, prompt_len:]
        tokenizer = self._get_tokenizer(agent_idx)
        pad_id = tokenizer.pad_token_id
        response_lens: List[int] = []
        completion_texts: List[str] = []
        for seq in response_tokens:
            pad_positions = (seq == pad_id).nonzero(as_tuple=False)
            resp_len = (
                pad_positions[0].item() if pad_positions.numel() > 0 else seq.size(0)
            )
            response_lens.append(resp_len)
            completion_texts.append(
                tokenizer.decode(seq[:resp_len], skip_special_tokens=True)
            )

        full_attention_mask = torch.ones_like(sequences, device=agent_device)

        with torch.no_grad():
            if self.args.use_separate_critic:
                critic_model = self.critics[agent_idx]
                if critic_model is None:
                    raise RuntimeError("Critic model missing for agent.")
                critic_device = self._critic_device(agent_idx)
                critic_sequences = sequences.to(critic_device)
                critic_mask = full_attention_mask.to(critic_device)
                value = self._value_for_critic_type(
                    critic_model,
                    critic_sequences,
                    critic_mask,
                    prompt_len,
                    response_lens,
                )
            else:
                value = self._value_for_critic_type(
                    agent_model,
                    sequences,
                    full_attention_mask,
                    prompt_len,
                    response_lens,
                )

        logprobs = []
        for seq, attn, resp_len in zip(sequences, full_attention_mask, response_lens):
            lp, _ = self._policy_eval(
                agent_model,
                seq.unsqueeze(0),
                attn.unsqueeze(0),
                prompt_len,
                resp_len,
                output_values=False,
            )
            logprobs.append(lp.squeeze(0))
        reference_kls = self._reference_kl_values(
            agent_idx,
            agent_model,
            sequences,
            full_attention_mask,
            prompt_len,
            response_lens,
        )

        return {
            "prompt": prompt,
            "prompt_len": prompt_len,
            "sequences": sequences,
            "attention_mask": full_attention_mask,
            "response_lens": response_lens,
            "logprobs": logprobs,
            "reference_kls": reference_kls,
            "values": value,
            "completions": completion_texts,
            "char_lengths": [len(txt) for txt in completion_texts],
        }

    def _reference_kl_values(
        self,
        agent_idx: int,
        policy_model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_lens: Sequence[int],
    ) -> List[float]:
        if not reference_kl_enabled(self.args):
            return []
        if not getattr(self, "reference_models", None):
            raise RuntimeError(
                "Reference KL is enabled but reference models are missing."
            )
        reference_model = self.reference_models[agent_idx]
        values: List[float] = []
        for seq, attn, response_len in zip(sequences, attention_mask, response_lens):
            kl_value = reference_kl_for_sequence(
                policy_model,
                reference_model,
                seq.unsqueeze(0),
                attn.unsqueeze(0),
                prompt_len,
                int(response_len),
            )
            values.append(float(kl_value.view(-1)[0].item()))
        return values

    def _kl_shaped_reward(
        self, reward: float, data: Dict[str, Any], index: int
    ) -> Tuple[float, Dict[str, float]]:
        if not reference_kl_enabled(self.args):
            return reward, {}
        reference_kl = 0.0
        penalty = 0.0
        raw_kls = data.get("reference_kls") or []
        if index < len(raw_kls):
            reference_kl = float(raw_kls[index])
        penalty = reference_kl_coef(self.args) * reference_kl
        return reward - penalty, {
            "reference_kl": reference_kl,
            "reference_kl_penalty": penalty,
            "environment_reward": reward,
        }

    def _expand_rewards(self, rewards: List[float], num_ret: int) -> List[List[float]]:
        num_agents = self.args.num_agents
        if len(rewards) == 1:
            return [[rewards[0]] * num_ret for _ in range(num_agents)]
        if len(rewards) == num_ret:
            return [list(rewards) for _ in range(num_agents)]
        if len(rewards) == num_agents:
            return [[rewards[a]] * num_ret for a in range(num_agents)]
        raise ValueError(
            "Reward function must return 1 value, num_generations values, or num_agents values."
        )

    def _collect_rollouts(self, item: Dict[str, Any]) -> List[RolloutSample]:
        num_turns = max(1, int(getattr(self.args, "num_turns", 1)))
        if num_turns > 1:
            return self._collect_rollouts_multi_turn(item, num_turns)

        num_ret = (
            1
            if bool(getattr(self, "_in_eval", False))
            else int(getattr(self.args, "num_generations", 1))
        )
        turn_prompts = [
            self._resolve_turn_prompt(item, agent_idx)
            for agent_idx in range(self.args.num_agents)
        ]

        def _generate_agent(agent_idx: int) -> Dict[str, Any]:
            agent_model = self.agents[agent_idx]
            prompt = turn_prompts[agent_idx]
            gen = self._generate_rollout(agent_model, prompt, agent_idx, num_ret)
            return {"agent_idx": agent_idx, **gen}

        rollout_data = self._run_agent_tasks(_generate_agent)
        prompts: List[str] = [entry["prompt"] for entry in rollout_data]
        completions_per_agent: List[List[str]] = [
            entry["completions"] for entry in rollout_data
        ]

        rewards = call_reward_function(
            self.reward_func,
            prompts,
            completions_per_agent,
            num_agents=self.args.num_agents,
            batch_items=[item],
            signature=self._reward_signature,
        )
        rewards = normalize_reward_lengths(
            [float(self.reward_processor(r)) for r in rewards],
            num_agents=self.args.num_agents,
            num_generations=num_ret,
            algorithm="IAC",
        )
        rewards_matrix = self._expand_rewards(rewards, num_ret=num_ret)

        rollouts: List[RolloutSample] = []
        for data in rollout_data:
            agent_idx = data["agent_idx"]
            for i in range(num_ret):
                seq = data["sequences"][i]
                attn = data["attention_mask"][i]
                resp_len = data["response_lens"][i]
                logprob = data["logprobs"][i]
                value = data["values"][i]
                reward = float(rewards_matrix[agent_idx][i])
                reward_cpu = torch.tensor([reward], dtype=torch.float32)
                shaped_reward, kl_meta = self._kl_shaped_reward(reward, data, i)
                shaped_reward_cpu = torch.tensor([shaped_reward], dtype=torch.float32)
                value_cpu = value.detach().cpu()
                returns_cpu = shaped_reward_cpu.clone()
                advantage_cpu = returns_cpu - value_cpu.to(dtype=returns_cpu.dtype)
                logprob_cpu = logprob.detach().cpu()

                rollouts.append(
                    RolloutSample(
                        agent_idx=agent_idx,
                        prompt=data["prompt"],
                        completion=self._get_tokenizer(agent_idx).decode(
                            seq[data["prompt_len"] : data["prompt_len"] + resp_len],
                            skip_special_tokens=True,
                        ),
                        full_input_ids=seq.detach().cpu(),
                        attention_mask=attn.detach().cpu(),
                        prompt_len=data["prompt_len"],
                        response_len=resp_len,
                        old_logprob=logprob_cpu,
                        old_value=value_cpu,
                        reward=reward_cpu,
                        returns=returns_cpu,
                        advantage=advantage_cpu,
                        metadata={
                            "char_length": data["char_lengths"][i],
                            **kl_meta,
                            "value_target": returns_cpu,
                        },
                    )
                )

        return rollouts

    def _collect_rollouts_multi_turn(
        self, item: Dict[str, Any], num_turns: int
    ) -> List[RolloutSample]:
        if self.args.num_generations != 1:
            raise ValueError("Multi-turn IAC currently supports num_generations == 1.")
        if self.external_transition is None:
            raise ValueError("external_transition is required for multi-turn IAC.")

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
                    self._resolve_turn_prompt(item, agent_idx)
                    for agent_idx in range(self.args.num_agents)
                ]
            else:
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
                external_prompts = list(transition_result)
                turn_prompts = [
                    self._resolve_turn_prompt(
                        item, agent_idx, external_prompt=external_prompts[agent_idx]
                    )
                    for agent_idx in range(self.args.num_agents)
                ]

            completions_per_agent: List[List[str]] = []
            for agent_idx, prompt in enumerate(turn_prompts):
                prompt_history[agent_idx].append(prompt)

            def _generate_agent_turn(agent_idx: int) -> Dict[str, Any]:
                agent_model = self.agents[agent_idx]
                prompt = turn_prompts[agent_idx]
                gen = self._generate_rollout(agent_model, prompt, agent_idx, num_ret=1)
                return {"agent_idx": agent_idx, **gen}

            rollout_data = self._run_agent_tasks(_generate_agent_turn)
            completions_per_agent = [entry["completions"] for entry in rollout_data]

            rewards = call_reward_function(
                self.reward_func,
                turn_prompts,
                completions_per_agent,
                num_agents=self.args.num_agents,
                batch_items=[item],
                signature=self._reward_signature,
            )
            rewards = normalize_reward_lengths(
                [float(self.reward_processor(r)) for r in rewards],
                num_agents=self.args.num_agents,
                num_generations=1,
                algorithm="IAC",
            )
            rewards_matrix = self._expand_rewards(rewards, num_ret=1)

            for data in rollout_data:
                agent_idx = data["agent_idx"]
                seq = data["sequences"][0]
                attn = data["attention_mask"][0]
                resp_len = data["response_lens"][0]
                logprob = data["logprobs"][0]
                value = data["values"][0]
                reward_val = float(rewards_matrix[agent_idx][0])
                reward_cpu = torch.tensor([reward_val], dtype=torch.float32)
                shaped_reward, kl_meta = self._kl_shaped_reward(reward_val, data, 0)
                shaped_reward_cpu = torch.tensor([shaped_reward], dtype=torch.float32)
                value_cpu = value.detach().cpu()
                logprob_cpu = logprob.detach().cpu()

                completion_text = data["completions"][0]
                sample = RolloutSample(
                    agent_idx=agent_idx,
                    prompt=data["prompt"],
                    completion=completion_text,
                    full_input_ids=seq.detach().cpu(),
                    attention_mask=attn.detach().cpu(),
                    prompt_len=data["prompt_len"],
                    response_len=resp_len,
                    old_logprob=logprob_cpu,
                    old_value=value_cpu,
                    reward=reward_cpu,
                    returns=shaped_reward_cpu.clone(),
                    advantage=torch.zeros_like(reward_cpu),
                    normalized_advantage=None,
                    metadata={
                        "char_length": data["char_lengths"][0],
                        "turn_idx": turn_idx,
                        **kl_meta,
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
                r = float(sample.returns.view(-1)[0].item())
                if t < len(traj) - 1:
                    next_v = float(traj[t + 1].old_value.view(-1)[0].item())
                    target = r + gamma * next_v
                else:
                    target = r
                sample.metadata["adv_target"] = torch.tensor([target]).cpu()
                sample.metadata["value_target"] = torch.tensor([target]).cpu()

        for agent_idx in range(self.args.num_agents):
            future = 0.0
            for sample in reversed(per_agent_samples[agent_idx]):
                immediate = float(sample.returns.view(-1)[0].item())
                future = immediate + gamma * future
                sample.returns = torch.tensor([future], dtype=torch.float32)
                sample.advantage = torch.zeros_like(sample.returns)
                sample.normalized_advantage = None

        return rollouts

    def _policy_eval(
        self,
        agent_model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_len: int,
        output_values: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate the actor to retrieve log-probabilities and (optional) value prediction.
        """

        outputs = agent_model(
            input_ids=sequences,
            attention_mask=attention_mask,
            output_values=output_values,
        )

        logprob = self._compute_sequence_stats(
            sequences, outputs.logits, prompt_len, response_len
        )

        value = None
        if output_values:
            # Use prompt-only values for V(h); use full sequence for Q(h,a).
            value = self._value_for_critic_type(
                agent_model,
                sequences,
                attention_mask,
                prompt_len,
                [response_len],
            )

        return logprob, value

    def _value_on_prompt_only(
        self,
        model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Compute V(h) using only the prompt/history tokens, excluding actions.
        """
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

    def _value_on_full_sequence(
        self,
        model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_lens: Sequence[int],
    ) -> torch.Tensor:
        """Compute Q(h,a) using the prompt+response sequence."""
        values: List[torch.Tensor] = []
        for seq, attn, resp_len in zip(sequences, attention_mask, response_lens):
            seq_len = int(prompt_len) + int(resp_len)
            seq_ids = seq[:seq_len].unsqueeze(0)
            seq_mask = attn[:seq_len].unsqueeze(0) if attn is not None else None
            outputs = model(
                input_ids=seq_ids,
                attention_mask=seq_mask,
                output_values=True,
            )
            if outputs.values is None:
                raise RuntimeError("Value head is missing for value computation.")
            values.append(outputs.values[:, -1])
        return torch.cat(values, dim=0)

    def _value_for_critic_type(
        self,
        model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_lens: Optional[Sequence[int]],
    ) -> torch.Tensor:
        if self.critic_type == "q":
            if response_lens is None:
                seq_len = sequences.size(1)
                response_lens = [seq_len - int(prompt_len)] * sequences.size(0)
            return self._value_on_full_sequence(
                model, sequences, attention_mask, prompt_len, response_lens
            )
        return self._value_on_prompt_only(model, sequences, attention_mask, prompt_len)

    def _critic_eval(
        self,
        critic_model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_len: int,
    ) -> torch.Tensor:
        return self._value_for_critic_type(
            critic_model,
            sequences,
            attention_mask,
            prompt_len,
            [response_len],
        )

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

        logprob_sum = response_log_probs.sum(dim=-1)

        return logprob_sum

    # Actor-Critic update logic
    def _ac_step(self, agent_idx: int, batch: List[RolloutSample]) -> Dict[str, float]:
        agent_model = self.agents[agent_idx]
        critic_model = (
            self.critics[agent_idx] if self.args.use_separate_critic else None
        )
        agent_optimizer = self.agent_optimizers[agent_idx]
        critic_optimizer = (
            self.critic_optimizers[agent_idx] if self.args.use_separate_critic else None
        )

        actor_losses: List[torch.Tensor] = []
        value_losses: List[torch.Tensor] = []
        agent_device = self._agent_device(agent_idx)
        critic_device = self._critic_device(agent_idx)

        for sample in batch:
            sequences = sample.full_input_ids.to(agent_device).unsqueeze(0)
            attention_mask = sample.attention_mask.to(agent_device).unsqueeze(0)

            # Policy log-prob uses full sequence; value uses prompt-only baseline.
            logprob, _ = self._policy_eval(
                agent_model,
                sequences,
                attention_mask,
                sample.prompt_len,
                sample.response_len,
                output_values=False,
            )
            if self.args.use_separate_critic:
                if critic_model is None:
                    raise RuntimeError("Critic model not initialised.")
                critic_sequences = sequences.to(critic_device)
                critic_attention_mask = attention_mask.to(critic_device)
                value = self._critic_eval(
                    critic_model,
                    critic_sequences,
                    critic_attention_mask,
                    sample.prompt_len,
                    sample.response_len,
                )
            else:
                value = self._value_for_critic_type(
                    agent_model,
                    sequences,
                    attention_mask,
                    sample.prompt_len,
                    [sample.response_len],
                )

            value_device = value.device
            old_value = sample.old_value.to(value_device, dtype=value.dtype)
            old_logprob = sample.old_logprob.to(agent_device)
            policy_advantage = sample.normalized_advantage.to(
                agent_device, dtype=logprob.dtype
            )
            returns = sample.returns.to(value_device, dtype=value.dtype)

            if (
                not torch.isfinite(logprob).all()
                or not torch.isfinite(old_logprob).all()
            ):
                raise FloatingPointError(
                    "Encountered non-finite logprob during AC step."
                )
            if not torch.isfinite(policy_advantage).all():
                raise FloatingPointError("Advantage contains non-finite values.")
            if not torch.isfinite(returns).all():
                raise FloatingPointError("Returns contain non-finite values.")

            policy_loss = -(logprob * policy_advantage)

            value_target = sample.metadata.get("value_target")
            if value_target is None:
                raise RuntimeError("value_target missing for critic update.")
            value_target = value_target.to(value_device, dtype=value.dtype)
            if (
                self.args.value_clip_range is not None
                and not self.args.use_separate_critic
            ):
                clipped_value = old_value + torch.clamp(
                    value - old_value,
                    -self.args.value_clip_range,
                    self.args.value_clip_range,
                )
                value_error = torch.max(
                    (value_target - value) ** 2,
                    (value_target - clipped_value) ** 2,
                )
            else:
                value_error = (value_target - value) ** 2

            actor_losses.append(policy_loss)
            value_losses.append(value_error)

        actor_loss = torch.stack(actor_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        if not torch.isfinite(actor_loss) or not torch.isfinite(value_loss):
            raise FloatingPointError(
                "Non-finite policy/value loss detected. Reduce learning rates or "
                "adjust normalization settings."
            )

        actor_total = actor_loss
        value_total = self.args.value_loss_coef * value_loss

        if not torch.isfinite(actor_total) or not torch.isfinite(value_total):
            raise FloatingPointError(
                "Non-finite combined AC loss encountered. Training halted."
            )

        if self.args.use_separate_critic:
            if critic_optimizer is None:
                raise RuntimeError("Critic optimizer missing.")
            agent_optimizer.zero_grad()
            actor_total.backward()
            agent_optimizer.step()

            critic_optimizer.zero_grad()
            value_total.backward()
            critic_optimizer.step()
        else:
            agent_optimizer.zero_grad()
            (actor_total + value_total).backward()
            agent_optimizer.step()

        return {
            "policy_loss": actor_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
        }

    def _update(
        self, agent_idx: int, rollouts: List[RolloutSample]
    ) -> Dict[str, float]:
        if not rollouts:
            return {}

        rewards = torch.stack([sample.reward for sample in rollouts]).float()
        returns_raw = torch.stack([sample.returns for sample in rollouts]).float()
        self._prepare_advantages(rollouts)

        metrics = defaultdict(list)
        metrics["reward_mean"].append(rewards.mean().item())
        if returns_raw.numel() > 0 and torch.isfinite(returns_raw).all():
            metrics["expected_return"].append(returns_raw.mean().item())
        reference_kls = [
            float(sample.metadata.get("reference_kl", 0.0))
            for sample in rollouts
            if "reference_kl" in sample.metadata
        ]
        if reference_kls:
            vals = torch.tensor(reference_kls, dtype=torch.float32)
            if torch.isfinite(vals).all():
                metrics["reference_kl_mean"].append(vals.mean().item())
        reference_penalties = [
            float(sample.metadata.get("reference_kl_penalty", 0.0))
            for sample in rollouts
            if "reference_kl_penalty" in sample.metadata
        ]
        if reference_penalties:
            vals = torch.tensor(reference_penalties, dtype=torch.float32)
            if torch.isfinite(vals).all():
                metrics["reference_kl_penalty_mean"].append(vals.mean().item())

        if self.metrics_callback is not None:
            extra = self.metrics_callback(rollouts)
            if isinstance(extra, dict):
                for key, value in extra.items():
                    metrics[key].append(float(value))
        random.shuffle(rollouts)
        for start in range(0, len(rollouts), self.args.train_batch_size):
            batch = rollouts[start : start + self.args.train_batch_size]
            step_metrics = self._ac_step(agent_idx, batch)
            for key, value in step_metrics.items():
                metrics[key].append(value)

        averaged = {
            key: float(sum(values) / len(values))
            for key, values in metrics.items()
            if values
        }
        return averaged

    def save_model(self, output_dir: str) -> None:
        if not self.dist_env.is_main:
            return
        os.makedirs(output_dir, exist_ok=True)
        if self.args.num_agents == 1:
            actor = unwrap_model(self.agents[0])
            actor.model.save_pretrained(output_dir)
            if actor.value_head is not None:
                torch.save(
                    actor.value_head.state_dict(),
                    os.path.join(output_dir, "value_head.pt"),
                )
            critic = unwrap_model(self.critics[0]) if self.critics else None
            if critic is not None:
                critic_dir = os.path.join(output_dir, "critic")
                os.makedirs(critic_dir, exist_ok=True)
                critic.model.save_pretrained(critic_dir)
                if critic.value_head is not None:
                    torch.save(
                        critic.value_head.state_dict(),
                        os.path.join(critic_dir, "value_head.pt"),
                    )
        else:
            for agent_idx, actor in enumerate(self.agents):
                actor = unwrap_model(actor)
                agent_dir = os.path.join(output_dir, f"agent_{agent_idx}")
                os.makedirs(agent_dir, exist_ok=True)
                actor.model.save_pretrained(agent_dir)
                if actor.value_head is not None:
                    torch.save(
                        actor.value_head.state_dict(),
                        os.path.join(agent_dir, "value_head.pt"),
                    )
                if not self.critics or agent_idx >= len(self.critics):
                    continue
                critic = unwrap_model(self.critics[agent_idx])
                critic_dir = os.path.join(agent_dir, "critic")
                os.makedirs(critic_dir, exist_ok=True)
                critic.model.save_pretrained(critic_dir)
                if critic.value_head is not None:
                    torch.save(
                        critic.value_head.state_dict(),
                        os.path.join(critic_dir, "value_head.pt"),
                    )

        if self.tokenizers:
            if self.args.num_agents == 1:
                self.tokenizers[0].save_pretrained(output_dir)
            else:
                for idx, tok in enumerate(self.tokenizers):
                    agent_dir = os.path.join(output_dir, f"agent_{idx}")
                    os.makedirs(agent_dir, exist_ok=True)
                    tok.save_pretrained(agent_dir)
        elif self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
