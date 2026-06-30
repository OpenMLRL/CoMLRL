import inspect
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
import wandb
from comlrl.models.actor_critic import CausalLMWithValueHead
from comlrl.schedulers import DeviceScheduler
from comlrl.utils.distributed import local_context, unwrap_model
from comlrl.utils.formatters import build_formatters
from comlrl.utils.model_loading import resolve_model_sources
from comlrl.utils.reference_kl import (
    clone_reference_models,
    load_reference_models_from_sources,
    reference_kl_coef,
    reference_kl_enabled,
    reference_kl_for_sequence,
    resolve_reference_devices,
    validate_reference_kl_config,
)
from comlrl.utils.reward_utils import call_reward_function, normalize_reward_lengths
from comlrl.utils.tokenizer_utils import apply_tokenizer_specials, resolve_tokenizers
from .ac_base import ActorCriticTrainerBase
from .iac import RolloutSample

RewardFunc = Callable[..., Sequence[float]]
Formatter = Callable[[Dict[str, Any]], str]
MetricsCallback = Callable[[List["RolloutSample"]], Dict[str, float]]


@dataclass
class MAACConfig:
    """Configuration container for Multi-Agent Actor-Critic with shared critic."""

    agent_learning_rate: float = 5e-6
    critic_learning_rate: float = 5e-6
    rollout_buffer_size: int = 8
    train_batch_size: Optional[int] = None
    value_loss_coef: float = 0.6
    advantage_normalization: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.6
    top_k: Optional[int] = None
    num_train_epochs: int = 40
    num_agents: int = 2
    num_generations: int = 1
    num_turns: int = 2
    parallel_training: str = "none"
    agent_devices: Optional[Union[str, Sequence[str]]] = None
    critic_devices: Optional[Union[str, Sequence[str]]] = None
    external_prompt_passthrough: bool = False
    discount: float = 0.9
    critic_type: str = "v"  # "v" (V(s)) or "q" (Q(s,a))
    early_termination_threshold: Optional[float] = -0.2
    eval_interval: int = 16
    eval_num_samples: int = 4
    eval_batch_size: int = 1
    logging_steps: int = 1
    reference_kl_enabled: bool = False
    reference_kl_coef: float = 0.1
    reference_devices: Optional[Union[str, Sequence[str]]] = None

    def __post_init__(self) -> None:
        if self.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")
        if self.train_batch_size is None:
            self.train_batch_size = self.rollout_buffer_size
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1.")
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1.")
        if self.num_generations < 1:
            raise ValueError("num_generations must be >= 1.")
        if self.num_turns < 1:
            raise ValueError("num_turns must be >= 1.")
        if self.num_turns > 1 and self.num_generations != 1:
            raise ValueError("Multi-turn MAAC currently supports num_generations == 1.")
        critic_type = (self.critic_type or "v").lower()
        if critic_type not in ("v", "q"):
            raise ValueError("critic_type must be one of: 'v', 'q'.")
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
            if self.critic_devices is None:
                raise ValueError(
                    "parallel_training='mp' requires explicit critic_devices."
                )
        self.parallel_training = mode
        validate_reference_kl_config(self, self.num_agents)


class MAACTrainer(ActorCriticTrainerBase):
    """Multi-Agent Actor-Critic with a shared critic conditioned on joint prompts."""

    algorithm_name: str = "MAAC"

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
        args: Optional[MAACConfig] = None,
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
        self.args = args if args is not None else MAACConfig()
        if reward_func is None or not callable(reward_func):
            raise ValueError("reward_func must be a callable.")
        if agent_model is None and agents is None:
            raise ValueError("Either agent_model or agents must be provided.")
        if (
            agents is None
            and self.args.num_agents > 1
            and isinstance(agent_model, PreTrainedModel)
        ):
            raise ValueError(
                "Multi-agent MAAC requires `agent_model` to be a pretrained identifier string."
            )
        if agents is not None and tokenizer is None:
            raise ValueError("Tokenizer must be provided when using agents.")
        if self.args.num_turns > 1 and external_transition is None:
            raise ValueError("Multi-turn MAAC requires an external_transition.")
        if critics is None and critic_model is None:
            raise ValueError(
                "Either critic_model or critics must be provided for MAAC."
            )
        self.reward_func = reward_func
        self.reward_processor = reward_processor or (lambda x: x)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.metrics_callback = metrics_callback
        self.model_config = model_config or {}
        self.parallel_training = (
            str(getattr(self.args, "parallel_training", "none")).strip().lower()
        )
        if self.parallel_training == "null":
            self.parallel_training = "none"
        if self.parallel_training not in {"none", "mp"}:
            raise ValueError("parallel_training only supports: none, mp.")
        if self.parallel_training == "mp":
            self.agent_devices = DeviceScheduler.resolve_devices(
                getattr(self.args, "agent_devices", None),
                self.args.num_agents,
                kind="agent_devices",
            )
            self.critic_device = DeviceScheduler.assign_shared_critic_device(
                self.agent_devices, getattr(self.args, "critic_devices", None)
            )
        else:
            single_device = DeviceScheduler.resolve_single_device(
                getattr(self.args, "agent_devices", None),
                getattr(self.args, "critic_devices", None),
            )
            self.agent_devices = [single_device] * self.args.num_agents
            self.critic_device = single_device
        self.device = self.agent_devices[0]
        self.dist_env = local_context(self.device)
        self._parallel_update_enabled = False

        tokenizers = resolve_tokenizers(agent_model, tokenizer, agents)
        if isinstance(tokenizers, list):
            self.tokenizers = tokenizers
            self.tokenizer = tokenizers[0] if tokenizers else None
        else:
            self.tokenizers = [tokenizers] * self.args.num_agents
            self.tokenizer = tokenizers
        self.external_transition = external_transition

        self.agents: List[CausalLMWithValueHead] = []
        actor_sources, _agent_name = resolve_model_sources(
            kind="agents",
            model=agent_model,
            models=agents,
            expected_count=self.args.num_agents,
            model_label="agent_model",
        )
        actor_model_kwargs = self._filter_model_kwargs(
            self.model_config.get("model_kwargs", {})
        )
        for idx, actor_source in enumerate(actor_sources):
            if actor_source is None:
                raise ValueError("agent_model must be provided for MAAC.")
            if isinstance(actor_source, CausalLMWithValueHead):
                agent_model = actor_source
            elif isinstance(actor_source, PreTrainedModel):
                base = actor_source
                agent_model = CausalLMWithValueHead(
                    base_model=base,
                    attach_value_head=False,
                    value_head_hidden_dim=None,
                )
            else:
                try:
                    base = AutoModelForCausalLM.from_pretrained(
                        actor_source, **actor_model_kwargs
                    )
                except (OSError, ValueError) as exc:
                    raise ValueError(
                        f"Failed to load actor model from identifier '{actor_source}'."
                    ) from exc
                agent_model = CausalLMWithValueHead(
                    base_model=base,
                    attach_value_head=False,
                    value_head_hidden_dim=None,
                )
            agent_model.to(self.agent_devices[idx])
            self.agents.append(agent_model)

        critic_sources, _critic_name = resolve_model_sources(
            kind="critics",
            model=critic_model,
            models=critics,
            expected_count=1,
            expected_label="1 critic",
            model_label="critic_model",
        )
        critic_source = critic_sources[0]
        if isinstance(critic_source, CausalLMWithValueHead):
            critic_model_instance = critic_source
        elif isinstance(critic_source, PreTrainedModel):
            base = critic_source
            critic_model_instance = CausalLMWithValueHead(
                base_model=base,
                attach_value_head=True,
                value_head_hidden_dim=self.model_config.get(
                    "critic_value_head_hidden_dim"
                ),
            )
        else:
            model_kwargs = self._filter_model_kwargs(
                self.model_config.get("critic_model_kwargs", {})
            )
            try:
                base = AutoModelForCausalLM.from_pretrained(
                    critic_source, **model_kwargs
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Failed to load critic model from identifier '{critic_source}'."
                ) from exc
            critic_model_instance = CausalLMWithValueHead(
                base_model=base,
                attach_value_head=True,
                value_head_hidden_dim=self.model_config.get(
                    "critic_value_head_hidden_dim"
                ),
            )
        critic_model_instance.to(self.critic_device)
        self.critics: List[CausalLMWithValueHead] = [critic_model_instance]

        self.reference_models: List[Any] = []
        self.reference_devices: List[torch.device] = []
        if reference_kl_enabled(self.args):
            self.reference_devices = resolve_reference_devices(
                self.args,
                self.agent_devices,
                self.args.num_agents,
            )
            if actor_sources and all(isinstance(src, str) for src in actor_sources):
                self.reference_models = load_reference_models_from_sources(
                    actor_sources,
                    devices=self.reference_devices,
                    model_kwargs=actor_model_kwargs,
                )
            else:
                self.reference_models = clone_reference_models(
                    self.agents,
                    devices=self.reference_devices,
                )

        if self.tokenizers and len(self.tokenizers) == len(self.agents):
            for idx, tok in enumerate(self.tokenizers):
                models = [self.agents[idx]]
                if idx < len(self.reference_models):
                    models.append(self.reference_models[idx])
                apply_tokenizer_specials(tok, models)
            apply_tokenizer_specials(self.tokenizers[0], [self.critics[0]])
        else:
            apply_tokenizer_specials(
                self.tokenizer, [*self.agents, self.critics[0], *self.reference_models]
            )

        self.formatters = build_formatters(formatters, self.args.num_agents)
        try:
            self._reward_signature = inspect.signature(reward_func)
        except (TypeError, ValueError):
            self._reward_signature = inspect.Signature()

        self.agent_optimizers = []
        for agent_model in self.agents:
            optimizer = torch.optim.AdamW(
                agent_model.parameters(),
                lr=self.args.agent_learning_rate,
            )
            self.agent_optimizers.append(optimizer)

        self.critic_optimizer = torch.optim.AdamW(
            self.critics[0].parameters(),
            lr=self.args.critic_learning_rate,
        )

        self.rollout_buffers = [[] for _ in range(self.args.num_agents)]
        self.wandb_config = wandb_config
        self.wandb_initialized = False
        self.env_step = 0
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

    def _init_wandb(self) -> None:
        if not self.dist_env.is_main:
            return
        if self.wandb_config is None:
            self.wandb_config = {}
        wandb_project = self.wandb_config.get("project", "comlrl")
        wandb_entity = self.wandb_config.get("entity")
        algo_tag = str(self.algorithm_name or "maac").lower()
        wandb_run_name = (
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
            "max_new_tokens": self.args.max_new_tokens,
            "num_generations": self.args.num_generations,
            "critic_type": self.args.critic_type,
            "reference_kl_enabled": reference_kl_enabled(self.args),
            "reference_kl_coef": reference_kl_coef(self.args),
            "reference_devices": getattr(self.args, "reference_devices", None),
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

        init_kwargs = {
            "project": wandb_project,
            "name": wandb_run_name,
            "entity": wandb_entity,
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

    def _build_critic_input(
        self, prompts: Sequence[str], action_completions: Optional[Sequence[str]] = None
    ) -> str:
        """Build centralized critic conditioning input.

        - critic_type='v': V(s) conditioned on joint prompt only.
        - critic_type='q': Q(s,a) conditioned on joint prompt + joint action text.
        """
        base = "\n\n".join([f"[Agent {idx}] {p}" for idx, p in enumerate(prompts)])
        if (self.args.critic_type or "v").lower() == "v":
            return base

        action_completions = list(action_completions or [])
        action_lines: List[str] = ["[Joint Action]"]
        for idx, comp in enumerate(action_completions):
            action_lines.append(f"[Agent {idx} action]\n{comp}")
        return base + "\n\n" + "\n\n".join(action_lines)

    def _critic_value_from_text(self, critic_input: str) -> Dict[str, Any]:
        encoded = self._encode_prompt(
            critic_input,
            tokenizer=self._get_tokenizer(0),
            device=self.critic_device,
        )
        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        prompt_len = ids.size(1)
        value = self._value_on_prompt_only(self.critics[0], ids, mask, prompt_len)
        return {
            "critic_input": critic_input,
            "input_ids": ids,
            "attention_mask": mask,
            "prompt_len": prompt_len,
            "value": value,
        }

    def _generate(self, agent_model, prompt: str, agent_idx: int) -> Dict[str, Any]:
        agent_device = self._agent_device(agent_idx)
        encoded_prompt = self._encode_prompt(
            prompt, agent_idx=agent_idx, device=agent_device
        )
        prompt_input_ids = encoded_prompt["input_ids"]
        prompt_attention_mask = encoded_prompt["attention_mask"]
        prompt_len = prompt_input_ids.size(1)

        num_ret = (
            1
            if bool(getattr(self, "_in_eval", False))
            else int(self.args.num_generations)
        )
        generation_kwargs: Dict[str, Any] = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": True,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "num_return_sequences": num_ret,
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
            "reference_kls": reference_kls,
            "completions": completion_texts,
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

    def _collect_rollouts(self, item: Dict[str, Any]) -> List[RolloutSample]:
        num_turns = max(1, int(getattr(self.args, "num_turns", 1)))
        if num_turns > 1:
            return self._collect_rollouts_multi_turn(item, num_turns)

        num_ret = int(self.args.num_generations)
        turn_prompts = [
            self._resolve_turn_prompt(item, agent_idx)
            for agent_idx in range(self.args.num_agents)
        ]

        def _generate_agent(agent_idx: int) -> Dict[str, Any]:
            agent_model = self.agents[agent_idx]
            prompt = turn_prompts[agent_idx]
            gen = self._generate(agent_model, prompt, agent_idx)
            return {
                "agent_idx": agent_idx,
                "prompt": prompt,
                "prompt_len": gen["prompt_len"],
                "sequences": gen["sequences"],
                "attention_mask": gen["attention_mask"],
                "response_lens": gen["response_lens"],
                "reference_kls": gen["reference_kls"],
                "completion_texts": gen["completions"],
            }

        rollout_data = self._run_agent_tasks(_generate_agent)
        prompts: List[str] = [entry["prompt"] for entry in rollout_data]
        completions_per_agent: List[List[str]] = [
            entry["completion_texts"] for entry in rollout_data
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
            algorithm="MAAC",
        )
        num_agents = self.args.num_agents
        if len(rewards) == 1:
            rewards_matrix = [[rewards[0]] * num_ret for _ in range(num_agents)]
        elif len(rewards) == num_ret:
            rewards_matrix = [list(rewards) for _ in range(num_agents)]
        elif len(rewards) == num_agents:
            rewards_matrix = [[rewards[a]] * num_ret for a in range(num_agents)]
        else:
            raise ValueError(
                "Reward function must return 1 value, num_generations values, "
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
                shaped_reward, kl_meta = self._kl_shaped_reward(reward, data, i)

                logprob, _ = self._policy_eval(
                    self.agents[agent_idx],
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
                reward_cpu = torch.tensor([reward], dtype=torch.float32)
                shaped_reward_cpu = torch.tensor([shaped_reward], dtype=torch.float32)
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
                        old_value=value,
                        reward=reward_cpu,
                        returns=shaped_reward_cpu.clone(),
                        advantage=torch.zeros_like(reward_cpu),
                        normalized_advantage=None,
                        metadata={
                            "joint_input_ids": joint_ids.detach().cpu(),
                            "joint_attention_mask": joint_mask.detach().cpu(),
                            "joint_prompt_len": joint_len,
                            "turn_idx": 0,
                            "generation_idx": i,
                            "batch_item": item,
                            "adv_target": shaped_reward_cpu,
                            **kl_meta,
                        },
                    )
                )

        for sample in rollouts:
            r = float(sample.returns.view(-1)[0].item())
            sample.metadata["value_target"] = torch.tensor([r]).cpu()

        return rollouts

    def _collect_rollouts_multi_turn(
        self, item: Dict[str, Any], num_turns: int
    ) -> List[RolloutSample]:
        if self.args.num_generations != 1:
            raise ValueError("Multi-turn MAAC currently supports num_generations == 1.")

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
                gen = self._generate(agent_model, prompt, agent_idx)
                return {
                    "agent_idx": agent_idx,
                    "prompt": prompt,
                    "prompt_len": gen["prompt_len"],
                    "sequences": gen["sequences"],
                    "attention_mask": gen["attention_mask"],
                    "response_lens": gen["response_lens"],
                    "reference_kls": gen["reference_kls"],
                    "completion_texts": gen["completions"],
                }

            rollout_data = self._run_agent_tasks(_generate_agent_turn)
            completions_per_agent = [
                entry["completion_texts"] for entry in rollout_data
            ]

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
                algorithm="MAAC",
            )
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
                reward_cpu = torch.tensor([reward_val], dtype=torch.float32)
                shaped_reward, kl_meta = self._kl_shaped_reward(reward_val, data, 0)
                shaped_reward_cpu = torch.tensor([shaped_reward], dtype=torch.float32)

                logprob, _ = self._policy_eval(
                    self.agents[agent_idx],
                    seq.unsqueeze(0),
                    attn.unsqueeze(0),
                    data["prompt_len"],
                    resp_len,
                    output_values=False,
                )

                value = joint_value.detach().cpu()
                logprob_cpu = logprob.detach().cpu()
                completion_text = data["completion_texts"][0]
                sample = RolloutSample(
                    agent_idx=agent_idx,
                    prompt=data["prompt"],
                    completion=completion_text,
                    full_input_ids=seq.detach().cpu(),
                    attention_mask=attn.detach().cpu(),
                    prompt_len=data["prompt_len"],
                    response_len=resp_len,
                    old_logprob=logprob_cpu,
                    old_value=value,
                    reward=reward_cpu,
                    returns=shaped_reward_cpu.clone(),
                    advantage=torch.zeros_like(reward_cpu),
                    normalized_advantage=None,
                    metadata={
                        "joint_input_ids": joint_ids.detach().cpu(),
                        "joint_attention_mask": joint_mask.detach().cpu(),
                        "joint_prompt_len": joint_len,
                        "turn_idx": turn_idx,
                        "generation_idx": 0,
                        "batch_item": item,
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
            "Reward function must return 1 value, num_generations values, or num_agents values."
        )

    # Advantage prep
    # Losses
    def _policy_eval(
        self,
        agent_model: CausalLMWithValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_len: int,
        output_values: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
            value = self._value_on_prompt_only(
                agent_model, sequences, attention_mask, prompt_len
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
        agent_model = self.agents[agent_idx]
        agent_optimizer = self.agent_optimizers[agent_idx]

        actor_losses: List[torch.Tensor] = []
        value_losses: List[torch.Tensor] = []
        agent_device = self._agent_device(agent_idx)
        critic_device = self.critic_device

        for sample in batch:
            sequences = sample.full_input_ids.to(agent_device).unsqueeze(0)
            attention_mask = sample.attention_mask.to(agent_device).unsqueeze(0)

            logprob, _ = self._policy_eval(
                agent_model,
                sequences,
                attention_mask,
                sample.prompt_len,
                sample.response_len,
                output_values=False,
            )

            joint_ids = sample.metadata["joint_input_ids"].to(critic_device)
            joint_mask = sample.metadata["joint_attention_mask"].to(critic_device)
            joint_len = int(sample.metadata["joint_prompt_len"])
            value = self._value_on_prompt_only(
                self.critics[0], joint_ids, joint_mask, joint_len
            )

            value_device = value.device
            old_value = sample.old_value.to(value_device, dtype=value.dtype)
            policy_advantage = sample.normalized_advantage.to(
                agent_device, dtype=logprob.dtype
            )
            value_target = sample.metadata.get("value_target")
            if value_target is None:
                raise RuntimeError("value_target missing for critic update.")
            returns = value_target.to(value_device, dtype=value.dtype)

            if not torch.isfinite(logprob).all():
                raise FloatingPointError(
                    "Encountered non-finite logprob during AC step."
                )
            if not torch.isfinite(policy_advantage).all():
                raise FloatingPointError("Advantage contains non-finite values.")
            if not torch.isfinite(returns).all():
                raise FloatingPointError("Returns contain non-finite values.")

            policy_loss = -(logprob * policy_advantage)
            value_error = (returns - value) ** 2

            actor_losses.append(policy_loss)
            value_losses.append(value_error)

        actor_loss = torch.stack(actor_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        value_total = self.args.value_loss_coef * value_loss
        if not torch.isfinite(actor_loss) or not torch.isfinite(value_loss):
            raise FloatingPointError("Non-finite loss detected.")

        agent_optimizer.zero_grad()
        actor_loss.backward()
        agent_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_total.backward()
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
        for start in range(0, len(rollouts), self.args.train_batch_size):
            batch = rollouts[start : start + self.args.train_batch_size]
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

    def _on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        epoch_metrics: Dict[str, List[float]],
    ) -> None:
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

        summary = self._summarize_epoch_metrics(epoch_metrics)
        if summary and getattr(self, "verbose", True) and self.dist_env.is_main:
            to_print = epoch_log if epoch_log else summary
            print(f"Epoch {epoch + 1}/{total_epochs} metrics: {to_print}")

    def save_model(self, output_dir: str) -> None:
        if not self.dist_env.is_main:
            return
        os.makedirs(output_dir, exist_ok=True)
        for agent_idx, actor in enumerate(self.agents):
            actor = unwrap_model(actor)
            agent_dir = os.path.join(output_dir, f"agent_{agent_idx}")
            os.makedirs(agent_dir, exist_ok=True)
            actor.model.save_pretrained(agent_dir)
        critic_dir = os.path.join(output_dir, "critic")
        os.makedirs(critic_dir, exist_ok=True)
        critic = unwrap_model(self.critics[0])
        critic.model.save_pretrained(critic_dir)
        if critic.value_head is not None:
            torch.save(
                critic.value_head.state_dict(),
                os.path.join(critic_dir, "value_head.pt"),
            )
        if self.tokenizers:
            for idx, tok in enumerate(self.tokenizers):
                agent_dir = os.path.join(output_dir, f"agent_{idx}")
                os.makedirs(agent_dir, exist_ok=True)
                tok.save_pretrained(agent_dir)
        elif self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
