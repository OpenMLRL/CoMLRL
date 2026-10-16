"""Tiny offline models, trainers and assertions shared by the test modules."""

from dataclasses import fields, is_dataclass
import random
from types import SimpleNamespace

from datasets import Dataset
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from comlrl.trainers.preference import (
    MADPOIterConfig,
    MARLHFConfig,
    MARLHFIterConfig,
    TaggedCentralizedComparatorAdapter,
)
from comlrl.trainers.reinforce import CentralizedMAGRPOConfig, CentralizedMAGRPOTrainer
from comlrl.utils.reward_utils import set_reward_range

GOOD = "<agent_0> good </agent_0> <agent_1> good </agent_1>"
BAD = "<agent_0> bad </agent_0> <agent_1> bad </agent_1>"
WORDS = [
    "<pad>",
    "<eos>",
    "<unk>",
    "joint",
    "left",
    "right",
    "good",
    "bad",
    "<agent_0>",
    "</agent_0>",
    "<agent_1>",
    "</agent_1>",
]


def word_tokenizer():
    backend = Tokenizer(
        WordLevel({token: i for i, token in enumerate(WORDS)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def model_for(tokenizer):
    """A dropout-free GPT-2 whose vocabulary matches ``tokenizer``."""
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=256,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )


def tiny_gpt2(vocab_size=32, n_embd=16):
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=vocab_size,
            n_positions=32,
            n_embd=n_embd,
            n_layer=1,
            n_head=1,
        )
    )


def stub_tokenizer():
    return SimpleNamespace(
        pad_token="<pad>", eos_token="</s>", pad_token_id=0, eos_token_id=1
    )


def zero_reward(*_args, **_kwargs):
    return [0.0]


class JointAdapter(TaggedCentralizedComparatorAdapter):
    def build_prompt(self, item, agent_prompts):
        assert agent_prompts == ["left", "right"]
        return "joint left right"


def task_reward(left, right, *, batch_items, prompts):
    assert prompts == ["left"]
    assert batch_items[0]["id"] == "task"
    return [float(a == "good") + float(b == "good") for a, b in zip(left, right)]


set_reward_range(task_reward, 0, 2)


def make_preference_trainer(cls, config_cls, tokenizer, tmp_path, **overrides):
    """A centralized preference trainer whose actor alternates GOOD/BAD samples."""
    config = dict(
        collaboration_mode="centralized",
        agent_devices="cpu",
        num_agents=2,
        num_train_epochs=1,
        preference_num_candidates=2,
        preference_pairs_per_sample=1,
        eval_interval=0,
        rollout_buffer_size=1,
        train_batch_size=1,
        max_new_tokens=8,
        agent_learning_rate=1e-3,
    )
    if issubclass(config_cls, MARLHFConfig):
        config.update(
            reward_model_device="cpu", reward_num_train_epochs=1, num_generations=2
        )
    if issubclass(config_cls, (MADPOIterConfig, MARLHFIterConfig)):
        config.update(
            num_iterations=2,
            preference_replay_dir=str(tmp_path / "replay"),
            policy_checkpoint_dir=str(tmp_path / "checkpoints"),
            log_reward_distribution=False,
        )
    config.update(overrides)
    dataset = Dataset.from_list([{"id": "task", "prompt": "task prompt"}])
    model = model_for(tokenizer)
    calls = []

    def generate(*, input_ids, num_return_sequences, **kwargs):
        calls.append(input_ids.clone())
        sequences = []
        for idx in range(num_return_sequences):
            # Target and comparator calls get different pairs.
            text = GOOD if (idx + len(calls)) % 2 else BAD
            ids = torch.tensor(tokenizer.encode(text), device=input_ids.device)
            sequences.append(torch.cat([input_ids[0], ids]))
        return SimpleNamespace(sequences=torch.stack(sequences))

    model.generate = generate
    trainer = cls(
        agents=[model],
        num_agents=2,
        tokenizer=tokenizer,
        args=config_cls(**config),
        train_dataset=dataset,
        eval_dataset=dataset,
        reward_func=task_reward,
        formatters=[lambda item: "left", lambda item: "right"],
        centralized_comparator_adapter=JointAdapter(),
    )
    trainer.verbose = False
    return trainer, calls


def make_centralized_magrpo_trainer(tokenizer, **overrides):
    """A centralized MAGRPO trainer whose actor alternates GOOD/BAD samples."""
    options = dict(
        agent_devices="cpu",
        num_agents=2,
        num_train_epochs=1,
        num_generations=2,
        eval_interval=0,
        rollout_buffer_size=1,
        train_batch_size=1,
        max_new_tokens=8,
        agent_learning_rate=1e-3,
    )
    options.update(overrides)
    model = model_for(tokenizer)
    prompts = []

    def generate(*, input_ids, num_return_sequences, **kwargs):
        prompts.append(tokenizer.decode(input_ids[0]))
        sequences = [
            torch.cat(
                [
                    input_ids[0],
                    torch.tensor(
                        tokenizer.encode(GOOD if index % 2 == 0 else BAD),
                        device=input_ids.device,
                    ),
                ]
            )
            for index in range(num_return_sequences)
        ]
        return SimpleNamespace(sequences=torch.stack(sequences))

    model.generate = generate
    dataset = Dataset.from_list([{"id": "task", "prompt": "task prompt"}])
    trainer = CentralizedMAGRPOTrainer(
        agents=[model],
        num_agents=2,
        tokenizer=tokenizer,
        args=CentralizedMAGRPOConfig(**options),
        train_dataset=dataset,
        eval_dataset=dataset,
        reward_func=task_reward,
        formatters=[lambda _: "left", lambda _: "right"],
        centralized_adapter=JointAdapter(),
    )
    trainer.verbose = False
    return trainer, prompts


def seed_all(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def rng_state():
    """Python, NumPy and torch generator states, plus CUDA once initialized."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def assert_nested_equal(actual, expected):
    """Exact equality through tensors, arrays, dataclasses and containers."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            assert_nested_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for value, target in zip(actual, expected):
            assert_nested_equal(value, target)
    else:
        assert actual == expected
