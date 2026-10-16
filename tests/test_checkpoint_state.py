"""Checkpoint state round trips use CPU models without downloads or training jobs."""

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset

from comlrl.trainers.reinforce.magrpo import MAGRPOConfig, NodeSample
from comlrl.utils.checkpoint import (
    TrainingCheckpointMixin,
    _decode,
    _encode,
    capture_trainer_state,
    restore_trainer_state,
)
from comlrl.utils.formatters import build_formatters
from test_centralized_collaboration import model_for, tokenizer


class StatefulReward:
    def __init__(self):
        self.calls = 3

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return [1.0]

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


class StatefulFormatter:
    def __init__(self):
        self.calls = 0

    def __call__(self, item, external_prompts=None):
        self.calls += 1
        prompt = item["prompt"] if external_prompts is None else external_prompts
        return f"{prompt}:{self.calls}"

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


class StatefulEvaluationDataset:
    def __init__(self):
        self.position = 0
        self.items = ["first", "second"]

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == len(self.items):
            raise StopIteration
        result = self.items[self.position]
        self.position += 1
        return result

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


class TinyTrainer(TrainingCheckpointMixin):
    def __init__(self, *, width=3, prompts=("first", "second"), **config):
        self.args = MAGRPOConfig(
            num_agents=2,
            num_turns=1,
            agent_devices="cpu",
            reference_kl_enabled=True,
            **config,
        )
        self.num_agents = 2
        self.agent_devices = [torch.device("cpu")] * 2
        self.critic_devices = [torch.device("cpu")] * 2
        self.parallel_training = "none"
        self.agents = [torch.nn.Linear(width, 2) for _ in range(2)]
        self.critics = [torch.nn.Linear(width, 1) for _ in range(2)]
        self.reference_models = copy.deepcopy(self.agents)
        for model in self.reference_models:
            model.requires_grad_(False)
            model.eval()
        self.reward_model = torch.nn.Linear(width, 1)
        self.optimizers = [
            torch.optim.AdamW(model.parameters(), lr=0.01) for model in self.agents
        ]
        self.critic_optimizers = [
            torch.optim.AdamW(model.parameters(), lr=0.02) for model in self.critics
        ]
        self.reward_optimizer = torch.optim.AdamW(
            self.reward_model.parameters(), lr=0.03
        )
        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
            for optimizer in self.optimizers
        ]
        self.tokenizers = []
        self.train_dataset = Dataset.from_dict({"prompt": list(prompts)})
        self.eval_dataset = self.train_dataset
        self.env_step = 17
        self._last_train_log_step = 11
        self._comparator_rng_states = {"cpu": torch.get_rng_state().clone()}
        self.target_seed = 71
        self.comparator_seed = -71
        self.rollout_buffers = [
            [
                NodeSample(
                    agent_idx=0,
                    turn_idx=0,
                    completions_data={"completion_input_ids": torch.tensor([1, 2, 3])},
                    returns=[1.25, -0.5],
                    node_mean_reward=0.5,
                    node_mean_return=0.375,
                    node_env_step=16,
                )
            ],
            [],
        ]
        self.reward_func = StatefulReward()
        self.dist_env = SimpleNamespace(is_main=True)
        self._checkpoint_progress = {"epoch": 1, "next_item": 0}


class DifferentTrainer(TinyTrainer):
    pass


class UnexpectedPayload:
    pass


@pytest.fixture(autouse=True)
def preserve_test_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def assert_nested_equal(actual, expected):
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


def advance_models(trainer):
    inputs = torch.tensor([[0.2, -0.4, 0.7], [0.6, 0.1, -0.2]])
    models = trainer.agents + trainer.critics + [trainer.reward_model]
    optimizers = (
        trainer.optimizers + trainer.critic_optimizers + [trainer.reward_optimizer]
    )
    for model, optimizer in zip(models, optimizers):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
    for scheduler in trainer.schedulers:
        scheduler.step()


def assert_models_equal(actual, expected):
    for name in ("agents", "critics", "reference_models"):
        for left, right in zip(getattr(actual, name), getattr(expected, name)):
            assert_nested_equal(left.state_dict(), right.state_dict())
    assert_nested_equal(
        actual.reward_model.state_dict(), expected.reward_model.state_dict()
    )


def test_cpu_roundtrip_restores_models_optimizer_moments_and_next_step(tmp_path):
    source = TinyTrainer()
    advance_models(source)
    state = capture_trainer_state(source)
    destination = TinyTrainer()

    restore_trainer_state(destination, state, tmp_path)

    assert_models_equal(destination, source)
    for name in ("optimizers", "critic_optimizers", "schedulers"):
        for left, right in zip(getattr(destination, name), getattr(source, name)):
            assert_nested_equal(left.state_dict(), right.state_dict())
    assert_nested_equal(
        destination.reward_optimizer.state_dict(), source.reward_optimizer.state_dict()
    )
    advance_models(source)
    advance_models(destination)
    assert_models_equal(destination, source)


def test_snapshot_is_detached_and_preserves_allowlisted_rollout_dataclass(tmp_path):
    trainer = TinyTrainer()
    state = capture_trainer_state(trainer)
    expected = copy.deepcopy(trainer.agents[0].state_dict())
    with torch.no_grad():
        trainer.agents[0].weight.add_(10)
    trainer.rollout_buffers[0][0].completions_data["completion_input_ids"].fill_(99)

    restored = TinyTrainer()
    restore_trainer_state(restored, state, tmp_path)

    assert_nested_equal(restored.agents[0].state_dict(), expected)
    sample = restored.rollout_buffers[0][0]
    assert isinstance(sample, NodeSample)
    assert torch.equal(
        sample.completions_data["completion_input_ids"], torch.tensor([1, 2, 3])
    )
    assert sample.returns == [1.25, -0.5]
    assert restored.env_step == 17
    assert restored._last_train_log_step == 11
    assert_nested_equal(restored._comparator_rng_states, trainer._comparator_rng_states)


def test_restored_training_does_not_mutate_reusable_in_memory_snapshot(tmp_path):
    source = TinyTrainer()
    advance_models(source)
    state = capture_trainer_state(source)
    before = copy.deepcopy(state)
    target = TinyTrainer()
    restore_trainer_state(target, state, tmp_path)

    advance_models(target)
    target.rollout_buffers[0].clear()
    target._comparator_rng_states["cpu"].zero_()

    assert_nested_equal(state, before)


def test_state_is_compatible_with_weights_only_loading(tmp_path):
    trainer = TinyTrainer()
    advance_models(trainer)
    path = tmp_path / "state.pt"
    state = capture_trainer_state(trainer)
    torch.save(_encode(state), path)

    loaded = _decode(torch.load(path, weights_only=True, map_location="cpu"))

    restored = TinyTrainer()
    restore_trainer_state(restored, loaded, tmp_path)
    assert_models_equal(restored, trainer)


def test_capture_preserves_rng_and_restore_replays_all_cpu_generators(tmp_path):
    trainer = TinyTrainer()
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    state = capture_trainer_state(trainer)
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert_nested_equal(after, before)
    expected = (random.random(), np.random.rand(4), torch.rand(4))
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)

    restore_trainer_state(trainer, state, tmp_path)

    actual = (random.random(), np.random.rand(4), torch.rand(4))
    assert_nested_equal(actual, expected)


def test_restore_rng_false_preserves_current_generators(tmp_path):
    trainer = TinyTrainer()
    state = capture_trainer_state(trainer)
    random.seed(13)
    np.random.seed(14)
    torch.manual_seed(15)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    restore_trainer_state(trainer, state, tmp_path, restore_rng=False)

    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert_nested_equal(after, before)


def test_stateful_callback_restores_its_state(tmp_path):
    source = TinyTrainer()
    source.reward_func.calls = 19
    state = capture_trainer_state(source)
    target = TinyTrainer()
    original_callback = target.reward_func

    restore_trainer_state(target, state, tmp_path)

    assert target.reward_func is original_callback
    assert target.reward_func.calls == 19


@pytest.mark.parametrize(
    "make_target",
    [
        lambda: DifferentTrainer(),
        lambda: TinyTrainer(width=4),
        lambda: TinyTrainer(discount=0.7),
        lambda: TinyTrainer(prompts=("changed", "second")),
        lambda: TinyTrainer(prompts=("first",)),
    ],
    ids=[
        "trainer-class",
        "model-shape",
        "training-config",
        "dataset-content",
        "dataset-length",
    ],
)
def test_incompatible_checkpoint_does_not_partially_mutate_trainer(
    tmp_path, make_target
):
    state = capture_trainer_state(TinyTrainer())
    target = make_target()
    before = capture_trainer_state(target)

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        restore_trainer_state(target, state, tmp_path)

    assert_nested_equal(capture_trainer_state(target), before)


def test_run_length_and_logging_config_can_change(tmp_path):
    source = TinyTrainer(num_train_epochs=2, logging_steps=1)
    state = capture_trainer_state(source)
    target = TinyTrainer(num_train_epochs=5, logging_steps=4)

    restore_trainer_state(target, state, tmp_path)

    assert target.args.num_train_epochs == 5
    assert target.args.logging_steps == 4
    assert_models_equal(target, source)


def test_late_model_incompatibility_is_checked_before_loading_any_model(tmp_path):
    source = TinyTrainer()
    advance_models(source)
    state = capture_trainer_state(source)
    target = TinyTrainer()
    target.reward_model = torch.nn.Linear(4, 1)
    target.reward_optimizer = torch.optim.AdamW(target.reward_model.parameters())
    before = capture_trainer_state(target)

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        restore_trainer_state(target, state, tmp_path)

    assert_nested_equal(capture_trainer_state(target), before)


def test_invalid_lazy_reward_model_leaves_uninitialized_trainer_unchanged(tmp_path):
    source = TinyTrainer()
    state = capture_trainer_state(source)
    target = TinyTrainer()
    target.reward_model = None
    target.reward_optimizer = None

    def init_reward_model():
        target.reward_model = torch.nn.Linear(4, 1)
        target.reward_optimizer = torch.optim.AdamW(target.reward_model.parameters())

    target._init_reward_model = init_reward_model
    before = capture_trainer_state(target)

    with pytest.raises(ValueError, match="shape|dtype"):
        restore_trainer_state(target, state, tmp_path)

    assert target.reward_model is None
    assert target.reward_optimizer is None
    assert_nested_equal(capture_trainer_state(target), before)


def test_public_save_preserves_rng_and_load_restores_state(tmp_path):
    source = TinyTrainer()
    advance_models(source)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    source.save_checkpoint(tmp_path / "checkpoint")

    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert_nested_equal(after, before)
    expected_random = (random.random(), np.random.rand(4), torch.rand(4))
    target = TinyTrainer()
    target.load_checkpoint(tmp_path / "checkpoint")
    assert_models_equal(target, source)
    assert target._checkpoint_progress == source._checkpoint_progress
    actual_random = (random.random(), np.random.rand(4), torch.rand(4))
    assert_nested_equal(actual_random, expected_random)


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_failed_checkpoint_write_keeps_existing_checkpoint_loadable(
    tmp_path, monkeypatch, failure
):
    source = TinyTrainer()
    advance_models(source)
    checkpoint = tmp_path / "checkpoint"
    source.save_checkpoint(checkpoint)
    saved_files = {
        path.relative_to(checkpoint): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    expected = TinyTrainer()
    expected.load_checkpoint(checkpoint)
    advance_models(source)

    def fail_save(*args, **kwargs):
        raise OSError("simulated storage failure")

    original_replace = os.replace
    publish_failed = False

    def fail_publish(source_path, target_path):
        nonlocal publish_failed
        if Path(target_path) == checkpoint and not publish_failed:
            publish_failed = True
            raise OSError("simulated storage failure")
        return original_replace(source_path, target_path)

    with monkeypatch.context() as patch:
        if failure == "write":
            patch.setattr(torch, "save", fail_save)
        else:
            patch.setattr(os, "replace", fail_publish)
        with pytest.raises(OSError, match="simulated storage failure"):
            source.save_checkpoint(checkpoint)

    assert {
        path.relative_to(checkpoint): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    } == saved_files
    restored = TinyTrainer()
    restored.load_checkpoint(checkpoint)
    assert_models_equal(restored, expected)


def corrupt_state(state, corruption):
    if corruption == "version":
        state["version"] = 999
    elif corruption == "missing-section":
        del state["models"]
    elif corruption == "unknown-runtime":
        state["runtime"]["unsupported_field"] = 9
    elif corruption == "invalid-python-rng":
        state["rng"]["python"] = None
    elif corruption == "invalid-numpy-rng":
        state["rng"]["numpy"] = ("not-a-generator", np.arange(3), 0, 0, 0.0)
    elif corruption == "invalid-torch-rng":
        state["rng"]["torch"] = torch.zeros(1, dtype=torch.uint8)
    elif corruption == "truncated-model-classes":
        state["models"]["agents"]["classes"] = []
    elif corruption == "missing-optimizer-state":
        del state["optimizers"]["optimizers"]["states"][0]["state"]
    elif corruption == "invalid-optimizer-moment":
        optimizer_state = state["optimizers"]["optimizers"]["states"][0]["state"]
        first = next(iter(optimizer_state.values()))
        first["exp_avg"] = torch.zeros(1)
    elif corruption == "missing-callback-state":
        del state["callbacks"]["reward_func"]["state"]
    elif corruption == "boolean-sequence":
        state["sequence"] = True
    else:
        raise AssertionError(corruption)


@pytest.mark.parametrize(
    "corruption",
    [
        "version",
        "missing-section",
        "unknown-runtime",
        "invalid-python-rng",
        "invalid-numpy-rng",
        "invalid-torch-rng",
        "truncated-model-classes",
        "missing-optimizer-state",
        "invalid-optimizer-moment",
        "missing-callback-state",
        "boolean-sequence",
    ],
)
def test_corrupt_state_is_rejected_without_partial_mutation(tmp_path, corruption):
    source = TinyTrainer()
    advance_models(source)
    state = capture_trainer_state(source)
    corrupt_state(state, corruption)
    target = TinyTrainer()
    before = capture_trainer_state(target)

    with pytest.raises((ValueError, RuntimeError, TypeError, KeyError)):
        restore_trainer_state(target, state, tmp_path)

    assert_nested_equal(capture_trainer_state(target), before)


def test_public_load_rejects_unallowlisted_pickle_without_mutation(tmp_path):
    target = TinyTrainer()
    before = capture_trainer_state(target)
    path = tmp_path / "unsafe.pt"
    torch.save(UnexpectedPayload(), path)

    with pytest.raises(pickle.UnpicklingError):
        target.load_checkpoint(path)

    assert_nested_equal(capture_trainer_state(target), before)


def test_public_load_rejects_unallowlisted_record_without_mutation(tmp_path):
    target = TinyTrainer()
    before = capture_trainer_state(target)
    path = tmp_path / "unknown-record.pt"
    torch.save(
        {
            "__kind__": "record",
            "class": "test_checkpoint_state.UnexpectedPayload",
            "fields": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="Unsupported checkpoint record"):
        target.load_checkpoint(path)

    assert_nested_equal(capture_trainer_state(target), before)


@pytest.mark.parametrize("changed", ["config", "generation_config", "requires_grad"])
def test_same_shape_model_metadata_mismatch_precedes_any_mutation(
    tmp_path, tokenizer, changed
):
    source = TinyTrainer()
    source.agents[1] = model_for(tokenizer)
    source.optimizers[1] = torch.optim.AdamW(source.agents[1].parameters())
    state = capture_trainer_state(source)
    target = TinyTrainer()
    target.agents[1] = model_for(tokenizer)
    target.optimizers[1] = torch.optim.AdamW(target.agents[1].parameters())
    if changed == "config":
        target.agents[1].config.layer_norm_epsilon *= 10
    elif changed == "generation_config":
        target.agents[1].generation_config.repetition_penalty = 1.2
    else:
        target.agents[1].transformer.wte.weight.requires_grad_(False)
    assert {
        name: tuple(tensor.shape)
        for name, tensor in target.agents[1].state_dict().items()
    } == {
        name: tuple(tensor.shape)
        for name, tensor in source.agents[1].state_dict().items()
    }
    before = capture_trainer_state(target)

    with pytest.raises(ValueError, match="configuration"):
        restore_trainer_state(target, state, tmp_path)

    assert_nested_equal(capture_trainer_state(target), before)


@pytest.mark.parametrize("changed", ["vocab", "chat_template", "eos"])
def test_tokenizer_metadata_mismatch_precedes_any_mutation(
    tmp_path, tokenizer, changed
):
    source = TinyTrainer()
    source.tokenizers = [copy.deepcopy(tokenizer)]
    state = capture_trainer_state(source)
    target = TinyTrainer()
    target.tokenizers = [copy.deepcopy(tokenizer)]
    altered = target.tokenizers[0]
    if changed == "vocab":
        from tokenizers.models import WordLevel

        vocab = altered.get_vocab()
        vocab["good"], vocab["bad"] = vocab["bad"], vocab["good"]
        altered.backend_tokenizer.model = WordLevel(vocab, unk_token="<unk>")
        assert len(altered) == len(source.tokenizers[0])
    elif changed == "chat_template":
        altered.chat_template = "{{ messages[0]['content'] }}"
    else:
        altered.eos_token = "<unk>"
        assert altered.eos_token_id != source.tokenizers[0].eos_token_id
    before = capture_trainer_state(target)

    with pytest.raises(ValueError, match="tokenizers"):
        restore_trainer_state(target, state, tmp_path)

    assert_nested_equal(capture_trainer_state(target), before)


@pytest.mark.parametrize("shared", [True, False])
def test_wrapped_stateful_formatters_resume_their_outputs(tmp_path, shared):
    source = TinyTrainer()
    source_callbacks = (
        [StatefulFormatter()] if shared else [StatefulFormatter(), StatefulFormatter()]
    )
    source.formatters = build_formatters(
        source_callbacks[0] if shared else source_callbacks, 2
    )
    item = {"prompt": "prompt"}
    source.formatters[0](item)
    source.formatters[1](item, external_prompts="external")
    state = capture_trainer_state(source)
    expected = [
        source.formatters[0](item),
        source.formatters[1](item, external_prompts="external"),
    ]
    target = TinyTrainer()
    target_callbacks = (
        [StatefulFormatter()] if shared else [StatefulFormatter(), StatefulFormatter()]
    )
    target.formatters = build_formatters(
        target_callbacks[0] if shared else target_callbacks, 2
    )
    wrappers = list(target.formatters)

    restore_trainer_state(target, state, tmp_path)

    assert target.formatters == wrappers
    assert [
        target.formatters[0](item),
        target.formatters[1](item, external_prompts="external"),
    ] == expected
    assert [callback.calls for callback in target_callbacks] == [
        callback.calls for callback in source_callbacks
    ]


def test_stateful_evaluation_dataset_and_logging_hooks_resume(tmp_path):
    source = TinyTrainer()
    source.eval_dataset = StatefulEvaluationDataset()
    source.eval_logger = StatefulReward()
    source.eval_aggregator = StatefulReward()
    assert next(source.eval_dataset) == "first"
    source.eval_logger()
    source.eval_aggregator()
    source.eval_aggregator()
    state = capture_trainer_state(source)
    target = TinyTrainer()
    target.eval_dataset = StatefulEvaluationDataset()
    target.eval_logger = StatefulReward()
    target.eval_aggregator = StatefulReward()
    originals = target.eval_dataset, target.eval_logger, target.eval_aggregator

    restore_trainer_state(target, state, tmp_path)

    assert originals == (
        target.eval_dataset,
        target.eval_logger,
        target.eval_aggregator,
    )
    assert next(target.eval_dataset) == next(source.eval_dataset) == "second"
    assert target.eval_logger.calls == source.eval_logger.calls == 4
    assert target.eval_aggregator.calls == source.eval_aggregator.calls == 5


@pytest.mark.parametrize("pass_none", [True, False])
def test_formatter_adapters_keep_default_and_external_prompt_behavior(pass_none):
    item = {"prompt": "original"}
    default = build_formatters(None, 2, pass_none_external_prompts=pass_none)
    assert [formatter(item) for formatter in default] == ["original", "original"]
    assert default[0]({}, external_prompts="external") == "external"
    assert default[0](item, external_prompts="") == ""
    assert default[0]({}) == ""

    seen = []

    def accepts_external(value, external_prompts="default-argument"):
        seen.append((value, external_prompts))
        return external_prompts

    wrapped = build_formatters(
        accepts_external, 1, pass_none_external_prompts=pass_none
    )[0]
    expected_default = None if pass_none else "default-argument"
    assert wrapped(item) == expected_default
    assert wrapped(item, external_prompts="external") == "external"
    assert seen == [(item, expected_default), (item, "external")]
    single_argument = build_formatters(lambda value: value["prompt"], 1)[0]
    assert single_argument(item, external_prompts="ignored") == "original"
