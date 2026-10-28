"""In-memory trainer state capture/restore; failures never partially mutate."""

import copy
from dataclasses import dataclass, field

import numpy as np
import pytest
from tokenizers.models import WordLevel
import torch

from comlrl.trainers.reinforce.magrpo import MAGRPOConfig, NodeSample
import comlrl.utils.checkpoint as checkpoint
from comlrl.utils.checkpoint import (
    capture_trainer_state,
    restore_trainer_state,
)
from comlrl.utils.formatters import build_formatters

from checkpoint_helpers import (
    StatefulReward,
    TinyTrainer,
    advance_models,
    assert_models_equal,
)
from helpers import assert_nested_equal, model_for


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


class DifferentTrainer(TinyTrainer):
    pass


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


def test_stateful_callback_restores_its_state(tmp_path):
    source = TinyTrainer()
    source.reward_func.calls = 19
    state = capture_trainer_state(source)
    target = TinyTrainer()
    original_callback = target.reward_func

    restore_trainer_state(target, state, tmp_path)

    assert target.reward_func is original_callback
    assert target.reward_func.calls == 19


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


def test_run_length_and_logging_config_can_change(tmp_path):
    source = TinyTrainer(num_train_epochs=2, logging_steps=1)
    state = capture_trainer_state(source)
    target = TinyTrainer(num_train_epochs=5, logging_steps=4)

    restore_trainer_state(target, state, tmp_path)

    assert target.args.num_train_epochs == 5
    assert target.args.logging_steps == 4
    assert_models_equal(target, source)


def test_portable_placement_fields_accept_legacy_snapshot(tmp_path):
    source = TinyTrainer()
    source.args.comparator_devices = "cuda:2"
    state = capture_trainer_state(source)
    # Version-1 snapshots written before these fields became portable.
    state["config"].update(comparator_devices="cuda:2", parallel_training="mp")
    target = TinyTrainer()
    target.args.comparator_devices = "cuda:1"
    target.args.parallel_training = "none"
    restore_trainer_state(target, state, tmp_path)
    assert target.args.comparator_devices == "cuda:1"


@pytest.mark.parametrize("lazy", [False, True])
def test_absent_reward_optimizer_is_authoritative(tmp_path, lazy):
    source = TinyTrainer()
    source.reward_optimizer = None
    state = capture_trainer_state(source)
    target = TinyTrainer()
    if lazy:
        target.reward_model = target.reward_optimizer = None

        def initialize():
            target.reward_model = torch.nn.Linear(3, 1)
            target.reward_optimizer = torch.optim.AdamW(
                target.reward_model.parameters()
            )

        target._init_reward_model = initialize
    restore_trainer_state(target, state, tmp_path)
    assert target.reward_optimizer is None
    assert_nested_equal(
        target.reward_model.state_dict(), source.reward_model.state_dict()
    )


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


def test_lazy_factory_device_mismatch_rolls_back_before_state_load(monkeypatch):
    state = capture_trainer_state(TinyTrainer())
    target = TinyTrainer()
    target.reward_model = target.reward_optimizer = None
    target.reward_device = torch.device("cpu")
    before = {
        name: value.clone() for name, value in target.agents[0].state_dict().items()
    }
    rng = torch.get_rng_state().clone()

    def initialize():
        target.reward_model = torch.nn.Linear(3, 1)
        target.reward_optimizer = torch.optim.AdamW(target.reward_model.parameters())

    actual_devices = checkpoint._rng_model_devices

    def devices(trainer):
        result = actual_devices(trainer)
        if trainer is target and trainer.reward_model is not None:
            result["reward_model.0"] = "cuda:0"
        return result

    target._init_reward_model = initialize
    monkeypatch.setattr(checkpoint, "_rng_model_devices", devices)
    with pytest.raises(ValueError, match="device backends"):
        restore_trainer_state(target, state)
    assert target.reward_model is None
    assert target.reward_optimizer is None
    assert_nested_equal(target.agents[0].state_dict(), before)
    assert torch.equal(torch.get_rng_state(), rng)


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


@pytest.mark.parametrize("value,resumes", [(None, True), (4, False)])
def test_config_field_newer_than_the_checkpoint_resumes_at_its_default(
    tmp_path, value, resumes
):
    state = capture_trainer_state(TinyTrainer())
    # A checkpoint written before this field existed has no entry for it.
    del state["config"]["train_batch_size"]
    target = TinyTrainer()
    target.args.train_batch_size = value
    if resumes:
        restore_trainer_state(target, state, tmp_path)
    else:
        with pytest.raises(ValueError, match="does not match: train_batch_size"):
            restore_trainer_state(target, state, tmp_path)


def test_config_defaults_are_built_only_for_fields_the_checkpoint_lacks(tmp_path):
    calls = []

    def factory():
        calls.append(torch.rand(1))  # draws from the global RNG
        return []

    @dataclass
    class FactoryConfig(MAGRPOConfig):
        extra_items: list = field(default_factory=factory)

    def trainer():
        target = TinyTrainer()
        target.args = FactoryConfig(num_agents=2, num_turns=1, agent_devices="cpu")
        return target

    state = capture_trainer_state(trainer())
    target = trainer()  # building the config itself runs the factory
    calls.clear()
    restore_trainer_state(target, state, tmp_path, restore_rng=False)
    assert calls == []
    del state["config"]["extra_items"]
    target = trainer()
    calls.clear()
    before = torch.get_rng_state()
    restore_trainer_state(target, state, tmp_path, restore_rng=False)
    assert len(calls) == 1 and torch.equal(torch.get_rng_state(), before)


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
