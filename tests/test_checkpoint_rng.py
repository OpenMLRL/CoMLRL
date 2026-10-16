"""Checkpoints restore global, CUDA-routed and comparator generator streams."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import random
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from comlrl.trainers.preference.iterative import MADPOIterTrainer
import comlrl.utils.checkpoint as checkpoint
from comlrl.utils.checkpoint import capture_trainer_state, restore_trainer_state

from checkpoint_helpers import TinyTrainer
from helpers import assert_nested_equal


class _Agent:
    def __init__(self, device):
        self.device = torch.device(device)

    def parameters(self):
        return iter([SimpleNamespace(device=self.device)])


def _trainer(devices):
    trainer = object.__new__(MADPOIterTrainer)
    trainer.args = SimpleNamespace(
        comparator_policy="current_copy",
        comparator_generation_mode="decentralized",
        comparator_centralized_agent_index=0,
        max_new_tokens=1,
    )
    trainer.agents = [_Agent(device) for device in devices]
    trainer.num_agents = len(devices)
    trainer._resolve_comparator_devices = lambda: [
        agent.device for agent in trainer.agents
    ]
    trainer._initialize_comparator_rng()
    return trainer


@pytest.fixture
def cuda_rng(monkeypatch):
    """Model device relabeling with CPU generators; never require a CUDA driver."""
    generator_class = torch.Generator
    generators = {}

    def generator(device):
        key = str(device)
        if key not in generators:
            generators[key] = generator_class().manual_seed(123)
        return generators[key]

    monkeypatch.setattr(torch, "Generator", lambda device="cpu": generator_class())
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: generator(device).get_state().clone(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device: generator(device).set_state(state),
    )
    return lambda device, count=4: torch.rand(count, generator=generator(device))


def _draw(trainer, index, count=4, draw=None):
    agent = trainer.agents[index]
    with trainer._comparator_rng(agent):
        return torch.rand(count) if draw is None else draw(agent.device, count)


def _restore_streams(target, source):
    target.comparator_seed = source.comparator_seed
    target._comparator_rng_states = {
        key: value.clone() for key, value in source._comparator_rng_states.items()
    }
    target._comparator_rng_streams = copy.deepcopy(source._comparator_rng_streams)
    target._checkpoint_prepare_training()


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


def test_cuda_rng_remap_supports_unused_device_count_change(monkeypatch):
    trainer = object()
    state = {
        "rng": {"cuda": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]},
        "rng_devices": {"agents.0": "cuda:2", "critics.0": "cuda:0"},
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        checkpoint,
        "_rng_model_devices",
        lambda trainer: {"agents.0": "cuda:1", "critics.0": "cuda:0"},
    )
    assert checkpoint._cuda_rng_routes(trainer, state) == {1: 2, 0: 0}


@pytest.mark.parametrize(
    "saved,current",
    [
        (
            {"agents.0": "cuda:0", "agents.1": "cuda:1"},
            {"agents.0": "cuda:0", "agents.1": "cuda:0"},
        ),
        (
            {"agents.0": "cuda:0", "agents.1": "cuda:0"},
            {"agents.0": "cuda:0", "agents.1": "cuda:1"},
        ),
    ],
)
def test_ambiguous_actor_rng_merge_or_split_is_rejected(monkeypatch, saved, current):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(checkpoint, "_rng_model_devices", lambda trainer: current)
    with pytest.raises(ValueError, match="merge or split"):
        checkpoint._cuda_rng_routes(
            object(),
            {
                "rng": {"cuda": [torch.tensor([1]), torch.tensor([2])]},
                "rng_devices": saved,
            },
        )


@pytest.mark.parametrize("source,target", [("cpu", "cuda:0"), ("cuda:0", "cpu")])
def test_model_rng_backend_changes_rejected_without_cuda_runtime(
    monkeypatch, source, target
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        checkpoint, "_rng_model_devices", lambda trainer: {"agents.0": target}
    )
    with pytest.raises(ValueError, match="device backends"):
        checkpoint._cuda_rng_routes(
            object(), {"rng": {}, "rng_devices": {"agents.0": source}}
        )


def test_lazy_reward_rng_device_is_resolved_before_model_load(monkeypatch):
    trainer = SimpleNamespace(reward_device=torch.device("cuda:1"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        checkpoint, "_rng_model_devices", lambda trainer: {"agents.0": "cuda:0"}
    )
    state = {
        "rng": {"cuda": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]},
        "rng_devices": {"agents.0": "cuda:0", "reward_model.0": "cuda:2"},
    }
    assert checkpoint._cuda_rng_routes(trainer, state) == {0: 0, 1: 2}


def test_lazy_reward_implicit_cuda_device_uses_current_index(monkeypatch):
    trainer = SimpleNamespace(reward_device=torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        checkpoint, "_rng_model_devices", lambda trainer: {"agents.0": "cuda:0"}
    )
    state = {
        "rng": {"cuda": [torch.tensor([1]), torch.tensor([2])]},
        "rng_devices": {"agents.0": "cuda:0", "reward_model.0": "cuda:1"},
    }
    assert checkpoint._cuda_rng_routes(trainer, state) == {0: 0, 1: 1}


@pytest.mark.parametrize("probe_device", ["cpu", "meta"])
def test_rng_probe_override_uses_planned_placement(probe_device, monkeypatch):
    trainer = SimpleNamespace(
        agents=[torch.nn.Linear(1, 1, device=probe_device)],
        _checkpoint_rng_devices_override={"agents.0": "cuda:2"},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    state = {
        "rng": {"cuda": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]},
        "rng_devices": {"agents.0": "cuda:1"},
    }
    assert checkpoint._rng_model_devices(trainer) == {"agents.0": "cuda:2"}
    assert checkpoint._cuda_rng_routes(trainer, state)[2] == 1


def test_nested_rng_routes_restore_latest_parent_rng_values(monkeypatch):
    trainer = SimpleNamespace(_checkpoint_rng_devices_override={"agents.0": "cuda:0"})
    state = {
        "rng": {"cuda": [torch.tensor([11]), torch.tensor([22]), torch.tensor([33])]},
        "rng_devices": {"agents.0": "cuda:0"},
    }
    nested = [
        {
            "saved": {"agents.0": "cuda:0", "critics.0": "cuda:1"},
            "current": {"agents.0": "cuda:0", "critics.0": "cuda:2"},
        }
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state_all", lambda: [torch.tensor([99])] * 3
    )
    restored = []
    monkeypatch.setattr(checkpoint, "restore_rng_state", restored.append)
    checkpoint._restore_trainer_rng(trainer, state, nested)
    assert [value.item() for value in restored[0]["cuda"]] == [11, 22, 22]


def test_nested_rng_routes_cannot_conflict_with_parent_streams(monkeypatch):
    trainer = SimpleNamespace(_checkpoint_rng_devices_override={"agents.0": "cuda:0"})
    state = {
        "rng": {"cuda": [torch.tensor([11]), torch.tensor([22])]},
        "rng_devices": {"agents.0": "cuda:0"},
    }
    nested = [{"saved": {"critics.0": "cuda:1"}, "current": {"critics.0": "cuda:0"}}]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="merge or split"):
        checkpoint._cuda_rng_routes(trainer, state, nested)


@pytest.mark.parametrize(
    "seed, target, comparator", [(7, 7, -7), (0, 0, 1), (2**64 - 5, -5, 5)]
)
def test_comparator_seed_mirrors_signed_torch_seed(seed, target, comparator):
    torch.manual_seed(seed)
    trainer = _trainer(["cpu"])
    assert (trainer.target_seed, trainer.comparator_seed) == (target, comparator)


def test_default_same_device_comparators_keep_one_legacy_stream():
    trainer = _trainer(["cpu", "cpu"])
    global_state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(trainer.comparator_seed)
    actual = torch.cat([_draw(trainer, 0), _draw(trainer, 1)])
    assert torch.equal(actual, torch.rand(8, generator=generator))
    assert torch.equal(global_state, torch.get_rng_state())
    assert trainer._comparator_rng_streams is None
    assert set(trainer._comparator_rng_states) == {"cpu"}


def test_preparation_preserves_existing_stream_and_global_rng():
    trainer = _trainer(["cpu", "cpu"])
    _draw(trainer, 0)
    old_state = trainer._comparator_rng_states["cpu"].clone()
    global_state = torch.get_rng_state().clone()
    trainer._checkpoint_prepare_training()
    assert trainer._comparator_rng_streams == {
        "0": {"id": "comparator-0", "backend": "cpu"},
        "1": {"id": "comparator-0", "backend": "cpu"},
    }
    assert torch.equal(old_state, trainer._comparator_rng_states["comparator-0"])
    generator = torch.Generator().set_state(old_state)
    assert torch.equal(_draw(trainer, 1), torch.rand(4, generator=generator))
    assert torch.equal(global_state, torch.get_rng_state())


def test_device_relabel_and_regroup_keep_original_logical_streams(cuda_rng):
    baseline = _trainer(["cuda:0", "cuda:0", "cuda:1"])
    baseline._checkpoint_prepare_training()
    for index in [0, 1, 2]:
        _draw(baseline, index, draw=cuda_rng)
    resumed = _trainer(["cuda:5", "cuda:6", "cuda:5"])
    _restore_streams(resumed, baseline)
    assert (
        resumed._comparator_rng_streams["0"]["id"]
        == resumed._comparator_rng_streams["1"]["id"]
    )
    assert (
        resumed._comparator_rng_streams["2"]["id"]
        != resumed._comparator_rng_streams["0"]["id"]
    )
    global_states = {
        str(agent.device): torch.cuda.get_rng_state(agent.device)
        for agent in resumed.agents
    }
    for index in [1, 2, 0, 2]:
        assert torch.equal(
            _draw(baseline, index, draw=cuda_rng), _draw(resumed, index, draw=cuda_rng)
        )
    for device, state in global_states.items():
        assert torch.equal(state, torch.cuda.get_rng_state(torch.device(device)))


@pytest.mark.parametrize(
    "original, relocated",
    [
        (["cuda:0", "cuda:0"], ["cuda:3", "cuda:4"]),
        (["cuda:0", "cuda:1"], ["cuda:3", "cuda:3"]),
    ],
)
def test_shared_stream_or_shared_device_serializes_comparator_draws(
    original, relocated, cuda_rng
):
    source = _trainer(original)
    source._checkpoint_prepare_training()
    trainer = _trainer(relocated)
    _restore_streams(trainer, source)
    entered = threading.Event()
    release = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()

    def first():
        with trainer._comparator_rng(trainer.agents[0], agent_idx=0):
            entered.set()
            assert release.wait(2)
            return cuda_rng(trainer.agents[0].device)

    def second():
        second_attempted.set()
        with trainer._comparator_rng(trainer.agents[1], agent_idx=1):
            second_entered.set()
            return cuda_rng(trainer.agents[1].device)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_result = executor.submit(first)
        assert entered.wait(2)
        second_result = executor.submit(second)
        assert second_attempted.wait(2)
        try:
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        assert torch.equal(first_result.result(), _draw(source, 0, draw=cuda_rng))
        assert torch.equal(second_result.result(), _draw(source, 1, draw=cuda_rng))


def test_legacy_relabel_is_rejected_instead_of_silently_reseeding(cuda_rng):
    legacy = _trainer(["cuda:0"])
    _draw(legacy, 0, draw=cuda_rng)
    target = _trainer(["cuda:4"])
    target._comparator_rng_states = copy.deepcopy(legacy._comparator_rng_states)
    with pytest.raises(ValueError, match="Legacy comparator RNG device groups"):
        target._checkpoint_prepare_training()
    assert target._comparator_rng_streams is None
    assert set(target._comparator_rng_states) == {"cuda:0"}


def test_cross_backend_resume_and_missing_stream_state_are_rejected(cuda_rng):
    source = _trainer(["cpu"])
    source._checkpoint_prepare_training()
    target = _trainer(["cuda:4"])
    with pytest.raises(ValueError, match="backend changed"):
        _restore_streams(target, source)
    source._comparator_rng_states.clear()
    with pytest.raises(ValueError, match="do not match logical streams"):
        source._checkpoint_prepare_training()
    with pytest.raises(ValueError, match="refusing to reseed"):
        _draw(source, 0)


def test_centralized_comparator_uses_only_selected_slot():
    trainer = _trainer(["cpu", "cpu"])
    trainer.args.comparator_generation_mode = "centralized"
    trainer.args.comparator_centralized_agent_index = 1
    trainer._checkpoint_prepare_training()
    assert trainer._comparator_rng_streams == {
        "1": {"id": "comparator-1", "backend": "cpu"}
    }
    _draw(trainer, 1)


def test_default_generation_preserves_one_argument_rng_override():
    trainer = _trainer(["cpu", "cpu"])
    calls = []

    @contextmanager
    def legacy_override(agent):
        calls.append(agent)
        yield

    trainer._comparator_rng = legacy_override
    trainer._run_agent_tasks = lambda function: [function(0), function(1)]
    trainer._generate_completions_with_external_prompts = lambda *args, **kwargs: {}
    trainer._generate_policy_outputs_for_item(
        trainer.agents, {}, num_candidates=1, use_comparator_rng=True
    )
    assert calls == trainer.agents
