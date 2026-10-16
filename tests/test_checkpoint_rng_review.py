"""Comparator RNG portability preserves the original per-device stream groups."""

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from types import SimpleNamespace

import pytest
import torch

from comlrl.trainers.preference.iterative import MADPOIterTrainer
from test_checkpoint_preference import (
    Interrupted,
    assert_same_training,
    iterative_trainer,
    seed,
    stop_after,
)


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


@pytest.fixture(autouse=True)
def preserve_cpu_rng():
    state = torch.get_rng_state().clone()
    try:
        yield
    finally:
        torch.set_rng_state(state)


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


def test_checkpoint_streams_resume_matches_default_training(tmp_path):
    seed(25)
    baseline = iterative_trainer(tmp_path / "baseline")
    baseline.train()
    expected = torch.get_rng_state().clone()

    seed(25)
    interrupted = iterative_trainer(tmp_path / "interrupted")
    stop_after(interrupted, lambda progress: progress.get("iteration") == 2)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path / "checkpoints")
    seed(777)
    resumed = iterative_trainer(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoints" / "checkpoint-2")
    assert_same_training(baseline, resumed)
    assert baseline.events == interrupted.events + resumed.events
    assert torch.equal(torch.get_rng_state(), expected)
    stream = resumed._comparator_rng_streams["0"]["id"]
    assert torch.equal(
        baseline._comparator_rng_states["cpu"], resumed._comparator_rng_states[stream]
    )
