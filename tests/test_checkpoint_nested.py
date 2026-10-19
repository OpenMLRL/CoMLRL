"""MARLHF's nested actor-critic child validates and restores before the parent."""

import random
import threading

import numpy as np
import pytest
import torch

from comlrl.trainers.pbrl import marlhf
from comlrl.utils.checkpoint import (
    STATE_FILENAME,
    _decode,
    capture_trainer_state,
    restore_trainer_state,
)

from checkpoint_helpers import Interrupted, TinyACTrainer, marlhf_ac_trainer, stop_after
from helpers import assert_nested_equal, model_for, rng_state, seed_all


@pytest.fixture
def nested_state(monkeypatch):
    monkeypatch.setattr(marlhf, "MAACTrainer", TinyACTrainer)
    seed_all(313)
    source = marlhf_ac_trainer()
    source.train(checkpoint_interval=2)
    return capture_trainer_state(source)


def corrupt_child(state, corruption):
    child = state["extra"]["actor_critic"]
    if corruption == "configuration":
        child["config"]["critic_type"] = "q"
    elif corruption == "critic shape":
        child["models"]["critics"]["states"][0]["weight"] = torch.ones(2, 1)
    elif corruption == "actor shape":
        child["models"]["agents"]["states"][0]["model.weight"] = torch.ones(2, 1)
    elif corruption == "reference shape":
        child["models"]["reference_models"]["states"][0]["weight"] = torch.ones(2, 1)
    elif corruption == "optimizer shape":
        child["optimizers"]["critic_optimizer"]["states"][0]["state"][0]["exp_avg"] = (
            torch.ones(2, 1)
        )
    elif corruption == "missing component":
        del child["optimizers"]["critic_optimizer"]
    else:
        raise AssertionError(corruption)


class CustomMARLHF(marlhf.MARLHFTrainer):
    def _create_actor_critic_trainer(self):
        child = super()._create_actor_critic_trainer()
        child.args.value_loss_coef = 0.73
        return child


class LockBearingInput:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = 0

    def __len__(self):
        return 1

    def __call__(self, *args, **kwargs):
        raise AssertionError("Child construction must not execute user callbacks.")

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def custom_parent():
    trainer = marlhf_ac_trainer()
    trainer.__class__ = CustomMARLHF
    return trainer


def configure_nested_cuda_state(
    state,
    target,
    monkeypatch,
    *,
    source_critic="cuda:1",
    target_critic="cuda:2",
    source_reward="cuda:0",
    target_reward="cuda:0",
):
    """Exercise routing with CPU weights and isolated mock CUDA generators."""
    current_rng = [
        torch.tensor([value], dtype=torch.uint8) for value in (201, 202, 203)
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [value.clone() for value in current_rng],
    )

    def set_rng(states):
        current_rng[:] = [value.clone() for value in states]

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_rng)
    state["rng"]["cuda"] = [
        torch.tensor([value], dtype=torch.uint8) for value in (101, 102, 103)
    ]
    state["rng_devices"] = {
        "agents.0": "cuda:0",
        "reward_model.0": source_reward,
    }
    child_state = state["extra"]["actor_critic"]
    child_state["rng"]["cuda"] = [
        torch.tensor([value], dtype=torch.uint8) for value in (11, 12, 13)
    ]
    child_state["rng_devices"] = {
        "agents.0": "cuda:0",
        "critics.0": source_critic,
    }
    target._checkpoint_rng_devices_override = {
        "agents.0": "cuda:0",
        "reward_model.0": target_reward,
    }
    target._build_actor_critic_common_config = lambda: {
        "num_train_epochs": 3,
        "num_agents": 1,
        "num_turns": 1,
        "parallel_training": "mp",
        "agent_devices": "cuda:0",
        "critic_devices": target_critic,
    }
    return current_rng


@pytest.mark.parametrize(
    "corruption",
    [
        "configuration",
        "critic shape",
        "actor shape",
        "reference shape",
        "optimizer shape",
        "missing component",
    ],
)
def test_nested_incompatibility_leaves_parent_and_rng_unchanged(
    nested_state, corruption
):
    corrupt_child(nested_state, corruption)
    seed_all(719)
    target = marlhf_ac_trainer()
    before = capture_trainer_state(target)
    with pytest.raises(ValueError, match="configuration|shape|components"):
        restore_trainer_state(target, nested_state)
    assert_nested_equal(capture_trainer_state(target), before)
    assert target.reward_model is None
    assert not hasattr(target, "_checkpoint_pending_ac_state")
    # The validation-only constructor allocates no real actor/critic weights.
    probe = TinyACTrainer.instances[-1]
    assert next(probe.agents[0].parameters()).device.type == "meta"
    assert next(probe.critics[0].parameters()).device.type == "meta"


@pytest.mark.parametrize("raise_in_constructor", [False, True])
def test_nested_probe_isolates_mutating_factory_inputs_and_rng(
    nested_state, monkeypatch, raise_in_constructor
):
    corrupt_child(nested_state, "configuration")
    target = marlhf_ac_trainer()
    target.tokenizers = []
    before = capture_trainer_state(target)

    def mutating_factory(**kwargs):
        with torch.no_grad():
            kwargs["agents"][0].weight.add_(100)
        kwargs["agents"][0].eval()
        kwargs["tokenizer"].append("changed only in the probe")
        random.random()
        np.random.random()
        torch.rand((), device="cpu")
        if raise_in_constructor:
            raise RuntimeError("custom constructor failed")
        return TinyACTrainer(**kwargs)

    monkeypatch.setattr(marlhf, "MAACTrainer", mutating_factory)
    error = RuntimeError if raise_in_constructor else ValueError
    with pytest.raises(error, match="custom constructor|configuration"):
        restore_trainer_state(target, nested_state)
    assert_nested_equal(capture_trainer_state(target), before)
    assert target.tokenizers == []


def test_nested_custom_factory_that_reads_values_uses_cpu_fallback(
    nested_state, monkeypatch
):
    corrupt_child(nested_state, "configuration")
    seen = []

    def value_reading_factory(**kwargs):
        seen.append(kwargs["agents"][0].weight.device.type)
        kwargs["agents"][0].weight.sum().item()
        return TinyACTrainer(**kwargs)

    monkeypatch.setattr(marlhf, "MAACTrainer", value_reading_factory)
    target = marlhf_ac_trainer()
    before = capture_trainer_state(target)
    with pytest.raises(ValueError, match="configuration"):
        restore_trainer_state(target, nested_state)
    assert seen == ["meta", "cpu"]
    assert_nested_equal(capture_trainer_state(target), before)


def test_custom_probe_cannot_allocate_on_training_accelerator(
    nested_state, monkeypatch
):
    calls = []

    def accelerator_factory(**kwargs):
        device = kwargs["agents"][0].weight.device.type
        calls.append(device)
        kwargs["agents"][0].weight.sum().item()  # Force the CPU fallback.
        torch.ones(1, device="cuda")  # Rejected even on a CPU-only test host.
        return TinyACTrainer(**kwargs)

    monkeypatch.setattr(marlhf, "MAACTrainer", accelerator_factory)
    target = marlhf_ac_trainer()
    before = capture_trainer_state(target)
    with pytest.raises(ValueError, match="CPU/meta devices"):
        restore_trainer_state(target, nested_state)
    assert calls == ["meta", "cpu"]
    assert_nested_equal(capture_trainer_state(target), before)


@pytest.mark.parametrize(
    "algorithm,separate,critic_source",
    [
        ("maac", True, "object"),
        ("maac", True, "local"),
        ("iac", True, "object"),
        ("iac", False, "none"),
    ],
)
def test_real_ac_factory_preflights_with_meta_weights(
    tokenizer, tmp_path, monkeypatch, algorithm, separate, critic_source
):
    critic_kwargs = {}
    if critic_source == "object":
        critic_kwargs["critics"] = [model_for(tokenizer)]
    elif critic_source == "local":
        model_for(tokenizer).save_pretrained(tmp_path / "critic")
        critic_kwargs["critic_model"] = str(tmp_path / "critic")
    trainer = marlhf.MARLHFTrainer(
        agents=[model_for(tokenizer)],
        num_agents=1,
        tokenizer=tokenizer,
        reward_func=lambda *args, **kwargs: [1.0],
        metrics_callback=LockBearingInput(),
        train_dataset=LockBearingInput(),
        args=marlhf.MARLHFConfig(
            num_agents=1,
            num_turns=1,
            num_generations=2,
            num_train_epochs=1,
            agent_devices="cpu",
            critic_devices="cpu",
            reward_model_device="cpu",
            reference_kl_enabled=True,
            rl_algorithm=algorithm,
            use_separate_critic=separate,
        ),
        **critic_kwargs,
    )
    trainer.agents[0].eval()
    trainer.agents[0].transformer.h[0].train()
    trainer.agents[0].transformer.wte.weight.requires_grad_(False)
    child = trainer._create_actor_critic_trainer()
    saved = capture_trainer_state(child)
    if critic_source == "local":

        def forbid_pretrained_weight_load(*args, **kwargs):
            raise AssertionError("Meta validation must not load pretrained weights.")

        # Transformers 5 rejects from_pretrained inside a meta device context;
        # older versions can needlessly materialize checkpoint tensors there.
        monkeypatch.setattr(
            marlhf.AutoModelForCausalLM,
            "from_pretrained",
            forbid_pretrained_weight_load,
        )
    before = capture_trainer_state(trainer)
    trainer._checkpoint_validate_extra_state({"actor_critic": saved}, None)
    assert_nested_equal(capture_trainer_state(trainer), before)
    probe = trainer._checkpoint_actor_critic_probe("meta")
    assert probe.metrics_callback is trainer.metrics_callback
    assert probe.train_dataset is trainer.train_dataset
    assert all(
        p.device.type == "meta" for actor in probe.agents for p in actor.parameters()
    )
    assert all(
        p.device.type == "meta" for critic in probe.critics for p in critic.parameters()
    )
    copied_actor = probe.agents[0].model
    assert copied_actor.lm_head.weight is copied_actor.transformer.wte.weight
    assert {name: p.requires_grad for name, p in copied_actor.named_parameters()} == {
        name: p.requires_grad for name, p in trainer.agents[0].named_parameters()
    }
    assert {name: module.training for name, module in copied_actor.named_modules()} == {
        name: module.training for name, module in trainer.agents[0].named_modules()
    }


def test_nested_factory_override_resumes_identically(tmp_path, monkeypatch):
    monkeypatch.setattr(marlhf, "MAACTrainer", TinyACTrainer)
    seed_all(412)
    continuous = custom_parent()
    continuous.train()
    continuous_child = TinyACTrainer.instances[-1]
    expected_rng = rng_state()

    seed_all(412)
    interrupted = custom_parent()
    stop_after(interrupted, lambda p: p.get("rl", {}).get("epoch") == 1)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)

    seed_all(611)
    resumed = custom_parent()
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoint-2")
    child = resumed._checkpoint_ac_trainer
    for left, right in (
        (continuous.agents[0], resumed.agents[0]),
        (continuous_child.critics[0], child.critics[0]),
        (continuous_child.reference_models[0], child.reference_models[0]),
        (continuous_child.agent_optimizers[0], child.agent_optimizers[0]),
        (continuous_child.critic_optimizer, child.critic_optimizer),
    ):
        assert_nested_equal(left.state_dict(), right.state_dict())
    assert child.args.value_loss_coef == 0.73
    assert continuous.env_step == resumed.env_step
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize("checkpoint_number", [1, 2])
def test_loading_reused_marlhf_parent_discards_stale_child_before_manual_save(
    tmp_path,
    monkeypatch,
    checkpoint_number,
):
    monkeypatch.setattr(marlhf, "MAACTrainer", TinyACTrainer)
    seed_all(31)
    source = marlhf_ac_trainer()
    stop_after(source, lambda progress: progress.get("rl", {}).get("epoch") == 1)
    with pytest.raises(Interrupted):
        source.train(checkpoint_dir=tmp_path / "source")
    source_path = tmp_path / "source" / f"checkpoint-{checkpoint_number}"
    expected = _decode(torch.load(source_path / STATE_FILENAME, weights_only=True))

    seed_all(74)
    target = marlhf_ac_trainer()
    target.train(checkpoint_dir=tmp_path / "previous-run")
    old_child = target._checkpoint_ac_trainer
    assert old_child is not None
    target.load_checkpoint(source_path)
    assert target._checkpoint_ac_trainer is None
    assert_nested_equal(
        target._checkpoint_pending_ac_state,
        expected["extra"].get("actor_critic"),
    )
    manual = tmp_path / "manual"
    target.save_checkpoint(manual)
    saved = _decode(torch.load(manual / STATE_FILENAME, weights_only=True))
    assert_nested_equal(saved["extra"], expected["extra"])


@pytest.mark.parametrize("cpu_fallback", [False, True])
def test_nested_critic_remap_uses_parent_latest_rng(
    nested_state, monkeypatch, cpu_fallback
):
    target = marlhf_ac_trainer()
    current_rng = configure_nested_cuda_state(nested_state, target, monkeypatch)
    seen = []

    def factory(**kwargs):
        seen.append(kwargs["agents"][0].weight.device.type)
        if cpu_fallback:
            kwargs["agents"][0].weight.sum().item()
        return TinyACTrainer(**kwargs)

    monkeypatch.setattr(marlhf, "MAACTrainer", factory)
    restore_trainer_state(target, nested_state)
    assert seen == (["meta", "cpu"] if cpu_fallback else ["meta"])
    assert current_rng[0].item() == 101
    # The child state contains the older value 12. Only its device mapping is
    # used; generator state comes from the final parent snapshot, value 102.
    assert current_rng[2].item() == 102
    assert TinyACTrainer.instances[-1]._checkpoint_rng_devices_override == {
        "agents.0": "cuda:0",
        "critics.0": "cuda:2",
    }


@pytest.mark.parametrize(
    "source_critic,target_critic,source_reward,target_reward,error",
    [
        ("cuda:1", "cuda:2", "cuda:2", "cuda:2", "merge or split"),
        ("cuda:0", "cuda:2", "cuda:0", "cuda:0", "merge or split"),
        ("cuda:1", "cpu", "cuda:0", "cuda:0", "device backends"),
    ],
)
def test_nested_rng_topology_failure_precedes_parent_mutation(
    nested_state,
    monkeypatch,
    source_critic,
    target_critic,
    source_reward,
    target_reward,
    error,
):
    target = marlhf_ac_trainer()
    configure_nested_cuda_state(
        nested_state,
        target,
        monkeypatch,
        source_critic=source_critic,
        target_critic=target_critic,
        source_reward=source_reward,
        target_reward=target_reward,
    )
    before = capture_trainer_state(target)
    with pytest.raises(ValueError, match=error):
        restore_trainer_state(target, nested_state)
    assert_nested_equal(capture_trainer_state(target), before)
    assert target.reward_model is None
    assert not hasattr(target, "_checkpoint_pending_ac_state")
