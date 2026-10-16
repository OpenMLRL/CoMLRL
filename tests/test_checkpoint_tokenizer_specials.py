"""Checkpoint resume after real generation normalizes tokenizer special IDs."""

import copy

from datasets import Dataset
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from comlrl.trainers.reinforce import (
    CentralizedMAGRPOConfig,
    CentralizedMAGRPOTrainer,
    MAGRPOConfig,
    MAGRPOTrainer,
)
from comlrl.utils.checkpoint import _decode, _encode

from checkpoint_helpers import Interrupted, stop_after
from helpers import assert_nested_equal, rng_state, seed_all, word_tokenizer


class _SplitRoles:
    """Accept generated words as role contributions, without a model stub."""

    def build_prompt(self, item, agent_prompts):
        return "joint " + " ".join(agent_prompts)

    def parse_completion(self, completion, item, num_agents):
        words = completion.split()
        return [" ".join(words[index::num_agents]) for index in range(num_agents)]


def _reward(*agent_completions, **kwargs):
    return [
        1.0 + sum(sum(map(ord, text)) for text in completions) / 1000.0
        for completions in zip(*agent_completions)
    ]


def _make_trainer(variant="magrpo", *, eos_mismatch=False, seed=37):
    seed_all(seed)
    actor_count = 1 if variant == "centralized" else 2
    tokenizers = [word_tokenizer() for _ in range(actor_count)]
    if variant == "heterogeneous":
        tokenizers[1].pad_token = "<unk>"
        tokenizers[1].eos_token = "right"
    agents = [
        GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=64,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                pad_token_id=None,
                eos_token_id=2 if eos_mismatch else tokenizer.eos_token_id,
            )
        )
        for tokenizer in tokenizers
    ]
    common = dict(
        num_agents=2,
        num_turns=1,
        num_train_epochs=2,
        num_generations=2,
        max_new_tokens=3,
        temperature=0.9,
        top_p=1.0,
        top_k=0,
        agent_devices="cpu",
        agent_learning_rate=1e-3,
        eval_interval=0,
        rollout_buffer_size=2,
        train_batch_size=2,
        early_termination_threshold=None,
        advantage_mode="raw",
        advantage_normalization=False,
    )
    kwargs = dict(
        agents=agents,
        tokenizer=tokenizers,
        num_agents=2,
        train_dataset=Dataset.from_dict({"prompt": ["left", "right"]}),
        reward_func=_reward,
        formatters=lambda item: item["prompt"],
    )
    if variant == "centralized":
        trainer = CentralizedMAGRPOTrainer(
            args=CentralizedMAGRPOConfig(**common),
            centralized_adapter=_SplitRoles(),
            **kwargs,
        )
    else:
        trainer = MAGRPOTrainer(args=MAGRPOConfig(**common), **kwargs)
    trainer.verbose = False
    # The fixture starts in the same state as a freshly loaded Qwen model.
    assert all(agent.config.pad_token_id is None for agent in trainer.agents)
    return trainer


def _model_states(trainer):
    return [copy.deepcopy(agent.state_dict()) for agent in trainer.agents]


@pytest.mark.parametrize(
    "variant,eos_mismatch",
    [
        ("magrpo", False),
        ("magrpo", True),
        ("centralized", True),
        ("heterogeneous", True),
    ],
)
def test_real_generation_resume_matches_uninterrupted_with_special_id_mismatch(
    variant, eos_mismatch, tmp_path
):
    baseline = _make_trainer(variant, eos_mismatch=eos_mismatch)
    initial = _model_states(baseline)
    seed_all(83)
    baseline.train()
    expected_rng = rng_state()
    expected_models = _model_states(baseline)
    expected_optimizers = [optimizer.state_dict() for optimizer in baseline.optimizers]
    for before, after in zip(initial, expected_models):
        assert any(
            not torch.equal(before[name], value) for name, value in after.items()
        )

    interrupted = _make_trainer(variant, eos_mismatch=eos_mismatch)
    stop_after(interrupted, lambda progress: progress["epoch"] == 1)
    seed_all(83)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)
    for agent, tokenizer in zip(interrupted.agents, interrupted.tokenizers):
        assert agent.config.pad_token_id == tokenizer.pad_token_id
        assert agent.config.eos_token_id == tokenizer.eos_token_id

    resumed = _make_trainer(variant, eos_mismatch=eos_mismatch, seed=991)
    seed_all(1021)
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoint-1")

    assert_nested_equal(_model_states(resumed), expected_models)
    assert_nested_equal(
        [optimizer.state_dict() for optimizer in resumed.optimizers],
        expected_optimizers,
    )
    assert resumed.env_step == baseline.env_step > interrupted.env_step > 0
    assert resumed._checkpoint_progress["epoch"] == 2
    assert_nested_equal(rng_state(), expected_rng)


@pytest.mark.parametrize("changed_config", ["config", "generation_config"])
def test_special_id_compatibility_does_not_allow_other_configuration_changes(
    changed_config, tmp_path
):
    interrupted = _make_trainer(eos_mismatch=True)
    stop_after(interrupted, lambda progress: progress["epoch"] == 1)
    seed_all(83)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)

    resumed = _make_trainer(eos_mismatch=True, seed=991)
    if changed_config == "config":
        resumed.agents[0].config.activation_function = "relu"
    else:
        resumed.agents[0].generation_config.temperature = 0.7
    before = _model_states(resumed)
    before_config = copy.deepcopy(resumed.agents[0].config.to_dict())
    with pytest.raises(ValueError, match="model or generation configuration"):
        resumed.train(resume_from_checkpoint=tmp_path / "checkpoint-1")
    assert resumed.env_step == 0
    assert_nested_equal(_model_states(resumed), before)
    assert resumed.agents[0].config.to_dict() == before_config


def test_checkpoint_before_generation_preserves_original_special_ids(tmp_path):
    source = _make_trainer("heterogeneous", eos_mismatch=True)
    original = [
        (copy.deepcopy(agent.config.to_dict()), agent.generation_config.to_dict())
        for agent in source.agents
    ]
    source.save_checkpoint(tmp_path / "pristine")

    restored = _make_trainer("heterogeneous", eos_mismatch=True, seed=991)
    restored.load_checkpoint(tmp_path / "pristine")
    assert_nested_equal(_model_states(restored), _model_states(source))
    for agent, (config, generation_config) in zip(restored.agents, original):
        assert agent.config.to_dict() == config
        assert agent.generation_config.to_dict() == generation_config
        assert agent.config.pad_token_id is None
        assert agent.config.eos_token_id == 2


def test_direct_load_restores_post_generation_special_ids(tmp_path):
    source = _make_trainer("heterogeneous", eos_mismatch=True)
    stop_after(source, lambda progress: progress["epoch"] == 1)
    seed_all(83)
    with pytest.raises(Interrupted):
        source.train(checkpoint_dir=tmp_path)

    restored = _make_trainer("heterogeneous", eos_mismatch=True, seed=991)
    restored.load_checkpoint(tmp_path / "checkpoint-1")
    for agent, saved_agent, tokenizer in zip(
        restored.agents, source.agents, restored.tokenizers
    ):
        assert agent.config.pad_token_id == tokenizer.pad_token_id
        assert agent.config.eos_token_id == tokenizer.eos_token_id
        assert agent.config.to_dict() == saved_agent.config.to_dict()
        assert (
            agent.generation_config.to_dict() == saved_agent.generation_config.to_dict()
        )


@pytest.mark.parametrize("failed_validation", ["optimizer", "assets"])
def test_later_preflight_failure_does_not_mutate_special_ids(
    failed_validation, tmp_path, monkeypatch
):
    interrupted = _make_trainer("heterogeneous", eos_mismatch=True)
    stop_after(interrupted, lambda progress: progress["epoch"] == 1)
    seed_all(83)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)
    state_path = tmp_path / "checkpoint-1" / "training_state.pt"
    state = _decode(torch.load(state_path, map_location="cpu", weights_only=True))
    resumed = _make_trainer("heterogeneous", eos_mismatch=True, seed=991)
    if failed_validation == "optimizer":
        del state["optimizers"]["optimizers"]["states"][0]["state"]
        error = "Incomplete checkpoint optimizers optimizer state"
    else:
        state["assets"] = {}

        def reject_assets(directory, manifest):
            raise ValueError("Invalid test checkpoint assets")

        monkeypatch.setattr(
            resumed, "_checkpoint_validate_assets", reject_assets, raising=False
        )
        error = "Invalid test checkpoint assets"
    torch.save(_encode(state), state_path)
    before = _model_states(resumed)
    before_configs = [copy.deepcopy(agent.config.to_dict()) for agent in resumed.agents]
    before_rng = rng_state()
    with pytest.raises(ValueError, match=error):
        resumed.load_checkpoint(tmp_path / "checkpoint-1")
    assert resumed.env_step == 0
    assert_nested_equal(_model_states(resumed), before)
    assert [agent.config.to_dict() for agent in resumed.agents] == before_configs
    assert_nested_equal(rng_state(), before_rng)


@pytest.mark.parametrize("first_actor_already_matches", [False, True])
def test_shared_config_with_conflicting_actor_special_ids_is_rejected_atomically(
    first_actor_already_matches, tmp_path
):
    source = _make_trainer("heterogeneous", eos_mismatch=True)
    stop_after(source, lambda progress: progress["epoch"] == 1)
    seed_all(83)
    with pytest.raises(Interrupted):
        source.train(checkpoint_dir=tmp_path)
    assert source.agents[0].config is not source.agents[1].config

    restored = _make_trainer("heterogeneous", eos_mismatch=True, seed=991)
    shared_config = restored.agents[0].config
    if first_actor_already_matches:
        shared_config.pad_token_id = restored.tokenizers[0].pad_token_id
        shared_config.eos_token_id = restored.tokenizers[0].eos_token_id
    restored.agents[1].config = shared_config
    restored.agents[1].transformer.config = shared_config
    before_models = _model_states(restored)
    before_config = copy.deepcopy(shared_config.to_dict())
    before_generation_configs = [
        copy.deepcopy(agent.generation_config.to_dict()) for agent in restored.agents
    ]
    before_optimizers = [
        copy.deepcopy(optimizer.state_dict()) for optimizer in restored.optimizers
    ]
    before_rng = rng_state()

    with pytest.raises(ValueError, match="configuration"):
        restored.load_checkpoint(tmp_path / "checkpoint-1")

    assert restored.agents[0].config is restored.agents[1].config
    assert shared_config.to_dict() == before_config
    assert_nested_equal(_model_states(restored), before_models)
    assert [
        agent.generation_config.to_dict() for agent in restored.agents
    ] == before_generation_configs
    assert_nested_equal(
        [optimizer.state_dict() for optimizer in restored.optimizers],
        before_optimizers,
    )
    assert restored.env_step == 0
    assert_nested_equal(rng_state(), before_rng)
