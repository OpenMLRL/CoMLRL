"""Preference actor checkpoints retain lazily applied tokenizer special IDs."""

import copy

import pytest
import torch

from comlrl.trainers.pbrl import (
    MADPOConfig,
    MADPOIterConfig,
    MADPOIterTrainer,
    MADPOTrainer,
    MARLHFConfig,
    MARLHFIterConfig,
    MARLHFIterTrainer,
    MARLHFTrainer,
)
from comlrl.utils.checkpoint import _model_metadata

from helpers import (
    JointAdapter,
    assert_nested_equal,
    model_for,
    rng_state,
    seed_all,
    word_tokenizer,
    zero_reward,
)


TRAINERS = [
    (MADPOTrainer, MADPOConfig),
    (MARLHFTrainer, MARLHFConfig),
    (MADPOIterTrainer, MADPOIterConfig),
    (MARLHFIterTrainer, MARLHFIterConfig),
]


def _trainer(trainer_class, config_class, collaboration_mode, directory):
    actor_count = 1 if collaboration_mode == "centralized" else 2
    tokenizers = [word_tokenizer() for _ in range(actor_count)]
    if actor_count == 2:
        # The second actor must use its own tokenizer during restoration.
        tokenizers[1].pad_token = "<unk>"
        tokenizers[1].eos_token = "<agent_0>"
    agents = [model_for(tokenizer) for tokenizer in tokenizers]
    for agent in agents:
        agent.config.pad_token_id = None
        agent.config.eos_token_id = 3

    config = dict(
        collaboration_mode=collaboration_mode,
        num_agents=2,
        num_train_epochs=1,
        agent_devices="cpu",
        eval_interval=0,
        max_new_tokens=2,
        preference_num_candidates=2,
        preference_pairs_per_sample=1,
        rollout_buffer_size=1,
        train_batch_size=1,
    )
    if issubclass(config_class, MARLHFConfig):
        config.update(reward_model_device="cpu", num_generations=2)
    if issubclass(config_class, (MADPOIterConfig, MARLHFIterConfig)):
        config.update(
            num_iterations=2,
            preference_replay_dir=str(directory / "replay"),
            policy_checkpoint_dir=str(directory / "policies"),
            log_reward_distribution=False,
        )
    trainer = trainer_class(
        agents=agents,
        num_agents=2,
        tokenizer=tokenizers,
        args=config_class(**config),
        train_dataset=[{"prompt": "joint left right"}],
        reward_func=zero_reward,
        formatters=[lambda _: "left", lambda _: "right"],
        centralized_comparator_adapter=JointAdapter(),
    )
    trainer.verbose = False
    return trainer


def _generate(trainer):
    # Exercise the genuine HF generate method, including its configuration use.
    return [
        trainer._generate_completions(
            agent,
            [{"prompt": "joint left right"}],
            agent_idx=index,
            max_new_tokens=2,
            prompts_override=["joint left right"],
        )
        for index, agent in enumerate(trainer.agents)
    ]


def _score_and_update(trainer):
    losses = []
    for index, optimizer in enumerate(trainer.optimizers):
        tokenizer = trainer.tokenizers[index]
        prompt = torch.tensor(tokenizer.encode("joint left right"))
        completion = torch.tensor(tokenizer.encode("good bad good"))
        optimizer.zero_grad()
        loss = -trainer._sequence_log_prob(index, prompt, completion)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().clone())
    return losses


@pytest.mark.parametrize("trainer_class,config_class", TRAINERS)
@pytest.mark.parametrize("collaboration_mode", ["centralized", "decentralized"])
@pytest.mark.parametrize("first_operation", ["generate", "logprob"])
def test_preference_actor_specials_checkpoint_continues_identically(
    tmp_path, trainer_class, config_class, collaboration_mode, first_operation
):
    seed_all(71)
    source = _trainer(trainer_class, config_class, collaboration_mode, tmp_path)
    for actor in source.agents:
        assert actor.config.pad_token_id is None
        assert actor.config.eos_token_id == 3

    if first_operation == "generate":
        _generate(source)
    else:
        _score_and_update(source)
    for actor, tokenizer in zip(source.agents, source.tokenizers):
        assert actor.config.pad_token_id == tokenizer.pad_token_id
        assert actor.config.eos_token_id == tokenizer.eos_token_id
    metadata = [_model_metadata(actor) for actor in source.agents]
    checkpoint = tmp_path / "checkpoint"
    source.save_checkpoint(checkpoint)

    expected_generation = _generate(source)
    expected_losses = _score_and_update(source)
    expected_models = copy.deepcopy([actor.state_dict() for actor in source.agents])
    expected_optimizers = copy.deepcopy(
        [optimizer.state_dict() for optimizer in source.optimizers]
    )
    expected_rng = rng_state()

    seed_all(999)
    resumed = _trainer(trainer_class, config_class, collaboration_mode, tmp_path)
    assert all(actor.config.pad_token_id is None for actor in resumed.agents)
    resumed.load_checkpoint(checkpoint)
    assert [_model_metadata(actor) for actor in resumed.agents] == metadata

    assert_nested_equal(_generate(resumed), expected_generation)
    assert_nested_equal(_score_and_update(resumed), expected_losses)
    assert_nested_equal(
        [actor.state_dict() for actor in resumed.agents], expected_models
    )
    assert_nested_equal(
        [optimizer.state_dict() for optimizer in resumed.optimizers],
        expected_optimizers,
    )
    assert_nested_equal(rng_state(), expected_rng)
