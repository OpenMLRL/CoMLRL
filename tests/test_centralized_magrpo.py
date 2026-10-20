"""Centralized MAGRPO trains one joint actor on all roles' text."""

import pytest
import torch

from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer

from helpers import BAD, GOOD, JointAdapter, make_centralized_magrpo_trainer, model_for


def test_config_rejects_unknown_mode_and_multiple_turns():
    assert MAGRPOConfig().collaboration_mode == "decentralized"
    assert MAGRPOConfig().num_turns == 2
    with pytest.raises(ValueError, match="collaboration_mode"):
        MAGRPOConfig(collaboration_mode="joint")
    with pytest.raises(ValueError, match="num_turns=1"):
        MAGRPOConfig(collaboration_mode="centralized", num_turns=2)


def test_one_actor_training_uses_joint_prompt_and_task_rewards(tokenizer):
    trainer, prompts = make_centralized_magrpo_trainer(tokenizer)
    assert len(trainer.agents) == len(trainer.optimizers) == trainer.num_agents == 1
    assert trainer.args.num_agents == 2
    before = trainer.agents[0].transformer.wte.weight.detach().clone()
    trainer.train()
    assert prompts and set(prompts) == {"joint left right"}
    assert trainer.env_step == 2
    assert not torch.equal(before, trainer.agents[0].transformer.wte.weight)
    assert trainer.reward_func.reward_range == (0.0, 2.0)


def test_eval_passes_separate_roles_to_eval_logger(tokenizer):
    trainer, _ = make_centralized_magrpo_trainer(tokenizer)
    observed = []
    trainer.eval_logger = lambda **kwargs: observed.append(kwargs) or []
    trainer.eval_aggregator = lambda *a, **k: {"turn_1/domain_metric": 3.0}
    result = trainer.evaluate(num_eval_samples=1)
    assert set(observed[0]) == {"agent_completions_turns", "batch_items", "prompts"}
    assert observed[0]["agent_completions_turns"] == [[["good"]], [["good"]]]
    assert observed[0]["batch_items"] == [{"id": "task", "prompt": "task prompt"}]
    assert observed[0]["prompts"] == ["task prompt"]
    assert result["eval/turn_1/reward_mean"] == 2.0
    assert result["eval/turn_1/domain_metric"] == 3.0


def test_joint_actor_gets_one_reference_model(tokenizer):
    trainer, _ = make_centralized_magrpo_trainer(
        tokenizer, reference_kl_enabled=True, reference_devices=["cpu"]
    )
    assert len(trainer.reference_models) == 1
    trainer.train()
    assert trainer.env_step == 2


def test_rejects_multiple_actors(tokenizer):
    with pytest.raises(ValueError, match="exactly one actor"):
        MAGRPOTrainer(
            agents=[model_for(tokenizer), model_for(tokenizer)],
            centralized_adapter=JointAdapter(),
            args=MAGRPOConfig(
                collaboration_mode="centralized", num_turns=1, agent_devices="cpu"
            ),
        )


def test_wandb_does_not_claim_a_comparator(tokenizer, monkeypatch):
    import comlrl.trainers.reinforce.magrpo as module

    trainer, _ = make_centralized_magrpo_trainer(tokenizer)
    captured = {}
    monkeypatch.setattr(module.wandb, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(module.wandb, "define_metric", lambda *args, **kwargs: None)
    trainer.wandb_config = {"project": "test"}
    trainer._init_wandb()
    assert captured["config"]["algorithm"] == "MAGRPO"
    assert captured["config"]["num_roles"] == 2
    assert captured["config"]["num_actor_models"] == 1
    assert "comparator_generation_mode" not in captured["config"]


def test_matching_the_decentralized_loss_reuses_the_magrpo_objective(tokenizer):
    from comlrl.trainers.reinforce import MAGRPOTrainer

    trainer, _ = make_centralized_magrpo_trainer(
        tokenizer, match_decentralized_loss=True
    )
    model = trainer.agents[0]
    prompt = torch.tensor(tokenizer.encode("joint left right"))
    sequences = [torch.tensor(tokenizer.encode(text)) for text in (GOOD, BAD)]
    data = {"prompt_input_ids": prompt[None], "completion_input_ids": [sequences]}
    actual = trainer._compute_loss_with_gradients(model, data, [2.0, 0.0])
    expected = MAGRPOTrainer._compute_loss_with_gradients(
        trainer, model, data, [2.0, 0.0]
    )
    torch.testing.assert_close(actual, expected)
    trainer.args.match_decentralized_loss = False
    joint = trainer._compute_loss_with_gradients(model, data, [2.0, 0.0])
    # The joint loss also scores each sequence's last token; MAGRPO's does not.
    assert not torch.allclose(joint, actual)
