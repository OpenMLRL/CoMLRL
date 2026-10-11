from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from comlrl.trainers.reinforce import (
    CentralizedMAGRPOConfig,
    CentralizedMAGRPOTrainer,
    MAGRPOConfig,
)
from test_centralized_collaboration import (
    Adapter,
    BAD,
    GOOD,
    model_for,
    task_reward,
    tokenizer,
)


def make_trainer(tokenizer, **overrides):
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
        centralized_adapter=Adapter(),
    )
    trainer.verbose = False
    return trainer, prompts


def test_config_rejects_wrong_mode_and_multiple_turns():
    assert CentralizedMAGRPOConfig().num_turns == 1
    assert MAGRPOConfig().num_turns == 2
    with pytest.raises(ValueError, match="centralized collaboration"):
        CentralizedMAGRPOConfig(collaboration_mode="decentralized")
    with pytest.raises(ValueError, match="num_turns=1"):
        CentralizedMAGRPOConfig(num_turns=2)


def test_one_actor_training_uses_joint_prompt_and_task_rewards(tokenizer):
    trainer, prompts = make_trainer(tokenizer)
    assert len(trainer.agents) == len(trainer.optimizers) == trainer.num_agents == 1
    assert trainer.args.num_agents == 2
    before = trainer.agents[0].transformer.wte.weight.detach().clone()
    trainer.train()
    assert prompts and set(prompts) == {"joint left right"}
    assert trainer.env_step == 2
    assert not torch.equal(before, trainer.agents[0].transformer.wte.weight)
    assert trainer.reward_func.reward_range == (0.0, 2.0)
    assert not hasattr(trainer, "reward_model")
    assert not hasattr(trainer, "comparator_policy")


def test_eval_passes_separate_roles_to_existing_logger(tokenizer):
    trainer, _ = make_trainer(tokenizer)
    observed = []
    trainer.eval_logger = lambda **kwargs: observed.append(kwargs) or []
    trainer.eval_aggregator = lambda *a, **k: {"turn_1/domain_metric": 3.0}
    result = trainer.evaluate(num_eval_samples=1)
    assert observed[0]["agent_completions_turns"] == [[["good"]], [["good"]]]
    assert result["eval/turn_1/reward_mean"] == 2.0
    assert result["eval/turn_1/domain_metric"] == 3.0
    assert not trainer._centralized_eval_items


def test_loss_matches_full_joint_sequence_policy_gradient(tokenizer):
    trainer, _ = make_trainer(tokenizer)
    model = trainer.agents[0]
    prompt = torch.tensor(tokenizer.encode("joint left right"))
    sequences = [torch.tensor(tokenizer.encode(text)) for text in (GOOD, BAD)]
    data = {"prompt_input_ids": prompt[None], "completion_input_ids": [sequences]}
    manual = []
    for tokens, advantage in zip(sequences, (1.0, -1.0)):
        logits = model(torch.cat([prompt, tokens])[None], use_cache=False).logits[
            0, len(prompt) - 1 : -1
        ]
        manual.append(
            -logits.log_softmax(-1).gather(1, tokens[:, None]).sum() * advantage
        )
    expected = torch.stack(manual).mean()
    actual = trainer._compute_loss_with_gradients(model, data, [2.0, 0.0])
    torch.testing.assert_close(actual, expected)
    expected.backward()
    expected_grad = model.transformer.wte.weight.grad.clone()
    model.zero_grad()
    actual.backward()
    torch.testing.assert_close(model.transformer.wte.weight.grad, expected_grad)
    for token in ("<agent_0>", "<agent_1>", "</agent_1>"):
        assert expected_grad[tokenizer.convert_tokens_to_ids(token)].abs().sum() > 0


def test_one_reference_model_for_joint_actor(tokenizer):
    trainer, _ = make_trainer(
        tokenizer, reference_kl_enabled=True, reference_devices=["cpu"]
    )
    assert len(trainer.reference_models) == 1
    trainer.train()
    assert trainer.env_step == 2


def test_rejects_multiple_actors(tokenizer):
    with pytest.raises(ValueError, match="exactly one actor"):
        CentralizedMAGRPOTrainer(
            agents=[model_for(tokenizer), model_for(tokenizer)],
            centralized_adapter=Adapter(),
            args=CentralizedMAGRPOConfig(agent_devices="cpu"),
        )


def test_wandb_does_not_claim_a_comparator(tokenizer, monkeypatch):
    import comlrl.trainers.reinforce.magrpo as module

    trainer, _ = make_trainer(tokenizer)
    captured = {}
    monkeypatch.setattr(module.wandb, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(module.wandb, "define_metric", lambda *args, **kwargs: None)
    trainer.wandb_config = {"project": "test"}
    trainer._init_wandb()
    assert captured["config"]["algorithm"] == "MAGRPO"
    assert captured["config"]["num_roles"] == 2
    assert captured["config"]["num_actor_models"] == 1
    assert "comparator_generation_mode" not in captured["config"]
