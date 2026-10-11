from types import SimpleNamespace

import pytest
import torch

from comlrl.trainers.preference import (
    MARLHFConfig,
    MARLHFIterConfig,
    MARLHFIterTrainer,
    MARLHFTrainer,
)
from comlrl.trainers.preference.marlhf import JointRewardModel
from test_centralized_collaboration import make_trainer, model_for, tokenizer


class TokenValueBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1)
        self.embedding = torch.nn.Embedding(10, 1)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(10, dtype=torch.float)[:, None])

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(hidden_states=[self.embedding(input_ids)])


def test_reward_pooling_uses_last_valid_position_for_either_padding_side():
    model = JointRewardModel(TokenValueBackbone())
    with torch.no_grad():
        model.reward_head.weight.fill_(1)
        model.reward_head.bias.zero_()
    ids = torch.tensor([[7, 8, 0], [0, 7, 8], [7, 0, 8], [7, 8, 9]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    scores = model(ids, mask)

    torch.testing.assert_close(scores, torch.tensor([8.0, 8.0, 8.0, 9.0]))
    scores.sum().backward()
    grad = model.backbone.embedding.weight.grad[:, 0]
    torch.testing.assert_close(grad[[8, 9]], torch.tensor([3.0, 1.0]))
    assert grad[:8].count_nonzero() == 0


@pytest.mark.parametrize("algorithm", ["iac", "maac"])
@pytest.mark.parametrize(
    "trainer_cls,config_cls",
    [(MARLHFTrainer, MARLHFConfig), (MARLHFIterTrainer, MARLHFIterConfig)],
)
def test_centralized_actor_critic_backends_train_local_models(
    algorithm, trainer_cls, config_cls, tokenizer, tmp_path
):
    trainer, prompts = make_trainer(
        trainer_cls,
        config_cls,
        tokenizer,
        tmp_path,
        rl_algorithm=algorithm,
        num_generations=1,
        reference_kl_enabled=True,
        reference_devices=["cpu"],
    )
    critic = model_for(tokenizer)
    trainer.critic_sources = [critic]
    original_generate = trainer.agents[0].generate

    def generate(**kwargs):
        result = original_generate(**kwargs)
        return result if kwargs.get("return_dict_in_generate") else result.sequences

    trainer.agents[0].generate = generate
    reward_models = []

    def init_reward_model():
        trainer.reward_model = JointRewardModel(model_for(tokenizer))
        trainer.reward_tokenizer = tokenizer
        trainer.reward_optimizer = torch.optim.AdamW(
            trainer.reward_model.parameters(), lr=1e-3
        )
        reward_models.append(
            (
                trainer.reward_model,
                trainer.reward_model.reward_head.weight.detach().clone(),
            )
        )

    trainer._init_reward_model = init_reward_model
    actor_before = trainer.agents[0].transformer.wte.weight.detach().clone()
    critic_before = critic.transformer.wte.weight.detach().clone()

    trainer.train()

    assert len(trainer.agents) == trainer.num_agents == 1
    assert trainer.args.num_agents == 2
    assert prompts and all(
        tokenizer.decode(ids[0]) == "joint left right" for ids in prompts
    )
    assert not torch.equal(actor_before, trainer.agents[0].transformer.wte.weight)
    assert not torch.equal(critic_before, critic.transformer.wte.weight)
    assert len(reward_models) == (2 if trainer_cls is MARLHFIterTrainer else 1)
    assert trainer.env_step == len(reward_models)
    assert all(
        not torch.equal(model.reward_head.weight, initial)
        for model, initial in reward_models
    )
    assert all(
        torch.isfinite(parameter).all() for parameter in trainer.agents[0].parameters()
    )
    assert trainer.evaluate(num_eval_samples=1)["eval/turn_1/reward_mean"] == 2.0
