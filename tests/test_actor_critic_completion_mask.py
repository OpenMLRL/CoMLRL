from types import SimpleNamespace

import pytest
import torch

from comlrl.trainers.actor_critic import IACTrainer, MAACTrainer, RolloutSample


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(10.0).reshape(1, 5, 2) / 10)
        self.values = torch.nn.Parameter(torch.zeros(1, 5))

    def forward(self, input_ids, attention_mask, output_values=False):
        length = input_ids.size(1)
        return SimpleNamespace(
            logits=self.logits[:, :length],
            values=self.values[:, :length] if output_values else None,
        )


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer])
@pytest.mark.parametrize("mask_values", [[True, False, True], [False, False, False]])
def test_policy_mask_selects_response_tokens_and_gradients(trainer_cls, mask_values):
    trainer = trainer_cls.__new__(trainer_cls)
    model = TokenModel()
    ids = torch.tensor([[0, 1, 0, 1, 0]])
    with torch.inference_mode():
        mask = torch.tensor(mask_values)

    actual, _ = trainer._policy_eval(
        model,
        ids,
        torch.ones_like(ids),
        2,
        3,
        output_values=False,
        completion_loss_mask=mask,
    )
    selected = (
        model.logits[:, 1:4].log_softmax(-1).gather(-1, ids[:, 2:, None]).squeeze(-1)
    )
    expected = selected[:, torch.tensor(mask_values)].sum(-1)

    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(
        expected.sum(), model.logits, retain_graph=True
    )[0]
    actual.sum().backward()
    torch.testing.assert_close(model.logits.grad, expected_grad)


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer])
@pytest.mark.parametrize("mask", [torch.ones(2), torch.ones(1, 3), torch.ones(4)])
def test_policy_mask_rejects_misaligned_tokens(trainer_cls, mask):
    trainer = trainer_cls.__new__(trainer_cls)
    ids = torch.tensor([[0, 1, 0, 1, 0]])
    with pytest.raises(ValueError, match="completion_loss_mask"):
        trainer._policy_eval(
            TokenModel(),
            ids,
            torch.ones_like(ids),
            2,
            3,
            output_values=False,
            completion_loss_mask=mask,
        )


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer])
def test_ac_step_reads_mask_from_rollout_metadata(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    actor = TokenModel()
    critic = TokenModel()
    trainer.args = SimpleNamespace(
        use_separate_critic=False,
        value_loss_coef=0.6,
        value_clip_range=None,
    )
    trainer.critic_type = "v"
    trainer.agents = [actor]
    trainer.agent_devices = [torch.device("cpu")]
    trainer.critic_device = torch.device("cpu")
    trainer.agent_optimizers = [torch.optim.SGD(actor.parameters(), lr=0.1)]
    trainer.critics = [critic]
    trainer.critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
    ids = torch.tensor([0, 1, 0, 1, 0])
    sample = RolloutSample(
        agent_idx=0,
        prompt="prompt",
        completion="response",
        full_input_ids=ids,
        attention_mask=torch.ones_like(ids),
        prompt_len=2,
        response_len=3,
        old_logprob=torch.zeros(1),
        old_value=torch.zeros(1),
        reward=torch.ones(1),
        returns=torch.ones(1),
        advantage=torch.ones(1),
        normalized_advantage=torch.ones(1),
        metadata={
            "completion_loss_mask": torch.tensor([True, False, True]),
            "value_target": torch.ones(1),
            "joint_input_ids": ids[None],
            "joint_attention_mask": torch.ones_like(ids)[None],
            "joint_prompt_len": 2,
        },
    )

    trainer._ac_step(0, [sample])

    assert actor.logits.grad[:, [1, 3]].abs().sum() > 0
    assert actor.logits.grad[:, [0, 2, 4]].count_nonzero() == 0
