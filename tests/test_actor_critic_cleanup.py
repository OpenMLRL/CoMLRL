import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from comlrl.trainers.actor_critic import IACTrainer, MAACTrainer, RolloutSample


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer])
@pytest.mark.parametrize(
    "num_agents,num_ret,rewards,expected",
    [
        (2, 3, [4.0], [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
        (2, 3, [1.0, 2.0, 3.0], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        (2, 3, [1.0, 2.0], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        (2, 2, [1.0, 2.0], [[1.0, 2.0], [1.0, 2.0]]),
    ],
)
def test_reward_broadcast_preserves_generation_precedence(
    trainer_cls, num_agents, num_ret, rewards, expected
):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(num_agents=num_agents)
    result = trainer._expand_rewards(rewards, num_ret)
    assert result == expected
    result[0][0] = -100.0
    assert result[1] == expected[1]


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer])
def test_reward_broadcast_rejects_unsupported_lengths(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(num_agents=2)
    with pytest.raises(ValueError, match="Reward function must return 1 value"):
        trainer._expand_rewards([1.0] * 4, 3)


def test_maac_single_turn_does_not_dispatch_multi_turn_reward_override():
    class CustomMAAC(MAACTrainer):
        def _expand_rewards(self, rewards, num_ret):
            raise AssertionError("This override only applies to multi-turn rollouts.")

    trainer = CustomMAAC.__new__(CustomMAAC)
    trainer.args = SimpleNamespace(
        num_agents=2,
        num_turns=1,
        num_generations=2,
        critic_type="v",
        reference_kl_enabled=False,
    )
    trainer.agents = [None, None]
    trainer._reward_signature = None
    trainer.reward_func = lambda *args, **kwargs: [1.0, 2.0]
    trainer.reward_processor = float
    trainer._resolve_turn_prompt = lambda item, agent_idx: f"agent {agent_idx}"
    trainer._get_tokenizer = lambda agent_idx: SimpleNamespace(
        decode=lambda *args, **kwargs: "completion"
    )
    trainer._generate = lambda *args: {
        "prompt_len": 1,
        "sequences": torch.tensor([[0, 1], [0, 2]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "response_lens": [1, 1],
        "reference_kls": [],
        "completions": ["one", "two"],
    }
    trainer._critic_value_from_text = lambda text: {
        "input_ids": torch.tensor([[0, 1]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "prompt_len": 2,
        "value": torch.tensor([0.1]),
    }
    trainer._policy_eval = lambda *args, **kwargs: (torch.tensor([-0.5]), None)

    samples = trainer._collect_rollouts({"prompt": "prompt"})

    assert [(s.agent_idx, s.reward.item()) for s in samples] == [
        (0, 1.0),
        (0, 2.0),
        (1, 1.0),
        (1, 2.0),
    ]


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer])
def test_sequence_logprob_excludes_prompt_and_trailing_tokens(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    probabilities = torch.tensor(
        [[[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1], [0.5, 0.5]]],
        dtype=torch.float64,
    )
    logits = probabilities.log().requires_grad_()
    sequences = torch.tensor([[0, 1, 0, 1, 0]])
    result = trainer._compute_sequence_stats(sequences, logits, 2, 2)
    assert result.item() == pytest.approx(math.log(0.3) + math.log(0.6))
    result.backward()
    assert torch.count_nonzero(logits.grad[:, [0, 3, 4], :]) == 0
    assert torch.count_nonzero(logits.grad[:, [1, 2], :]) == 4


@pytest.mark.parametrize("trainer_cls", [IACTrainer, MAACTrainer])
def test_policy_evaluation_keeps_overridable_helpers(trainer_cls):
    class CustomTrainer(trainer_cls):
        def _compute_sequence_stats(self, *args):
            return torch.tensor([7.0])

        def _value_on_prompt_only(self, *args):
            return torch.tensor([9.0])

    trainer = CustomTrainer.__new__(CustomTrainer)
    trainer.critic_type = "v"
    ids = torch.tensor([[0, 1]])
    model = lambda **kwargs: SimpleNamespace(logits=torch.zeros(1, 2, 2))
    logprob, value = trainer._policy_eval(
        model, ids, torch.ones_like(ids), 1, 1, output_values=True
    )
    assert logprob.item() == 7.0
    assert value.item() == 9.0


class _ConstantActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
        self.value = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.calls = []

    def forward(self, input_ids, attention_mask, output_values):
        self.calls.append((input_ids.shape[1], output_values))
        logits = torch.stack([self.policy, -self.policy]).expand(*input_ids.shape, 2)
        values = self.value.expand(input_ids.shape) if output_values else None
        return SimpleNamespace(logits=logits, values=values)


@pytest.mark.parametrize(
    "trainer_cls,separate,critic_type",
    [
        (IACTrainer, False, "v"),
        (IACTrainer, False, "q"),
        (IACTrainer, True, "v"),
        (IACTrainer, True, "q"),
        (MAACTrainer, True, "v"),
        (MAACTrainer, True, "q"),
    ],
)
def test_fixed_rollout_update_matches_analytic_gradients(
    trainer_cls, separate, critic_type
):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(
        use_separate_critic=separate, value_loss_coef=0.6, value_clip_range=0.2
    )
    trainer.critic_type = critic_type
    trainer.agent_devices = trainer.critic_devices = [torch.device("cpu")]
    trainer.critic_device = torch.device("cpu")
    actor = _ConstantActorCritic()
    critic = _ConstantActorCritic() if separate else actor
    trainer.agents = [actor]
    trainer.critics = [critic] if separate else []
    trainer.agent_optimizers = [torch.optim.SGD(actor.parameters(), lr=0.1)]
    trainer.critic_optimizers = [torch.optim.SGD(critic.parameters(), lr=0.1)]
    trainer.critic_optimizer = trainer.critic_optimizers[0]
    samples = []
    for ids, advantage, target in [([0, 0, 1], 0.5, 1.2), ([0, 1, 1], -0.25, 0.2)]:
        samples.append(
            RolloutSample(
                agent_idx=0,
                prompt="prompt",
                completion="completion",
                full_input_ids=torch.tensor(ids),
                attention_mask=torch.ones(3, dtype=torch.long),
                prompt_len=1,
                response_len=2,
                old_logprob=torch.tensor([-2.0], dtype=torch.float64),
                old_value=torch.tensor([0.35], dtype=torch.float64),
                reward=torch.tensor([target], dtype=torch.float64),
                returns=torch.tensor([target], dtype=torch.float64),
                advantage=torch.tensor([advantage], dtype=torch.float64),
                normalized_advantage=torch.tensor([advantage], dtype=torch.float64),
                metadata={
                    "value_target": torch.tensor([target], dtype=torch.float64),
                    "joint_input_ids": torch.tensor([[0, 1]]),
                    "joint_attention_mask": torch.ones(1, 2, dtype=torch.long),
                    "joint_prompt_len": 2,
                },
            )
        )

    rng_before = torch.get_rng_state().clone()
    trainer._ac_step(0, samples)

    policy_grad = (math.tanh(0.25) - 1.0 / (1.0 + math.exp(-0.5))) / 2
    value_grad = 0.6 * ((0.4 - 1.2) + (0.4 - 0.2))
    assert actor.policy.grad.item() == pytest.approx(policy_grad)
    assert critic.value.grad.item() == pytest.approx(value_grad)
    assert actor.policy.item() == pytest.approx(0.25 - 0.1 * policy_grad)
    assert critic.value.item() == pytest.approx(0.4 - 0.1 * value_grad)
    assert torch.equal(torch.get_rng_state(), rng_before)
    value_length = 2 if trainer_cls is MAACTrainer else (3 if critic_type == "q" else 1)
    if separate:
        assert actor.calls == [(3, False)] * 2
        assert critic.calls == [(value_length, True)] * 2
        assert actor.value.grad is None
        assert critic.policy.grad is None
    else:
        assert actor.calls == [(3, False), (value_length, True)] * 2
