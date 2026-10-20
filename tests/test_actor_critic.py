"""IAC and MAAC rewards, log-probabilities, updates, evaluation and logging."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import wandb

from comlrl.trainers.actor_critic import IACTrainer, MAACTrainer, RolloutSample
from comlrl.trainers.actor_critic.ac_base import ActorCriticTrainerBase
from comlrl.trainers.actor_critic.iac import IACConfig
from comlrl.trainers.actor_critic.maac import MAACConfig

from helpers import tiny_gpt2, zero_reward

TRAINERS = [IACTrainer, MAACTrainer]


class _TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.arange(10.0).reshape(1, 5, 2) / 10)
        self.values = nn.Parameter(torch.zeros(1, 5))

    def forward(self, input_ids, attention_mask, output_values=False):
        length = input_ids.size(1)
        return SimpleNamespace(
            logits=self.logits[:, :length],
            values=self.values[:, :length] if output_values else None,
        )


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
    "num_agents,num_ret,rewards,expected",
    [
        (2, 3, [4.0], [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
        (2, 3, [1.0, 2.0, 3.0], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        (2, 3, [1.0, 2.0], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        (2, 2, [1.0, 2.0], [[1.0, 2.0], [1.0, 2.0]]),
    ],
)
def test_expand_rewards_preserves_generation_precedence(
    num_agents, num_ret, rewards, expected
):
    trainer = ActorCriticTrainerBase.__new__(ActorCriticTrainerBase)
    trainer.args = SimpleNamespace(num_agents=num_agents)
    result = trainer._expand_rewards(rewards, num_ret)
    assert result == expected
    result[0][0] = -100.0
    assert result[1] == expected[1]


def test_expand_rewards_rejects_unsupported_lengths():
    trainer = ActorCriticTrainerBase.__new__(ActorCriticTrainerBase)
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


def test_sequence_stats_exclude_prompt_and_trailing_tokens():
    trainer = ActorCriticTrainerBase.__new__(ActorCriticTrainerBase)
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


@pytest.mark.parametrize("trainer_cls", TRAINERS)
def test_policy_eval_uses_overridable_helpers(trainer_cls):
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


@pytest.mark.parametrize("trainer_cls", TRAINERS)
@pytest.mark.parametrize("mask_values", [[True, False, True], [False, False, False]])
def test_policy_eval_mask_selects_response_tokens_and_gradients(
    trainer_cls, mask_values
):
    trainer = trainer_cls.__new__(trainer_cls)
    model = _TokenModel()
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


@pytest.mark.parametrize("trainer_cls", TRAINERS)
@pytest.mark.parametrize("mask", [torch.ones(2), torch.ones(1, 3), torch.ones(4)])
def test_policy_eval_rejects_misaligned_mask(trainer_cls, mask):
    trainer = trainer_cls.__new__(trainer_cls)
    ids = torch.tensor([[0, 1, 0, 1, 0]])
    with pytest.raises(ValueError, match="completion_loss_mask"):
        trainer._policy_eval(
            _TokenModel(),
            ids,
            torch.ones_like(ids),
            2,
            3,
            output_values=False,
            completion_loss_mask=mask,
        )


@pytest.mark.parametrize("trainer_cls", TRAINERS)
def test_ac_step_reads_mask_from_rollout_metadata(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    actor = _TokenModel()
    critic = _TokenModel()
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


@pytest.mark.parametrize("per_sample", [False, True])
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
def test_ac_step_matches_analytic_gradients(
    trainer_cls, separate, critic_type, per_sample
):
    """Per-sample backward gives the same gradients and update as the batch mean."""
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.args = SimpleNamespace(
        use_separate_critic=separate,
        value_loss_coef=0.6,
        value_clip_range=0.2,
        per_sample_backward=per_sample,
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


def test_sample_loss_shares_stay_within_the_batch_mean_range():
    """fp16: 2 * 59840 overflows, but 59840 / 4 * 2 equals the weighted batch mean."""
    trainer = ActorCriticTrainerBase.__new__(ActorCriticTrainerBase)
    trainer.args = SimpleNamespace(value_loss_coef=2.0)
    actor_share, value_share = trainer._sample_loss_shares(
        torch.tensor([1.0], dtype=torch.float16),
        torch.tensor([59840.0], dtype=torch.float16),
        batch_size=4,
    )
    assert actor_share.item() == 0.25
    assert value_share.item() == 29920.0


@pytest.mark.parametrize(
    "config_cls,fifth",
    [(MAACConfig, "value_loss_coef"), (IACConfig, "value_clip_range")],
)
def test_per_sample_backward_keeps_positional_config_arguments(config_cls, fifth):
    config = config_cls(5e-6, 5e-6, 8, 4, 0.1)
    assert getattr(config, fifth) == 0.1
    assert config.per_sample_backward is False


def test_log_metrics_logs_each_call_directly(monkeypatch):
    trainer = ActorCriticTrainerBase()
    trainer.wandb_initialized = True
    trainer.env_step = 5
    trainer.args = SimpleNamespace(logging_steps=1)
    trainer._last_train_log_step = -1
    logged = []
    monkeypatch.setattr(
        wandb, "log", lambda metrics, step: logged.append(dict(metrics))
    )

    trainer._log_metrics({"loss": 1.0})
    trainer._log_metrics({"reward": 2.0})
    assert logged == [{"loss": 1.0}, {"reward": 2.0}]


def test_evaluate_collects_rollouts_in_eval_mode():
    class EvalTrainer(ActorCriticTrainerBase):
        def __init__(self):
            self.eval_dataset = [{"id": 0}, {"id": 1}]
            self.args = SimpleNamespace(eval_batch_size=1, eval_num_samples=1)
            self.wandb_initialized = False
            self.env_step = 0
            self.verbose = False
            self.eval_flags = []

        def _collect_rollouts(self, item):
            self.eval_flags.append(bool(getattr(self, "_in_eval", False)))
            return [
                SimpleNamespace(
                    metadata={},
                    reward=torch.tensor([1.0]),
                    returns=torch.tensor([1.5]),
                    old_value=torch.tensor([0.5]),
                )
            ]

    trainer = EvalTrainer()
    metrics = trainer.evaluate()
    assert trainer.eval_flags == [True]
    assert trainer._in_eval is False
    assert "eval/turn_1/reward_mean" in metrics


@pytest.mark.parametrize("trainer_cls", TRAINERS)
def test_wandb_init_reports_actor_critic_config(
    trainer_cls, stub_tokenizer, monkeypatch
):
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id="run-id", project="project", entity="team")

    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(wandb, "init", init)
    devices = dict(num_agents=1, num_turns=1, agent_devices="cpu", critic_devices="cpu")
    if trainer_cls is IACTrainer:
        args, critics = IACConfig(use_separate_critic=False, **devices), None
    else:
        args, critics = MAACConfig(**devices), [tiny_gpt2()]
    trainer = trainer_cls(
        agents=[tiny_gpt2()],
        critics=critics,
        tokenizer=stub_tokenizer,
        reward_func=zero_reward,
        args=args,
        wandb_config={"project": "project", "entity": "team"},
    )
    assert trainer.wandb_initialized
    assert len(calls) == 1
    assert calls[0]["project"] == "project"
    assert calls[0]["entity"] == "team"
    assert calls[0]["config"]["algorithm"] == trainer.algorithm_name
    assert calls[0]["config"]["num_agents"] == 1
    assert calls[0]["config"]["reference_kl_enabled"] is False
