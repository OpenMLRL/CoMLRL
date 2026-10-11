"""Fixed CPU rollouts preserve MAGRPO's existing rewards and update semantics."""

from types import SimpleNamespace

import pytest
import torch

from comlrl.trainers.reinforce.magrpo import MAGRPOConfig, MAGRPOTrainer


class _Policy(torch.nn.Module):
    def __init__(self, agent_idx, forward_calls):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(40).reshape(5, 8) / 100)
        self.agent_idx = agent_idx
        self.forward_calls = forward_calls

    def forward(self, input_ids, attention_mask, use_cache):
        self.forward_calls.append((self.agent_idx, input_ids.tolist()))
        return SimpleNamespace(logits=self.logits[: input_ids.shape[1]][None])


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, policy, records):
        super().__init__(policy.parameters(), lr=0.01)
        self.policy = policy
        self.records = records

    def step(self, closure=None):
        record = self.records[-1]
        record["gradient"] = self.policy.logits.grad.detach().clone()
        result = super().step(closure)
        record["after"] = self.policy.logits.detach().clone()
        return result


class _ScriptedTrainer(MAGRPOTrainer):
    def __init__(self, num_agents, joint_mode):
        self.args = MAGRPOConfig(
            num_agents=num_agents,
            num_turns=2,
            num_generations=2,
            joint_mode=joint_mode,
            early_termination_threshold=None,
            rollout_buffer_size=1,
            train_batch_size=1,
            advantage_normalization=False,
            agent_devices="cpu",
        )
        self.num_agents = num_agents
        self.parallel_training = "none"
        self.advantage_mode = "mean"
        self.env_step = 0
        self.wandb_initialized = False
        self.rollout_buffers = [[] for _ in range(num_agents)]
        self.next_nodes = [0] * num_agents
        self.generations = []
        self.rewards = []
        self.transitions = []
        self.forward_calls = []
        self.records = []
        self.agents = [_Policy(i, self.forward_calls) for i in range(num_agents)]
        self.optimizers = [_RecordingSGD(agent, self.records) for agent in self.agents]
        self.reward_func = self._scripted_reward
        self.reward_processor = lambda reward: reward
        self.external_transition = self._transition

    def _generate_completions_with_external_prompts(
        self, agent, batch_items, agent_idx=0, external_prompts=None, **kwargs
    ):
        node = self.next_nodes[agent_idx]
        self.next_nodes[agent_idx] += 1
        self.generations.append((node, agent_idx, external_prompts))
        return {
            "prompts": [f"prompt-{node}"],
            "completions": [[f"{node}:{agent_idx}:{sample}" for sample in range(2)]],
            "prompt_input_ids": torch.tensor([[1, node + 1]]),
            "completion_input_ids": [
                [torch.tensor([sample + 2, agent_idx + 4, 7]) for sample in range(2)]
            ],
        }

    def _scripted_reward(self, *completions, batch_items=None, prompts=None):
        parsed = [tuple(map(int, values[0].split(":"))) for values in completions]
        reward = 10 * parsed[0][0] + sum(
            (agent + 1) * sample for _, agent, sample in parsed
        )
        self.rewards.append((tuple(item[2] for item in parsed), reward))
        assert batch_items == [{"prompt": "fixture"}]
        return [float(reward)]

    def _transition(self, *, agent_completions, **kwargs):
        self.transitions.append(tuple(agent_completions))
        return [f"next-{len(self.transitions)}"] * self.num_agents

    def _compute_loss_with_gradients(self, agent, completions_data, returns):
        loss = super()._compute_loss_with_gradients(agent, completions_data, returns)
        self.records.append(
            {
                "agent": agent.agent_idx,
                "node": int(completions_data["prompt_input_ids"][0, 1]) - 1,
                "step": self.env_step,
                "returns": list(returns),
                "loss": loss.detach().clone(),
                "before": agent.logits.detach().clone(),
            }
        )
        return loss


def _run_scenario(num_agents, joint_mode):
    trainer = _ScriptedTrainer(num_agents, joint_mode)
    assert trainer._train_step_returns({"prompt": "fixture"}) is None
    return trainer


@pytest.mark.parametrize("joint_mode", ["align", "aligned", "cross", "crossed"])
def test_two_turn_rewards_returns_and_update_order(joint_mode):
    trainer = _run_scenario(2, joint_mode)
    crossed = joint_mode in {"cross", "crossed"}
    combinations = [(0, 0), (0, 1), (1, 0), (1, 1)] if crossed else [(0, 0), (1, 1)]
    node_count = len(combinations) + 1
    assert trainer.env_step == len(combinations) * node_count
    assert trainer.generations == [
        (node, agent, f"next-{node}" if node else None)
        for node in range(node_count)
        for agent in range(2)
    ]
    assert trainer.transitions == [
        tuple(f"0:{agent}:{sample}" for agent, sample in enumerate(combination))
        for combination in combinations
    ]
    assert trainer.rewards == [
        (combination, 10 * node + combination[0] + 2 * combination[1])
        for node in range(node_count)
        for combination in combinations
    ]
    assert [(record["node"], record["agent"]) for record in trainer.records] == [
        (node, agent) for node in range(node_count) for agent in range(2)
    ]
    if joint_mode == "cross":
        expected = [
            [[35.0, 77.0], [44.0, 68.0]],
            *[
                [[20.0 * node + 2, 20.0 * node + 4], [20.0 * node + 1, 20.0 * node + 5]]
                for node in range(1, 5)
            ],
        ]
    elif joint_mode == "crossed":
        # Preserve the existing alias behavior: only "cross" aggregates returns.
        expected = [[[11.5, 23.5, 32.5, 44.5]] * 2] + [
            [[10.0 * node + offset for offset in (0, 2, 1, 3)]] * 2
            for node in range(1, 5)
        ]
    else:
        expected = [[[11.5, 24.5]] * 2, [[10.0, 13.0]] * 2, [[20.0, 23.0]] * 2]
    assert [record["returns"] for record in trainer.records] == [
        values for node in expected for values in node
    ]
    _assert_loss_and_gradient_regression(trainer)


def _assert_loss_and_gradient_regression(trainer):
    for record in trainer.records:
        weights = record["before"].clone().requires_grad_()
        returns = torch.tensor(record["returns"])
        advantages = returns - returns.mean()
        # Freeze the pre-cleanup token boundary; changing it changes the objective.
        log_probs = weights[1:3].log_softmax(-1)
        losses = [
            -(log_probs[0, sample + 2] + log_probs[1, record["agent"] + 4]) * advantage
            for sample, advantage in enumerate(advantages[:2])
        ]
        expected_loss = torch.stack(losses).mean()
        (expected_gradient,) = torch.autograd.grad(expected_loss, weights)
        torch.testing.assert_close(record["loss"], expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(
            record["gradient"], expected_gradient, rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            record["after"],
            record["before"] - 0.01 * expected_gradient,
            rtol=0,
            atol=1e-7,
        )
    assert trainer.forward_calls == [
        (record["agent"], [[1, record["node"] + 1, sample + 2, record["agent"] + 4]])
        for record in trainer.records
        for sample in range(2)
    ]


@pytest.mark.parametrize("joint_mode", ["aligned", "cross", "crossed", "unknown"])
def test_single_agent_retains_joint_mode_fallback(joint_mode):
    trainer = _run_scenario(1, joint_mode)
    assert trainer.env_step == 6
    assert [record["returns"] for record in trainer.records] == [
        [10.5, 21.5],
        [10.0, 11.0],
        [20.0, 21.0],
    ]
    assert [(record["node"], record["agent"]) for record in trainer.records] == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]
    _assert_loss_and_gradient_regression(trainer)


def test_multiple_agents_still_reject_unknown_joint_mode():
    trainer = _ScriptedTrainer(2, "unknown")
    with pytest.raises(ValueError, match="Unsupported joint_mode: unknown"):
        trainer._train_step_returns({"prompt": "fixture"})
    assert trainer.env_step == 0
    assert trainer.rewards == trainer.records == []


def test_reward_normalization_preserves_mutation_and_list_identity():
    trainer = _ScriptedTrainer(2, "aligned")
    second = ["0:1:1"]
    completions = ["0:0:0", second]
    assert trainer._compute_rewards(
        ["fixture"], completions, batch_items=[{"prompt": "fixture"}]
    ) == [2.0]
    assert completions == [["0:0:0"], ["0:1:1"]]
    assert completions[1] is second
