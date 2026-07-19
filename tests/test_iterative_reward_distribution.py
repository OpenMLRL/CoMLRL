from types import SimpleNamespace

from comlrl.trainers.preference import MADPOIterTrainer


def _policy_outputs(prefix):
    return [
        {
            "completions": [[f"{prefix}-agent-{agent_idx}"]],
            "prompts": [f"prompt-{agent_idx}"],
        }
        for agent_idx in range(2)
    ]


def test_preference_distribution_and_pair_metadata_use_raw_rewards():
    trainer = object.__new__(MADPOIterTrainer)
    trainer.num_agents = 2
    trainer.agents = [object(), object()]
    trainer.args = SimpleNamespace(
        preference_num_candidates=1,
        comparator_num_candidates=None,
        joint_mode="aligned",
    )

    current_outputs = _policy_outputs("current")
    comparator_outputs = _policy_outputs("comparator")
    trainer._generate_policy_outputs_for_item = (
        lambda *_args, **_kwargs: current_outputs
    )
    trainer._generate_comparator_outputs_for_item = (
        lambda *_args, **_kwargs: comparator_outputs
    )

    reward_results = iter(
        [
            ([3.0], [-1.0]),
            ([2.0], [-2.0]),
        ]
    )
    trainer._compute_raw_and_processed_rewards = lambda *_args, **_kwargs: next(
        reward_results
    )

    recorded = {}
    trainer._record_iteration_reward_distribution = lambda **kwargs: recorded.update(
        kwargs
    )
    trainer._select_policy_comparison_pairs = lambda *_args: [
        (("current", 0), ("comparator", 0))
    ]
    trainer._preference_tensors_from_text = lambda *_args, **_kwargs: SimpleNamespace()

    pairs = trainer._generate_preference_pairs_for_item(
        {"prompt": "test"},
        iteration_idx=0,
    )

    assert recorded == {
        "target_rewards": [3.0],
        "comparator_rewards": [2.0],
    }
    assert len(pairs) == 1
    assert pairs[0].winner_reward == -1.0
    assert pairs[0].loser_reward == -2.0
    assert pairs[0].raw_rewards == [3.0, 2.0]
    assert pairs[0].target_raw_reward == 3.0
    assert pairs[0].comparator_raw_reward == 2.0
