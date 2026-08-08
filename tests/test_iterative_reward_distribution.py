from types import SimpleNamespace

import numpy as np
import pytest

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


def test_reward_distribution_uses_declared_range_and_counts():
    trainer = object.__new__(MADPOIterTrainer)

    def reward_func(*_args, **_kwargs):
        return []

    reward_func.reward_range = (0.0, 3.0)
    trainer.reward_func = reward_func
    trainer._reward_distribution_range = None

    edges = trainer._reward_distribution_bin_edges()
    counts, returned_edges = trainer._reward_distribution_counts(
        [0.0, 0.5, 1.5, 3.0],
        edges,
    )

    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(3.0)
    assert len(edges) == 17
    assert np.array_equal(returned_edges, edges)
    assert np.sum(counts) == 4

    serialized = trainer._reward_distribution_json_series(
        [0.0, 0.5, 1.5, 3.0],
        edges,
    )
    assert sum(serialized["counts"]) == 4
    assert "density" not in serialized


def test_reward_distribution_requires_declared_reward_range():
    trainer = object.__new__(MADPOIterTrainer)
    trainer.reward_func = lambda *_args, **_kwargs: []
    trainer._reward_distribution_range = None

    with pytest.raises(ValueError, match="reward_func.reward_range"):
        trainer._reward_distribution_bin_edges()


def test_reward_distribution_charts_label_count_axes():
    edges = np.linspace(0.0, 3.0, 17)
    series = [("target", np.ones(16)), ("comparator", np.full(16, 0.5))]

    line = MADPOIterTrainer._reward_distribution_line_image(
        title="Candidate Reward Distribution",
        edges=edges,
        series=series,
    )
    bars = MADPOIterTrainer._reward_distribution_bar_image(
        title="Candidate Reward Distribution",
        edges=edges,
        series=series,
    )

    for chart in (line, bars):
        assert chart is not None
        assert 'width="240"' in chart.html
        assert 'height="80"' in chart.html
        assert ">reward</text>" in chart.html
        assert ">count</text>" in chart.html
