"""Decentralized MADPO/MADPOIter updates, rewards, charts and output paths."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from comlrl.trainers.pbrl import (
    MADPOConfig,
    MADPOIterTrainer,
    MADPOTrainer,
    iterative,
)

from helpers import model_for

OUTPUT_DIRECTORIES = [
    ("_preference_replay_dir", "preference_replay", "_preference_replay_dir_path"),
    (
        "_reward_distribution_dir",
        "reward_distributions",
        "_reward_distribution_dir_path",
    ),
    ("_policy_checkpoint_dir", "policy_checkpoints", "_policy_checkpoint_dir_path"),
]


def _policy_outputs(prefix):
    return [
        {
            "completions": [[f"{prefix}-agent-{agent_idx}"]],
            "prompts": [f"prompt-{agent_idx}"],
        }
        for agent_idx in range(2)
    ]


@pytest.fixture
def render_chart(monkeypatch):
    """Render reward charts with W&B's HTML wrapper reduced to its payload."""
    monkeypatch.setattr(
        iterative.wandb, "Html", lambda html: SimpleNamespace(html=html)
    )
    return MADPOIterTrainer._reward_distribution_image


@pytest.mark.parametrize("source", ["inference", "generation"])
def test_sequence_log_prob_accepts_inference_tensors(tokenizer, source):
    torch.manual_seed(19)
    model = model_for(tokenizer)
    trainer = MADPOTrainer(
        agents=[model],
        num_agents=1,
        tokenizer=tokenizer,
        args=MADPOConfig(num_agents=1, agent_devices="cpu"),
        formatters=lambda _: "joint left right",
        reward_func=lambda completions: [0.0],
    )
    if source == "generation":
        data = trainer._generate_completions(
            model,
            [{}],
            num_return_sequences=1,
            max_new_tokens=4,
            min_new_tokens=4,
        )
        prompt = data["prompt_input_ids"][0]
        completion = data["completion_input_ids"][0][0]
    else:
        with torch.inference_mode():
            prompt = torch.tensor(tokenizer.encode("joint left right"))
            completion = torch.tensor(tokenizer.encode("good bad good bad"))
    assert completion.is_inference()

    normal_prompt = prompt.clone()
    normal_completion = completion.clone()
    expected = trainer._sequence_log_prob(0, normal_prompt, normal_completion)
    expected_gradients = torch.autograd.grad(expected, tuple(model.parameters()))
    actual = trainer._sequence_log_prob(0, prompt, completion)
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(prompt, normal_prompt, rtol=0, atol=0)
    torch.testing.assert_close(completion, normal_completion, rtol=0, atol=0)


def test_preference_update_metrics_and_agent_order():
    trainer = object.__new__(MADPOTrainer)
    trainer.num_agents = 2
    events = []
    trainer.optimizers = [
        SimpleNamespace(
            zero_grad=lambda idx=idx: events.append(("zero_grad", idx)),
            step=lambda idx=idx: events.append(("step", idx)),
        )
        for idx in range(2)
    ]
    trainer._detached_agent_deltas = lambda batch: events.append(("deltas", len(batch)))

    def loss(agent_idx, batch, deltas):
        events.append(("loss", agent_idx))
        return SimpleNamespace(backward=lambda: events.append(("backward", agent_idx)))

    trainer._agent_preference_loss = loss
    batch = [SimpleNamespace(candidate_reward_mean=value) for value in [1.0, 2.0, 6.0]]
    assert trainer._update_from_preference_batch(batch) == {
        "turn_1/reward_mean": 3.0,
        "turn_1/expected_return": 3.0,
    }
    assert events == [
        ("deltas", 3),
        ("zero_grad", 0),
        ("loss", 0),
        ("backward", 0),
        ("step", 0),
        ("zero_grad", 1),
        ("loss", 1),
        ("backward", 1),
        ("step", 1),
    ]
    events.clear()
    assert trainer._update_from_preference_batch([]) == {}
    assert events == []


def test_preference_pairs_and_distribution_use_raw_rewards():
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


@pytest.mark.parametrize(
    "value, label", [(2.0, "2"), (1.5, "1.5"), (0.25, "0.25"), (0.05, "0.050")]
)
def test_reward_chart_axis_tick_labels(value, label):
    assert MADPOIterTrainer._format_axis_tick(value) == label


@pytest.mark.parametrize("kind", ["line", "bar"])
def test_reward_chart_empty_series(kind, render_chart):
    edges = np.array([0.0, 1.0])
    assert render_chart(title="Empty", edges=edges, series=[], kind=kind) is None
    chart = render_chart(
        title="Empty", edges=edges, series=[("target", np.array([]))], kind=kind
    )
    if kind == "bar":
        assert chart is None
    else:
        assert ">reward</text>" in chart.html
        assert "<polyline" not in chart.html


@pytest.mark.parametrize("kind", ["line", "bar"])
def test_reward_chart_geometry_labels_and_escaping(kind, render_chart):
    chart = render_chart(
        kind=kind,
        title='Rewards < 1 & "quoted"',
        edges=np.array([0.0, 0.5, 1.0]),
        series=[
            ("target", np.array([0, 4])),
            ('unknown <&"', np.array([2, 1])),
            ("unused", np.array([])),
        ],
    ).html
    assert "Rewards &lt; 1 &amp; &quot;quoted&quot;" in chart
    assert "unused" not in chart
    assert 'width="240" height="80"' in chart
    assert ">reward</text>" in chart
    assert ">count</text>" in chart
    assert "#e87030" in chart
    assert "#2878d6" in chart
    svg = ET.fromstring(chart)[0]
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    if kind == "line":
        lines = svg.findall("svg:polyline", namespace)
        assert [line.attrib["points"] for line in lines] == [
            "79,62 180,14",
            "79,38 180,50",
        ]
        assert chart.index('stroke="#505050"') < chart.index("<polyline")
    else:
        bars = [
            rect.attrib
            for rect in svg.findall("svg:rect", namespace)
            if "opacity" in rect.attrib and "height" in rect.attrib
        ][:4]
        assert [bar["height"] for bar in bars] == ["0.0", "24.0", "48.0", "12.0"]
        assert chart.index('opacity="0.82"') < chart.index('stroke="#505050"')


@pytest.mark.parametrize("method,directory,cache_attribute", OUTPUT_DIRECTORIES)
@pytest.mark.parametrize(
    "wandb_config,base_dir",
    [
        (None, "."),
        ([], "."),
        ({}, "."),
        ({"output_dir": "runs/42"}, "runs/42"),
        (
            {
                "output_dir": "run",
                "config_sections": {"output": {"base_dir": "section"}},
                "dir": "wandb",
            },
            "run",
        ),
        ({"config_sections": {"output": {"base_dir": "section"}}}, "section"),
        (
            {"config_sections": {"output": {"base_dir": "section"}}, "dir": "wandb"},
            "section",
        ),
        ({"config_sections": ["ignored"], "dir": "wandb"}, "wandb"),
        ({"config_sections": {"output": []}, "dir": "wandb"}, "wandb"),
        ({"output_dir": "", "dir": "wandb"}, "wandb"),
    ],
)
def test_output_directories_follow_config_precedence(
    method,
    directory,
    cache_attribute,
    wandb_config,
    base_dir,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    trainer = object.__new__(MADPOIterTrainer)
    trainer.args = SimpleNamespace()
    trainer.wandb_config = wandb_config

    path = getattr(trainer, method)()

    assert path == str((tmp_path / base_dir / directory).resolve())
    assert Path(path).is_dir()
    assert getattr(trainer, cache_attribute) == path


@pytest.mark.parametrize("method,directory,cache_attribute", OUTPUT_DIRECTORIES)
def test_explicit_output_directories_and_independent_caches(
    method, directory, cache_attribute, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    trainer = object.__new__(MADPOIterTrainer)
    trainer.args = SimpleNamespace(
        preference_replay_dir="explicit-replay",
        policy_checkpoint_dir="explicit-checkpoint",
        reward_distribution_dir="ignored",
    )
    trainer.wandb_config = {"output_dir": "run"}
    expected = {
        "_preference_replay_dir": tmp_path / "explicit-replay",
        "_policy_checkpoint_dir": tmp_path / "explicit-checkpoint",
        "_reward_distribution_dir": tmp_path / "run" / directory,
    }[method]

    assert getattr(trainer, method)() == str(expected)
    trainer.args = None
    trainer.wandb_config = None
    monkeypatch.setattr(iterative.os, "makedirs", Mock(side_effect=AssertionError))
    assert getattr(trainer, method)() == str(expected)
    assert getattr(trainer, cache_attribute) == str(expected)
    assert all(
        not hasattr(trainer, other_cache)
        for _, _, other_cache in OUTPUT_DIRECTORIES
        if other_cache != cache_attribute
    )


def test_replay_writer_uses_subclass_directory_override(tmp_path):
    class ReplayTrainer(MADPOIterTrainer):
        def _preference_replay_dir(self):
            path = Path(super()._preference_replay_dir()) / "custom"
            path.mkdir()
            return str(path)

    trainer = object.__new__(ReplayTrainer)
    trainer.args = SimpleNamespace(preference_replay_dir=tmp_path)
    trainer.wandb_config = None
    assert trainer._write_iteration_preference_pairs(0, []) is None
    assert (tmp_path / "custom" / "iteration_0001.json").is_file()
