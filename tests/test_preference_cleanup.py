from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from comlrl.trainers.preference import MADPOIterTrainer, MADPOTrainer
from comlrl.trainers.preference import iterative


OUTPUT_DIRECTORIES = [
    ("_preference_replay_dir", "preference_replay", "_preference_replay_dir_path"),
    (
        "_reward_distribution_dir",
        "reward_distributions",
        "_reward_distribution_dir_path",
    ),
    ("_policy_checkpoint_dir", "policy_checkpoints", "_policy_checkpoint_dir_path"),
]


@pytest.mark.parametrize("method,directory,cache_attribute", OUTPUT_DIRECTORIES)
@pytest.mark.parametrize(
    "wandb_config,job_id,base_dir",
    [
        (None, None, "."),
        ([], "42", "."),
        ({}, "42", "."),
        ({"output_dir": "run/job_42"}, "42", "run/job_42"),
        (
            {
                "output_dir": "run",
                "config_sections": {"output": {"base_dir": "section"}},
                "dir": "wandb",
            },
            "42",
            "run",
        ),
        ({"config_sections": {"output": {"base_dir": "section"}}}, None, "section"),
        (
            {"config_sections": {"output": {"base_dir": "section"}}, "dir": "wandb"},
            "42",
            "section",
        ),
        ({"config_sections": ["ignored"], "dir": "wandb"}, "42", "wandb"),
        ({"config_sections": {"output": []}, "dir": "wandb"}, None, "wandb"),
        ({"output_dir": "", "dir": "wandb"}, "", "wandb"),
    ],
)
def test_output_directory_precedence(
    method,
    directory,
    cache_attribute,
    wandb_config,
    job_id,
    base_dir,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    if job_id is not None:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
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
    monkeypatch.setenv("SLURM_JOB_ID", "42")
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


def test_replay_writer_keeps_subclass_directory_override(tmp_path):
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


@pytest.mark.parametrize("kind", ["line", "bar"])
def test_reward_charts_preserve_empty_series_behavior(kind, monkeypatch):
    monkeypatch.setattr(iterative.wandb, "Html", lambda html: html)
    render = getattr(MADPOIterTrainer, f"_reward_distribution_{kind}_image")
    edges = np.array([0.0, 1.0])
    assert render(title="Empty", edges=edges, series=[]) is None
    chart = render(title="Empty", edges=edges, series=[("target", np.array([]))])
    if kind == "bar":
        assert chart is None
    else:
        assert ">reward</text>" in chart
        assert "<polyline" not in chart


@pytest.mark.parametrize("kind", ["line", "bar"])
def test_reward_charts_preserve_geometry_and_escape_labels(kind, monkeypatch):
    monkeypatch.setattr(iterative.wandb, "Html", lambda html: html)
    render = getattr(MADPOIterTrainer, f"_reward_distribution_{kind}_image")
    chart = render(
        title='Rewards < 1 & "quoted"',
        edges=np.array([0.0, 0.5, 1.0]),
        series=[
            ("target", np.array([0, 4])),
            ('unknown <&"', np.array([2, 1])),
            ("unused", np.array([])),
        ],
    )
    assert "Rewards &lt; 1 &amp; &quot;quoted&quot;" in chart
    assert "unused" not in chart
    assert 'width="240" height="80"' in chart
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


def test_preference_update_keeps_metrics_and_agent_update_order():
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
