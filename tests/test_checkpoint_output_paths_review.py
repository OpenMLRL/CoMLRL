"""Ordinary and resumed iterative runs must agree on output path conventions."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from comlrl.trainers.preference import MADPOIterTrainer
from comlrl.trainers.preference.checkpointing import (
    _preference_resume_destinations,
    _resume_output_path,
)
from comlrl.trainers.preference.output_paths import resolve_output_path


def trainer_for_paths():
    trainer = object.__new__(MADPOIterTrainer)
    trainer.args = SimpleNamespace()
    trainer.wandb_config = None
    return trainer


@pytest.mark.parametrize("prefix", ["relative/../outputs", "~/outputs"])
@pytest.mark.parametrize("source", ["explicit", "output_dir", "section", "wandb_dir"])
def test_training_and_resume_select_the_same_literal_path(
    tmp_path, monkeypatch, prefix, source
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "different-home"))
    trainer = trainer_for_paths()
    if source == "explicit":
        trainer.args.preference_replay_dir = prefix
        trainer.wandb_config = {"output_dir": "ignored"}
        expected = (tmp_path / prefix).resolve()
    else:
        trainer.wandb_config = {
            "output_dir": {"output_dir": prefix},
            "section": {"config_sections": {"output": {"base_dir": prefix}}},
            "wandb_dir": {"dir": prefix},
        }[source]
        expected = (tmp_path / prefix / "preference_replay").resolve()
    before = dict(vars(trainer))
    resumed = _resume_output_path(
        trainer,
        "preference_replay",
        "_preference_replay_dir_path",
        "preference_replay_dir",
    )
    assert resumed == expected
    assert vars(trainer) == before
    assert list(tmp_path.iterdir()) == []
    assert trainer._preference_replay_dir() == str(expected)
    assert expected.is_dir()
    assert not (tmp_path / "different-home").exists()


@pytest.mark.parametrize("cached", ["~/cached", Path("relative-cached")])
def test_cached_path_is_verbatim_and_wins_over_every_other_source(
    tmp_path, monkeypatch, cached
):
    monkeypatch.chdir(tmp_path)
    trainer = trainer_for_paths()
    trainer._preference_replay_dir_path = cached
    trainer.args.preference_replay_dir = "ignored-explicit"
    trainer.args.policy_checkpoint_dir = "ignored-policy"
    trainer.wandb_config = {"output_dir": "ignored-wandb"}
    trainer._checkpoint_directory = "ignored-checkpoint"
    assert (
        resolve_output_path(
            "preference_replay",
            cached=cached,
            configured="ignored",
            wandb_config={"output_dir": "ignored"},
            default_base="ignored",
        )
        is cached
    )
    assert trainer._preference_replay_dir() is cached
    assert (
        _resume_output_path(
            trainer,
            "preference_replay",
            "_preference_replay_dir_path",
            "preference_replay_dir",
        )
        == (tmp_path / cached).resolve()
    )
    del trainer.args
    assert trainer._preference_replay_dir() is cached
    assert (
        _resume_output_path(
            trainer,
            "preference_replay",
            "_preference_replay_dir_path",
            "preference_replay_dir",
        )
        == (tmp_path / cached).resolve()
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "checkpoint,replay,policy,expected",
    [
        ("~/checkpoints", "other/replay", "other/policy", "home/checkpoints"),
        (None, "~/run/replay", "~/run/policy", "~/run"),
        (None, "single/replay", None, "single"),
        (None, "first/replay", "second/policy", "."),
        (None, None, None, "."),
    ],
)
def test_resume_additional_fallbacks_preserve_their_source_conventions(
    tmp_path, monkeypatch, checkpoint, replay, policy, expected
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    trainer = trainer_for_paths()
    trainer._checkpoint_directory = checkpoint
    trainer.args.preference_replay_dir = replay
    trainer.args.policy_checkpoint_dir = policy
    before = dict(vars(trainer))
    path = _resume_output_path(
        trainer, "reward_distributions", "_reward_distribution_dir_path"
    )
    assert path == (tmp_path / expected / "reward_distributions").resolve()
    assert vars(trainer) == before
    assert list(tmp_path.iterdir()) == []
    # Checkpoint-specific fallback does not affect ordinary training.
    assert trainer._reward_distribution_dir() == str(tmp_path / "reward_distributions")


def test_checkpoint_tilde_fallback_matches_actual_checkpoint_save(
    tmp_path, monkeypatch
):
    from test_checkpoint_state import TinyTrainer
    from comlrl.utils.checkpoint import STATE_FILENAME

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    trainer = TinyTrainer()
    with trainer._checkpoint_context(checkpoint_dir="~/checkpoints"):
        destination = _resume_output_path(
            trainer, "reward_distributions", "_reward_distribution_dir_path"
        )
        assert destination == tmp_path / "home/checkpoints/reward_distributions"
        assert not destination.parent.exists()
        trainer._checkpoint_commit({"epoch": 1})
        checkpoint = destination.parent / "checkpoint-1" / STATE_FILENAME
        assert checkpoint.is_file()
        assert not destination.exists()
    assert not (tmp_path / "~").exists()


@pytest.mark.parametrize(
    "wandb_config,expected",
    [
        (
            {
                "output_dir": "first",
                "config_sections": {"output": {"base_dir": "second"}},
                "dir": "third",
            },
            "first",
        ),
        (
            {"config_sections": {"output": {"base_dir": "second"}}, "dir": "third"},
            "second",
        ),
        ({"config_sections": ["ignored"], "dir": "third"}, "third"),
        ({"config_sections": {"output": []}, "dir": "third"}, "third"),
    ],
)
def test_wandb_source_priority_precedes_resume_fallback(
    tmp_path, monkeypatch, wandb_config, expected
):
    monkeypatch.chdir(tmp_path)
    trainer = trainer_for_paths()
    trainer.wandb_config = wandb_config
    trainer._checkpoint_directory = "ignored-checkpoint"
    trainer.args.policy_checkpoint_dir = "ignored-parent/policy"
    assert (
        _resume_output_path(
            trainer, "reward_distributions", "_reward_distribution_dir_path"
        )
        == tmp_path / expected / "reward_distributions"
    )
    assert list(tmp_path.iterdir()) == []


def test_resume_preflight_keeps_symlink_boundary_check_without_creating_outputs(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "source-checkpoint"
    checkpoint.mkdir()
    (tmp_path / "alias").symlink_to(checkpoint, target_is_directory=True)
    trainer = trainer_for_paths()
    trainer.args.preference_replay_dir = "alias/nested"
    before = dict(vars(trainer))
    with pytest.raises(ValueError, match="outside the source checkpoint"):
        _preference_resume_destinations(trainer, checkpoint)
    assert vars(trainer) == before
    assert list(checkpoint.iterdir()) == []
    assert not (tmp_path / "policy_checkpoints").exists()
