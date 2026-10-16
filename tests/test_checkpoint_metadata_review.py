"""Resume metadata is available before eager W&B initialization (review C)."""

import copy
from functools import wraps
from pathlib import Path

import pytest
import torch
import wandb

from comlrl.utils import read_checkpoint_metadata
import comlrl.utils.checkpoint as checkpoint
from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer
from test_centralized_collaboration import model_for, tokenizer
from test_checkpoint_core_review import FakeRun
from test_checkpoint_preference import pair
from test_checkpoint_state import TinyTrainer, assert_nested_equal


def write_state(path, state):
    path.mkdir()
    torch.save(checkpoint._encode(state), path / checkpoint.STATE_FILENAME)
    return path


@pytest.mark.parametrize("file_path", [False, True])
def test_metadata_reads_only_public_fields_without_restoring(
    tmp_path, monkeypatch, file_path
):
    trainer = TinyTrainer()
    trainer._checkpoint_run_id = "a" * 32
    trainer._checkpoint_sequence = 3
    trainer._checkpoint_record_wandb_run(FakeRun("saved-id"))
    trainer._checkpoint_progress = {"phase": "train", "preference_pairs": [pair(1)]}
    path = Path(trainer.save_checkpoint(tmp_path / "checkpoint-3"))
    state_path = path / checkpoint.STATE_FILENAME
    before = state_path.read_bytes()
    rng = checkpoint.capture_rng_state()
    original_load = torch.load
    calls = []

    @wraps(original_load)
    def load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(
        wandb, "init", lambda **kwargs: pytest.fail("metadata initialized W&B")
    )
    # No training records, callbacks or model classes need decoding/importing.
    monkeypatch.setattr(
        checkpoint.importlib,
        "import_module",
        lambda *a: pytest.fail("decoded training record"),
    )
    metadata = read_checkpoint_metadata(state_path if file_path else path)
    assert metadata == {
        "version": checkpoint.CHECKPOINT_VERSION,
        "trainer_class": "test_checkpoint_state.TinyTrainer",
        "sequence": 3,
        "run_id": "a" * 32,
        "wandb": {"id": "saved-id", "project": "research", "entity": "team"},
    }
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    assert state_path.read_bytes() == before
    assert_nested_equal(checkpoint.capture_rng_state(), rng)


def test_legacy_metadata_without_logging_or_run_identity(tmp_path):
    state = checkpoint.capture_trainer_state(TinyTrainer())
    state.pop("wandb")
    state.pop("run_id")
    metadata = read_checkpoint_metadata(write_state(tmp_path / "legacy", state))
    assert metadata["wandb"] is None
    assert metadata["run_id"] is None


def test_metadata_supports_torch_without_mmap(tmp_path, monkeypatch):
    path = TinyTrainer().save_checkpoint(tmp_path / "saved")
    original_load = torch.load
    calls = []

    def load(path, *, map_location, weights_only):
        calls.append((map_location, weights_only))
        return original_load(path, map_location=map_location, weights_only=weights_only)

    monkeypatch.setattr(torch, "load", load)
    assert read_checkpoint_metadata(path)["sequence"] == 0
    assert calls == [("cpu", True)]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("version", 99, "version"),
        ("sequence", -1, "sequence"),
        ("sequence", True, "sequence"),
        ("trainer_class", None, "trainer class"),
        ("run_id", "invalid", "run identity"),
        ("wandb", {"id": ""}, "W&B"),
        ("wandb", {"id": "saved", "entity": 10}, "W&B"),
        ("wandb", {"id": "saved", "secret": "unexpected"}, "W&B"),
    ],
)
def test_invalid_metadata_rejected(tmp_path, field, value, match):
    state = checkpoint.capture_trainer_state(TinyTrainer())
    state[field] = value
    with pytest.raises(ValueError, match=match):
        read_checkpoint_metadata(write_state(tmp_path / "bad", state))


def test_metadata_rejects_missing_required_field(tmp_path):
    state = checkpoint.capture_trainer_state(TinyTrainer())
    state.pop("sequence")
    with pytest.raises(ValueError, match="Incomplete"):
        read_checkpoint_metadata(write_state(tmp_path / "incomplete", state))


@pytest.mark.parametrize("override", [None, {"id": "caller-id", "resume": "allow"}])
def test_preconstruction_metadata_opens_only_the_intended_run(
    tmp_path, tokenizer, monkeypatch, override
):
    calls, runs = [], []

    def init(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        run = FakeRun(kwargs.get("id", "initial-id"))
        runs.append(run)
        monkeypatch.setattr(wandb, "run", run)
        return run

    monkeypatch.setattr(wandb, "run", None)
    monkeypatch.setattr(wandb, "init", init)

    def make(config):
        return MAGRPOTrainer(
            agents=[model_for(tokenizer)],
            num_agents=1,
            tokenizer=tokenizer,
            train_dataset=[{"prompt": "left"}],
            reward_func=lambda *a, **k: [1.0],
            args=MAGRPOConfig(num_agents=1, num_turns=1, agent_devices="cpu"),
            wandb_config=config,
        )

    original = make({"project": "research", "entity": "team"})
    path = original.save_checkpoint(tmp_path / "saved")
    runs[0].finish()
    monkeypatch.setattr(wandb, "run", None)
    calls.clear()
    identity = read_checkpoint_metadata(path)["wandb"]
    config = {**identity, "resume": "allow", **(override or {})}
    resumed = make(config)
    assert len(calls) == 1  # Eager constructor reuses the saved identity directly.
    assert calls[0]["id"] == (override or {}).get("id", "initial-id")
    assert calls[0]["resume"] == "allow"
    resumed.load_checkpoint(path)
    resumed._checkpoint_init_wandb()
    assert len(calls) == 1  # Restore does not finish/reopen a second run.
    assert not runs[-1].finished
    assert resumed.wandb_initialized
    assert_nested_equal(resumed.agents[0].state_dict(), original.agents[0].state_dict())
