"""Atomic checkpoint files, retention and output-directory ownership checks."""

import json
import os
from pathlib import Path
import pickle
import random

import numpy as np
import pytest
import torch

import comlrl.utils.checkpoint as checkpoint
from comlrl.utils.checkpoint import capture_trainer_state

from checkpoint_helpers import TinyTrainer, advance_models, assert_models_equal
from helpers import assert_nested_equal


class UnexpectedPayload:
    pass


def _save_run(directory, count=1):
    trainer = TinyTrainer()
    with trainer._checkpoint_context(checkpoint_dir=directory):
        for sequence in range(1, count + 1):
            trainer._checkpoint_commit({"epoch": sequence})
    return trainer


def _snapshot(root, sequence, *, run_id="f" * 32, kind="owned"):
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"checkpoint-{sequence}"
    if kind == "file":
        target.write_bytes(b"unrelated file")
        return target
    if kind == "symlink":
        target.symlink_to(root / "missing-target", target_is_directory=True)
        return target
    target.mkdir()
    (target / checkpoint.STATE_FILENAME).write_bytes(b"existing snapshot")
    if kind != "legacy":
        (target / checkpoint._OWNER_FILENAME).write_text(
            json.dumps(
                {"run_id": run_id, "sequence": sequence + (kind == "bad-sequence")}
            )
        )
    return target


def _untouched_trainer():
    trainer = TinyTrainer()
    trainer._checkpoint_directory = "original-output"
    trainer.load_checkpoint = lambda path: pytest.fail(
        "restore ran before directory preflight"
    )
    trainer._checkpoint_prepare_training = lambda: pytest.fail(
        "preparation ran before directory preflight"
    )
    return trainer, dict(vars(trainer)), capture_trainer_state(trainer)


def _assert_untouched(trainer, attributes, state):
    assert set(vars(trainer)) == set(attributes)
    assert all(vars(trainer)[name] is value for name, value in attributes.items())
    assert_nested_equal(capture_trainer_state(trainer), state)


def test_public_save_preserves_rng_and_load_restores_state(tmp_path):
    source = TinyTrainer()
    advance_models(source)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    source.save_checkpoint(tmp_path / "checkpoint")

    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert_nested_equal(after, before)
    expected_random = (random.random(), np.random.rand(4), torch.rand(4))
    target = TinyTrainer()
    target.load_checkpoint(tmp_path / "checkpoint")
    assert_models_equal(target, source)
    assert target._checkpoint_progress == source._checkpoint_progress
    actual_random = (random.random(), np.random.rand(4), torch.rand(4))
    assert_nested_equal(actual_random, expected_random)


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_failed_checkpoint_write_keeps_existing_checkpoint_loadable(
    tmp_path, monkeypatch, failure
):
    source = TinyTrainer()
    advance_models(source)
    checkpoint = tmp_path / "checkpoint"
    source.save_checkpoint(checkpoint)
    saved_files = {
        path.relative_to(checkpoint): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    expected = TinyTrainer()
    expected.load_checkpoint(checkpoint)
    advance_models(source)

    def fail_save(*args, **kwargs):
        raise OSError("simulated storage failure")

    original_replace = os.replace
    publish_failed = False

    def fail_publish(source_path, target_path):
        nonlocal publish_failed
        if Path(target_path) == checkpoint and not publish_failed:
            publish_failed = True
            raise OSError("simulated storage failure")
        return original_replace(source_path, target_path)

    with monkeypatch.context() as patch:
        if failure == "write":
            patch.setattr(torch, "save", fail_save)
        else:
            patch.setattr(os, "replace", fail_publish)
        with pytest.raises(OSError, match="simulated storage failure"):
            source.save_checkpoint(checkpoint)

    assert {
        path.relative_to(checkpoint): path.read_bytes()
        for path in checkpoint.rglob("*")
        if path.is_file()
    } == saved_files
    restored = TinyTrainer()
    restored.load_checkpoint(checkpoint)
    assert_models_equal(restored, expected)


def test_all_assets_fsynced_before_publish_and_parent_after(tmp_path, monkeypatch):
    trainer = TinyTrainer()
    events = []
    inode_names = {}

    def assets(root):
        asset = root / "assets" / "policy" / "model.safetensors"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"weights")
        for path in [asset, asset.parent, asset.parent.parent, root]:
            inode_names[path.stat().st_ino] = path.name
        return {}

    trainer._checkpoint_save_assets = assets
    original_fsync, original_replace = os.fsync, os.replace

    def fsync(fd):
        info = os.fstat(fd)
        events.append(("sync", inode_names.get(info.st_ino, info.st_ino)))
        original_fsync(fd)

    def replace(source, destination):
        events.append(("rename", Path(destination).name))
        return original_replace(source, destination)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    trainer.save_checkpoint(tmp_path / "checkpoint")
    published = events.index(("rename", "checkpoint"))
    for name in ["model.safetensors", "policy", "assets"]:
        assert ("sync", name) in events[:published]
    assert events[-1] == ("sync", tmp_path.stat().st_ino)


def test_asset_sync_failure_does_not_replace_previous_checkpoint(tmp_path, monkeypatch):
    trainer = TinyTrainer()
    destination = tmp_path / "checkpoint"
    trainer.save_checkpoint(destination)
    previous = (destination / checkpoint.STATE_FILENAME).read_bytes()
    monkeypatch.setattr(
        checkpoint,
        "_fsync_tree",
        lambda root: (_ for _ in ()).throw(OSError("sync failed")),
    )
    trainer.env_step += 1
    with pytest.raises(OSError, match="sync failed"):
        trainer.save_checkpoint(destination)
    assert (destination / checkpoint.STATE_FILENAME).read_bytes() == previous
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_public_load_rejects_unallowlisted_pickle_without_mutation(tmp_path):
    target = TinyTrainer()
    before = capture_trainer_state(target)
    path = tmp_path / "unsafe.pt"
    torch.save(UnexpectedPayload(), path)

    with pytest.raises(pickle.UnpicklingError):
        target.load_checkpoint(path)

    assert_nested_equal(capture_trainer_state(target), before)


def test_public_load_rejects_unallowlisted_record_without_mutation(tmp_path):
    target = TinyTrainer()
    before = capture_trainer_state(target)
    path = tmp_path / "unknown-record.pt"
    torch.save(
        {
            "__kind__": "record",
            "class": "test_checkpoint_files.UnexpectedPayload",
            "fields": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="Unsupported checkpoint record"):
        target.load_checkpoint(path)

    assert_nested_equal(capture_trainer_state(target), before)


def test_retention_only_prunes_owned_snapshots_after_success(tmp_path):
    unrelated = tmp_path / "checkpoint-900"
    malformed = tmp_path / "checkpoint-901"
    trainer = TinyTrainer()
    with trainer._checkpoint_context(checkpoint_dir=tmp_path, checkpoint_total_limit=2):
        unrelated.mkdir()
        (unrelated / "keep.txt").write_text("not a training checkpoint")
        malformed.mkdir()
        (malformed / checkpoint._OWNER_FILENAME).write_text("[]")
        for epoch in range(4):
            trainer._checkpoint_commit({"epoch": epoch + 1})
    assert sorted(
        path.name
        for path in tmp_path.glob("checkpoint-*")
        if (path / checkpoint.STATE_FILENAME).exists()
    ) == ["checkpoint-3", "checkpoint-4"]
    assert (unrelated / "keep.txt").exists()
    assert malformed.exists()
    # A subsequent resume also preflights future names; keep these unrelated
    # directories outside the automatic snapshot namespace before resuming.
    unrelated.rename(tmp_path / "unrelated")
    malformed.rename(tmp_path / "malformed")
    resumed = TinyTrainer()
    with resumed._checkpoint_context(
        resume_from_checkpoint=tmp_path / "checkpoint-4",
        checkpoint_dir=tmp_path,
        checkpoint_total_limit=1,
    ):
        resumed._checkpoint_commit({"epoch": 5})
    assert (tmp_path / "checkpoint-5" / checkpoint.STATE_FILENAME).exists()
    assert not (tmp_path / "checkpoint-4").exists()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_retention_limit_rejected_before_training(tmp_path, limit):
    trainer = TinyTrainer()
    before = capture_trainer_state(trainer)
    with pytest.raises(ValueError, match="checkpoint_total_limit"):
        with trainer._checkpoint_context(
            checkpoint_dir=tmp_path, checkpoint_total_limit=limit
        ):
            pytest.fail("invalid retention started training")
    assert_nested_equal(capture_trainer_state(trainer), before)


def test_automatic_save_does_not_overwrite_foreign_run(tmp_path):
    first, second = TinyTrainer(), TinyTrainer()
    with first._checkpoint_context(checkpoint_dir=tmp_path):
        first._checkpoint_commit({"epoch": 1})
    saved = (tmp_path / "checkpoint-1" / checkpoint.STATE_FILENAME).read_bytes()
    with pytest.raises(FileExistsError, match="another run"):
        with second._checkpoint_context(
            checkpoint_dir=tmp_path, checkpoint_total_limit=1
        ):
            second._checkpoint_commit({"epoch": 1})
    assert (tmp_path / "checkpoint-1" / checkpoint.STATE_FILENAME).read_bytes() == saved


def test_automatic_save_expands_user_directory(tmp_path, monkeypatch):
    original = Path.expanduser
    monkeypatch.setattr(
        Path,
        "expanduser",
        lambda self: tmp_path / "runs" if str(self) == "~/runs" else original(self),
    )
    trainer = TinyTrainer()
    with trainer._checkpoint_context(checkpoint_dir="~/runs", checkpoint_total_limit=1):
        trainer._checkpoint_commit({"epoch": 1})
        trainer._checkpoint_commit({"epoch": 2})
    assert (tmp_path / "runs" / "checkpoint-2" / checkpoint.STATE_FILENAME).exists()
    assert not (tmp_path / "runs" / "checkpoint-1").exists()


@pytest.mark.parametrize("kind", ["owned", "legacy", "file", "symlink"])
def test_new_run_rejects_any_existing_snapshot_before_mutation(tmp_path, kind):
    _snapshot(tmp_path, 99, kind=kind)
    trainer, attributes, state = _untouched_trainer()
    with pytest.raises(FileExistsError, match="already contains another run"):
        with trainer._checkpoint_context(
            checkpoint_dir=tmp_path, checkpoint_interval=10
        ):
            pytest.fail("training started despite a checkpoint collision")
    _assert_untouched(trainer, attributes, state)


@pytest.mark.parametrize(
    "kind", ["foreign", "legacy", "bad-sequence", "file", "symlink"]
)
@pytest.mark.parametrize("source_is_file", [False, True])
def test_resume_rejects_future_conflicts_before_loading(tmp_path, kind, source_is_file):
    source_root = tmp_path / "source"
    source = _save_run(source_root)
    output = tmp_path / "output"
    run_id = "f" * 32 if kind == "foreign" else source._checkpoint_run_id
    _snapshot(output, 7, run_id=run_id, kind=kind)
    path = source_root / "checkpoint-1"
    if source_is_file:
        path = path / checkpoint.STATE_FILENAME
    trainer, attributes, state = _untouched_trainer()
    with pytest.raises(FileExistsError):
        with trainer._checkpoint_context(
            resume_from_checkpoint=path, checkpoint_dir=output
        ):
            pytest.fail("training started despite a future conflict")
    _assert_untouched(trainer, attributes, state)


def test_legacy_destination_reports_missing_ownership_before_restore(tmp_path):
    source_root = tmp_path / "source"
    _save_run(source_root)
    output = tmp_path / "output"
    _snapshot(output, 2, kind="legacy")
    trainer, attributes, state = _untouched_trainer()
    with pytest.raises(FileExistsError, match="Legacy or unowned.*ownership metadata"):
        with trainer._checkpoint_context(
            resume_from_checkpoint=source_root / "checkpoint-1", checkpoint_dir=output
        ):
            pytest.fail("legacy destination was not rejected")
    _assert_untouched(trainer, attributes, state)


def test_same_run_future_snapshots_can_be_replaced_on_resume(tmp_path):
    source = _save_run(tmp_path, count=3)
    trainer = TinyTrainer()
    with trainer._checkpoint_context(
        resume_from_checkpoint=tmp_path / "checkpoint-1", checkpoint_dir=tmp_path
    ):
        assert trainer._checkpoint_run_id == source._checkpoint_run_id
        assert trainer._checkpoint_sequence == 1
        trainer._checkpoint_commit({"epoch": 2})
    metadata = checkpoint.read_checkpoint_metadata(tmp_path / "checkpoint-2")
    assert metadata["run_id"] == source._checkpoint_run_id
    assert metadata["sequence"] == 2
    assert (tmp_path / "checkpoint-3").is_dir()


def test_resume_ignores_existing_names_at_or_before_saved_sequence(tmp_path):
    source_root = tmp_path / "source"
    _save_run(source_root, count=3)
    output = tmp_path / "output"
    _snapshot(output, 1, kind="legacy")
    _snapshot(output, 3)
    trainer = TinyTrainer()
    with trainer._checkpoint_context(
        resume_from_checkpoint=source_root / "checkpoint-3", checkpoint_dir=output
    ):
        trainer._checkpoint_commit({"epoch": 4})
    assert (output / "checkpoint-4" / checkpoint.STATE_FILENAME).is_file()


def test_legacy_source_can_resume_into_an_empty_directory(tmp_path):
    source = TinyTrainer()
    source.save_checkpoint(tmp_path / "legacy-source")
    assert (
        checkpoint.read_checkpoint_metadata(tmp_path / "legacy-source")["run_id"]
        is None
    )
    trainer = TinyTrainer()
    with trainer._checkpoint_context(
        resume_from_checkpoint=tmp_path / "legacy-source",
        checkpoint_dir=tmp_path / "output",
    ):
        trainer._checkpoint_commit({"epoch": 1})
    assert checkpoint.read_checkpoint_metadata(tmp_path / "output" / "checkpoint-1")[
        "run_id"
    ]


@pytest.mark.parametrize("kind", ["owned", "legacy", "symlink"])
def test_commit_rechecks_conflicts_created_after_preflight(tmp_path, kind):
    trainer = TinyTrainer()
    with pytest.raises(FileExistsError):
        with trainer._checkpoint_context(checkpoint_dir=tmp_path):
            target = _snapshot(tmp_path, 1, kind=kind)
            trainer._checkpoint_commit({"epoch": 1})
    if kind == "symlink":
        assert target.is_symlink()
    else:
        assert (target / checkpoint.STATE_FILENAME).read_bytes() == b"existing snapshot"


def test_new_run_allows_unrelated_output_entries(tmp_path):
    (tmp_path / "notes.txt").write_text("keep")
    (tmp_path / "_checkpoint_assets").mkdir()
    trainer = _save_run(tmp_path)
    assert trainer._checkpoint_sequence == 1
    assert (tmp_path / "notes.txt").read_text() == "keep"


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_output_file_is_rejected_before_mutating_trainer(tmp_path, kind):
    output = tmp_path / "output"
    if kind == "file":
        output.write_text("keep")
    else:
        output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    trainer, attributes, state = _untouched_trainer()
    with pytest.raises(NotADirectoryError):
        with trainer._checkpoint_context(checkpoint_dir=output):
            pytest.fail("training started with an output file")
    _assert_untouched(trainer, attributes, state)
