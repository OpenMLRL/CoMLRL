"""Automatic checkpoint destinations are validated before restoring or training."""

import json

import pytest

import comlrl.utils.checkpoint as checkpoint
from test_checkpoint_state import TinyTrainer, assert_nested_equal


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
    return trainer, dict(vars(trainer)), checkpoint.capture_trainer_state(trainer)


def _assert_untouched(trainer, attributes, state):
    assert set(vars(trainer)) == set(attributes)
    assert all(vars(trainer)[name] is value for name, value in attributes.items())
    assert_nested_equal(checkpoint.capture_trainer_state(trainer), state)


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
