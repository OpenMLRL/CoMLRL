"""Exercise checkpoint retention with portable preference assets and reuse."""

import hashlib
from pathlib import Path
import random
import shutil

import pytest
import torch

from comlrl.trainers.preference import MADPOIterTrainer, MARLHFIterTrainer
from comlrl.utils.checkpoint import STATE_FILENAME, _decode
from test_checkpoint_preference import (
    Interrupted,
    TinyACTrainer,
    assert_same_training,
    iterative_trainer,
    marlhf_ac_trainer,
    seed,
    stop_after,
)
from test_checkpoint_state import assert_nested_equal


@pytest.mark.parametrize("cls", [MADPOIterTrainer, MARLHFIterTrainer])
def test_iterative_retention_prunes_snapshots_and_resumes_shared_assets(tmp_path, cls):
    seed(51)
    baseline = iterative_trainer(tmp_path / "baseline", cls)
    baseline.train()
    expected_rng = (random.random(), torch.rand(()).item())

    seed(51)
    interrupted = iterative_trainer(tmp_path / "original", cls)
    checkpoints = tmp_path / "checkpoints"
    pool = checkpoints / ".comlrl-preference-assets"
    orphan_bytes = b"asset left by an interrupted, unpublished snapshot"
    orphan = pool / hashlib.sha256(orphan_bytes).hexdigest()
    commit = interrupted._checkpoint_commit

    def commit_and_interrupt(progress, force=False):
        commit(progress, force=force)
        if progress["iteration"] == 2:
            orphan.write_bytes(orphan_bytes)
            orphan.chmod(0o444)
            assert orphan.stat().st_nlink == 1
        if progress["iteration"] == 3:
            raise Interrupted

    interrupted._checkpoint_commit = commit_and_interrupt
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=checkpoints, checkpoint_total_limit=2)
    assert sorted(path.name for path in checkpoints.glob("checkpoint-*")) == [
        "checkpoint-2",
        "checkpoint-3",
    ]
    assert not orphan.exists()
    relative = "assets/policy_history/initial/agent_0/weights.pt"
    previous = checkpoints / "checkpoint-2" / relative
    latest = checkpoints / "checkpoint-3" / relative
    assert latest.stat().st_ino == previous.stat().st_ino
    assert not latest.stat().st_mode & 0o222
    links = [
        path
        for path in checkpoints.rglob("*")
        if path.is_file() and path.stat().st_ino == latest.stat().st_ino
    ]
    assert latest.stat().st_nlink == len(links)
    assert latest.stat().st_nlink >= 3  # Two retained snapshots plus the pool.
    assert all(path.stat().st_nlink > 1 for path in pool.iterdir())

    shutil.rmtree(tmp_path / "original")
    seed(92)
    resumed = iterative_trainer(tmp_path / "resumed", cls)
    resumed.train(
        resume_from_checkpoint=checkpoints / "checkpoint-3",
        checkpoint_dir=checkpoints,
        checkpoint_total_limit=2,
    )
    assert_same_training(baseline, resumed)
    assert baseline.events == interrupted.events + resumed.events
    assert expected_rng == (random.random(), torch.rand(()).item())
    assert sorted(path.name for path in checkpoints.glob("checkpoint-*")) == [
        "checkpoint-3",
        "checkpoint-4",
    ]
    assert Path(resumed._policy_checkpoint_dir_path).is_relative_to(
        tmp_path / "resumed"
    )
    assert Path(resumed._preference_replay_dir_path).is_relative_to(
        tmp_path / "resumed"
    )
    restored = Path(resumed._policy_checkpoint_dir_path) / "initial/agent_0/weights.pt"
    assert restored.stat().st_ino != latest.stat().st_ino
    assert restored.stat().st_mode & 0o200
    assert all(path.stat().st_nlink > 1 for path in pool.iterdir())


@pytest.mark.parametrize("checkpoint_number", [1, 2])
def test_loading_reused_marlhf_parent_discards_stale_child_before_manual_save(
    tmp_path,
    monkeypatch,
    checkpoint_number,
):
    from comlrl.trainers.preference import marlhf

    monkeypatch.setattr(marlhf, "MAACTrainer", TinyACTrainer)
    seed(31)
    source = marlhf_ac_trainer()
    stop_after(source, lambda progress: progress.get("rl", {}).get("epoch") == 1)
    with pytest.raises(Interrupted):
        source.train(checkpoint_dir=tmp_path / "source")
    source_path = tmp_path / "source" / f"checkpoint-{checkpoint_number}"
    expected = _decode(torch.load(source_path / STATE_FILENAME, weights_only=True))

    seed(74)
    target = marlhf_ac_trainer()
    target.train(checkpoint_dir=tmp_path / "previous-run")
    old_child = target._checkpoint_ac_trainer
    assert old_child is not None
    target.load_checkpoint(source_path)
    assert target._checkpoint_ac_trainer is None
    assert_nested_equal(
        target._checkpoint_pending_ac_state,
        expected["extra"].get("actor_critic"),
    )
    manual = tmp_path / "manual"
    target.save_checkpoint(manual)
    saved = _decode(torch.load(manual / STATE_FILENAME, weights_only=True))
    assert_nested_equal(saved["extra"], expected["extra"])
