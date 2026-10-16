"""Asset digest reuse must avoid rereading immutable history on every save."""

from collections import Counter
import hashlib
import os
from pathlib import Path

import pytest

from comlrl.trainers.preference import checkpointing
from test_checkpoint_assets_review import source_assets


class ObservedReader:
    def __init__(self, stream, path, counts, after_read=None):
        self.stream, self.path, self.counts = stream, path, counts
        self.after_read = after_read

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def fileno(self):
        return self.stream.fileno()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.counts[self.path] += len(data)
        if self.after_read is not None and data:
            callback, self.after_read = self.after_read, None
            callback()
        return data


def observe_hash_reads(monkeypatch, *, changed_path=None, after_read=None):
    counts = Counter()
    original = Path.open

    def open_file(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        if mode == "rb":
            callback = after_read if path == changed_path else None
            return ObservedReader(stream, path, counts, callback)
        return stream

    monkeypatch.setattr(Path, "open", open_file)
    return counts


def test_history_hash_and_copy_bytes_grow_only_with_new_assets(tmp_path, monkeypatch):
    trainer, replay, initial = source_assets(tmp_path / "run")
    counts = observe_hash_reads(monkeypatch)
    copy_bytes = 0
    copy2 = checkpointing.shutil.copy2

    def counted_copy(source, destination, *args, **kwargs):
        nonlocal copy_bytes
        copy_bytes += Path(source).stat().st_size
        return copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(checkpointing.shutil, "copy2", counted_copy)
    unique_bytes = replay.stat().st_size + initial.stat().st_size
    for index in range(6):
        policy = initial.parents[2] / f"iteration_{index:04d}/agent_0/weights.bin"
        policy.parent.mkdir(parents=True)
        payload = bytes([index]) * (128 * 1024)
        policy.write_bytes(payload)
        unique_bytes += len(payload)
        checkpointing.save_preference_assets(
            trainer, tmp_path / "checkpoints" / f"checkpoint-{index}"
        )
        # One source hash and one verification hash per newly copied file.
        assert sum(counts.values()) == 2 * unique_bytes
        assert copy_bytes == unique_bytes
    before_reads, before_copies = sum(counts.values()), copy_bytes
    checkpointing.save_preference_assets(
        trainer, tmp_path / "checkpoints" / "checkpoint-unchanged"
    )
    assert sum(counts.values()) == before_reads
    assert copy_bytes == before_copies
    assert len(trainer._checkpoint_asset_digest_cache) == 8
    assert not any(".asset-" in name for name in trainer._checkpoint_asset_digest_cache)


@pytest.mark.parametrize("change", ["mtime", "same_size_restored_mtime", "replace"])
def test_digest_cache_invalidates_modified_or_replaced_source(
    tmp_path, monkeypatch, change
):
    path = tmp_path / "weights.bin"
    original, modified = b"original weights", b"modified weights"
    path.write_bytes(original)
    cache = {}
    counts = observe_hash_reads(monkeypatch)
    assert (
        checkpointing._asset_digest(path, cache=cache)
        == hashlib.sha256(original).hexdigest()
    )
    assert (
        checkpointing._asset_digest(path, cache=cache)
        == hashlib.sha256(original).hexdigest()
    )
    assert sum(counts.values()) == len(original)
    previous = path.stat()
    if change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(modified)
        os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        os.replace(replacement, path)
    else:
        path.write_bytes(modified)
        if change == "same_size_restored_mtime":
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert checkpointing._asset_stat_key(path.stat()) != checkpointing._asset_stat_key(
        previous
    )
    assert (
        checkpointing._asset_digest(path, cache=cache)
        == hashlib.sha256(modified).hexdigest()
    )
    assert sum(counts.values()) == len(original) + len(modified)
    assert len(cache) == 1


def test_reused_blob_needs_no_byte_reads_and_changed_source_gets_new_blob(
    tmp_path, monkeypatch
):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"first")
    pool, cache = tmp_path / "pool", {}
    counts = observe_hash_reads(monkeypatch)
    first = tmp_path / "first/weights.bin"
    checkpointing._copy_checkpoint_asset(source, first, pool, digest_cache=cache)
    first_reads = sum(counts.values())
    second = tmp_path / "second/weights.bin"
    checkpointing._copy_checkpoint_asset(source, second, pool, digest_cache=cache)
    assert sum(counts.values()) == first_reads
    assert first.stat().st_ino == second.stat().st_ino
    source.write_bytes(b"new weights")
    third = tmp_path / "third/weights.bin"
    checkpointing._copy_checkpoint_asset(source, third, pool, digest_cache=cache)
    assert sum(counts.values()) == first_reads + 2 * len(b"new weights")
    assert third.stat().st_ino != first.stat().st_ino


def test_source_change_during_hash_is_rejected_without_caching(tmp_path, monkeypatch):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"before")
    cache = {}
    observe_hash_reads(
        monkeypatch,
        changed_path=source,
        after_read=lambda: source.write_bytes(b"after!"),
    )
    with pytest.raises(RuntimeError, match="changed during checkpoint saving"):
        checkpointing._asset_digest(source, cache=cache)
    assert not cache


def test_source_change_during_initial_copy_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"original")
    copy2 = checkpointing.shutil.copy2

    def change_after_copy(src, dst, *args, **kwargs):
        result = copy2(src, dst, *args, **kwargs)
        if Path(src) == source:
            source.write_bytes(b"modified")
        return result

    monkeypatch.setattr(checkpointing.shutil, "copy2", change_after_copy)
    destination, pool = tmp_path / "snapshot/weights.bin", tmp_path / "pool"
    with pytest.raises(RuntimeError, match="changed during checkpoint saving"):
        checkpointing._copy_checkpoint_asset(source, destination, pool, digest_cache={})
    assert not destination.exists()
    assert not list(pool.iterdir())


def test_corrupted_initial_copy_is_rejected_before_pool_publication(
    tmp_path, monkeypatch
):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"original")
    copy2 = checkpointing.shutil.copy2

    def corrupt_copy(src, dst, *args, **kwargs):
        result = copy2(src, dst, *args, **kwargs)
        Path(dst).write_bytes(b"corrupt!")
        return result

    monkeypatch.setattr(checkpointing.shutil, "copy2", corrupt_copy)
    destination, pool = tmp_path / "snapshot/weights.bin", tmp_path / "pool"
    with pytest.raises(RuntimeError, match="changed during checkpoint saving"):
        checkpointing._copy_checkpoint_asset(source, destination, pool, digest_cache={})
    assert not destination.exists()
    assert not list(pool.iterdir())


def test_cached_source_change_during_link_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"original")
    cache, pool = {}, tmp_path / "pool"
    checkpointing._copy_checkpoint_asset(
        source, tmp_path / "first/weights.bin", pool, digest_cache=cache
    )
    destination = tmp_path / "second/weights.bin"
    link = os.link

    def change_during_link(src, dst, *args, **kwargs):
        result = link(src, dst, *args, **kwargs)
        if Path(dst) == destination:
            source.write_bytes(b"modified")
        return result

    monkeypatch.setattr(os, "link", change_during_link)
    with pytest.raises(RuntimeError, match="changed during checkpoint saving"):
        checkpointing._copy_checkpoint_asset(
            source, destination, pool, digest_cache=cache
        )
    assert not destination.exists()


@pytest.mark.parametrize("damage", ["size", "writable"])
def test_existing_blob_size_and_readonly_guards_remain(tmp_path, damage):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"original")
    cache, pool = {}, tmp_path / "pool"
    checkpointing._copy_checkpoint_asset(
        source, tmp_path / "first/weights.bin", pool, digest_cache=cache
    )
    blob = pool / hashlib.sha256(b"original").hexdigest()
    blob.chmod(0o644)
    if damage == "size":
        blob.write_bytes(b"short")
        blob.chmod(0o444)
    with pytest.raises(ValueError, match="damaged blob"):
        checkpointing._copy_checkpoint_asset(
            source, tmp_path / "second/weights.bin", pool, digest_cache=cache
        )
