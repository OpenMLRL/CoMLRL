"""CPU-only launch tests: no torch, transformers, model downloads, or GPU needed."""

import importlib.util
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "runtime_cache_under_test",
    Path(__file__).resolve().parents[1] / "comlrl/runtime.py",
)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class RuntimeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.torch = patch.dict(sys.modules, {"torch": None})
        self.torch.start()
        self.addCleanup(self.torch.stop)

    def launch_env(self, job="123"):
        os.environ.update(SLURM_JOB_ID=job, COMLRL_CUDA_CACHE_ROOT=self.temp.name)

    def test_non_slurm_is_unchanged(self):
        with patch.object(runtime.tempfile, "mkdtemp") as create:
            self.assertIsNone(runtime.configure_job_cuda_cache())
            create.assert_not_called()

    def test_private_cache_and_idempotence(self):
        self.launch_env()
        before = dict(os.environ)
        with patch("sys.stderr", new=io.StringIO()) as log:
            cache = runtime.configure_job_cuda_cache()
            self.assertEqual(runtime.configure_job_cuda_cache(), cache)
        self.assertEqual(Path(cache).parent.resolve(), Path(self.temp.name).resolve())
        self.assertEqual(stat.S_IMODE(Path(cache).stat().st_mode), 0o700)
        self.assertEqual(dict(os.environ), {**before, "CUDA_CACHE_PATH": cache})
        self.assertEqual(log.getvalue().count("[runtime]"), 1)

    def test_separate_launches_do_not_share_index(self):
        self.launch_env("123")
        first = runtime.configure_job_cuda_cache()
        del os.environ["CUDA_CACHE_PATH"]
        os.environ["SLURM_JOB_ID"] = "456"
        second = runtime.configure_job_cuda_cache()
        self.assertNotEqual(first, second)
        self.assertTrue(Path(first).is_dir())  # Never deletes another cache.

    def test_child_process_inherits_cache_without_creating_another(self):
        self.launch_env()
        cache = runtime.configure_job_cuda_cache()
        code = (
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('runtime', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "print(module.configure_job_cuda_cache())"
        )
        child = subprocess.run(
            [sys.executable, "-c", code, runtime.__file__],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(child.stdout.strip(), cache)
        self.assertEqual(child.stderr, "")
        self.assertEqual(len(list(Path(self.temp.name).iterdir())), 1)

    def test_explicit_cache_is_preserved(self):
        self.launch_env()
        os.environ["CUDA_CACHE_PATH"] = "/explicit/user/cache"
        with patch.object(runtime.tempfile, "mkdtemp") as create:
            self.assertEqual(runtime.configure_job_cuda_cache(), "/explicit/user/cache")
            create.assert_not_called()

    def test_disabled_cache_is_preserved(self):
        self.launch_env()
        os.environ["CUDA_CACHE_DISABLE"] = "1"
        self.assertIsNone(runtime.configure_job_cuda_cache())
        self.assertNotIn("CUDA_CACHE_PATH", os.environ)

    def test_legacy_job_id_and_slurm_tmpdir(self):
        os.environ.update(SLURM_JOBID="123_4", SLURM_TMPDIR=self.temp.name)
        cache = runtime.configure_job_cuda_cache()
        self.assertEqual(Path(cache).parent.resolve(), Path(self.temp.name).resolve())

    def test_tmpdir_does_not_redirect_default_to_network_home(self):
        os.environ.update(SLURM_JOB_ID="123", TMPDIR="/network/home/tmp")
        with patch.object(
            runtime.tempfile, "mkdtemp", return_value="/tmp/private"
        ) as create:
            with patch.object(runtime, "_filesystem_type", return_value="ext4"):
                runtime.configure_job_cuda_cache()
        self.assertEqual(create.call_args.kwargs["dir"], "/tmp")

    def test_shared_slurm_tmpdir_falls_back_to_tmp(self):
        os.environ.update(SLURM_JOB_ID="123", SLURM_TMPDIR=self.temp.name)
        with patch.object(runtime, "_filesystem_type", side_effect=["nfs", "ext4"]):
            with patch.object(
                runtime.tempfile, "mkdtemp", return_value="/tmp/private"
            ) as create:
                runtime.configure_job_cuda_cache()
        self.assertEqual(create.call_args.kwargs["dir"], "/tmp")

    def test_explicit_shared_root_is_rejected(self):
        self.launch_env()
        with patch.object(runtime, "_filesystem_type", return_value="nfs"):
            with self.assertRaisesRegex(RuntimeError, "node-local"):
                runtime.configure_job_cuda_cache()
        self.assertNotIn("CUDA_CACHE_PATH", os.environ)

    def test_invalid_or_missing_override_root(self):
        self.launch_env()
        for value in ["relative/path", str(Path(self.temp.name) / "missing")]:
            os.environ["COMLRL_CUDA_CACHE_ROOT"] = value
            with self.assertRaises((RuntimeError, ValueError)):
                runtime.configure_job_cuda_cache()

    def test_invalid_job_id(self):
        self.launch_env("../../escape")
        with self.assertRaises(ValueError):
            runtime.configure_job_cuda_cache()

    def test_already_initialized_cuda_is_rejected(self):
        self.launch_env()
        initialized_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_initialized=lambda: True)
        )
        with patch.dict(sys.modules, {"torch": initialized_torch}):
            with self.assertRaisesRegex(RuntimeError, "before initializing CUDA"):
                runtime.configure_job_cuda_cache()

    def test_create_failure_does_not_fall_back_to_shared_cache(self):
        self.launch_env()
        with patch.object(
            runtime.tempfile, "mkdtemp", side_effect=PermissionError("read-only")
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot create"):
                runtime.configure_job_cuda_cache()
        self.assertNotIn("CUDA_CACHE_PATH", os.environ)

    def test_mount_parser_uses_longest_match_and_unescapes_spaces(self):
        mounts = (
            "1 0 0:1 / / rw - ext4 disk rw\n"
            "5 1 0:5 / /home rw - autofs systemd-1 rw\n"
            "2 1 0:2 / /home rw - nfs server:/home rw\n"
            "3 1 0:3 / /tmp rw - tmpfs tmpfs rw\n"
            "4 3 0:4 / /tmp/shared\\040dir rw - nfs server:/tmp rw\n"
        )
        with patch.object(Path, "exists", return_value=True), patch.object(
            Path, "read_text", return_value=mounts
        ), patch.object(Path, "resolve", lambda self: self):
            self.assertEqual(runtime._filesystem_type(Path("/home/user/cache")), "nfs")
            self.assertEqual(runtime._filesystem_type(Path("/tmp/local")), "tmpfs")
            self.assertEqual(
                runtime._filesystem_type(Path("/tmp/shared dir/cache")), "nfs"
            )


if __name__ == "__main__":
    unittest.main()
