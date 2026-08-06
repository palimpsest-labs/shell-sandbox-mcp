"""Tests for path-containment, local-binary resolution, and cwd validation. Run with the venv python that has `mcp` installed:

    PYTHONPATH=src <venv>/bin/python -m unittest discover -s tests -v
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.server import (
    CommandNode,
    EmptyInvocation,
    Expansion,
    FdPlan,
    Invocation,
    InvocationError,
    ProgramNode,
    Redirect,
    SENTINEL_ARG,
    SENTINEL_HD,
    _expand_command,
    _capture_stdout,
    _extract_redirects,
    _build_invocation,
    _resolve_fd_targets,
    _run_segment_core,
    _run_pipeline_core,
    _serialize_command,
    MAX_SUBST_DEPTH,
    MAX_SUBST_COUNT,
    MAX_SUBST_OUTPUT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exec(path: Path) -> None:
    path.write_text("#!/bin/sh\necho hi\n")
    path.chmod(0o755)

# ---------------------------------------------------------------------------
# _contained_path / _resolve_local_binary
# ---------------------------------------------------------------------------


class LocalBinaryResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "build").mkdir()
        _make_exec(self.root / "hello")
        _make_exec(self.root / "build" / "tool")
        _make_exec(self.root / "plain_name")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_bare_name_resolves(self) -> None:
        self.assertEqual(
            server._resolve_local_binary("hello", self.root),
            str((self.root / "hello").resolve()),
        )

    def test_relative_path_resolves(self) -> None:
        self.assertEqual(
            server._resolve_local_binary("./hello", self.root),
            str((self.root / "hello").resolve()),
        )
        self.assertEqual(
            server._resolve_local_binary("build/tool", self.root),
            str((self.root / "build" / "tool").resolve()),
        )

    def test_absolute_path_inside_cwd(self) -> None:
        self.assertEqual(
            server._resolve_local_binary(str(self.root / "hello"), self.root),
            str((self.root / "hello").resolve()),
        )

    def test_dotdot_escape_rejected(self) -> None:
        self.assertIsNone(server._resolve_local_binary("../escape", self.root))
        # craft a path that resolves above root
        self.assertIsNone(server._resolve_local_binary("../../etc/passwd", self.root))

    def test_absolute_path_outside_cwd_rejected(self) -> None:
        self.assertIsNone(server._resolve_local_binary("/etc/hostname", self.root))

    def test_dot_and_dotdot_rejected(self) -> None:
        self.assertIsNone(server._resolve_local_binary(".", self.root))
        self.assertIsNone(server._resolve_local_binary("..", self.root))

    def test_nonexistent_rejected(self) -> None:
        self.assertIsNone(server._resolve_local_binary("./nope", self.root))

    def test_non_executable_rejected(self) -> None:
        (self.root / "notexec").write_text("#!/bin/sh\n")
        (self.root / "notexec").chmod(0o644)
        self.assertIsNone(server._resolve_local_binary("./notexec", self.root))

    def test_symlink_escaping_cwd_rejected(self) -> None:
        # symlink inside root -> /etc; resolve() follows it, containment fails
        (self.root / "evil").symlink_to("/etc")
        self.assertIsNone(server._resolve_local_binary("evil/hostname", self.root))

    def test_symlink_within_cwd_allowed(self) -> None:
        target = self.root / "build" / "tool"
        link = self.root / "mylink"
        link.symlink_to(target)
        self.assertEqual(
            server._resolve_local_binary("mylink", self.root),
            str(target.resolve()),
        )

# ---------------------------------------------------------------------------
# _validate_cwd
# ---------------------------------------------------------------------------


class ValidateCwdTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tmpdir = Path(tempfile.gettempdir())
        # /tmp is an allowed dir in the default config
        self.allowed = self.tmpdir / ("sandbox-test-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self.sub = self.allowed / "sub"
        self.sub.mkdir()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def test_allowed_dir_and_subdir_ok(self) -> None:
        self.assertIsNone(server._validate_cwd(self.allowed.resolve(), str(self.allowed)))
        self.assertIsNone(server._validate_cwd(self.sub.resolve(), str(self.sub)))

    def test_missing_dir_error(self) -> None:
        missing = self.allowed / "does-not-exist"
        err = server._validate_cwd(missing.resolve(), "does-not-exist")
        self.assertIn("Directory not found", err)

    def test_outside_allowed_rejected(self) -> None:
        # home dir is NOT under an allowed dir (~/projects or /tmp)
        home = Path.home()
        err = server._validate_cwd(home, str(home))
        self.assertIn("not in allowed paths", err)

    def test_uses_raw_input_in_message(self) -> None:
        home = Path.home()
        err = server._validate_cwd(home, "~/user-typed-path")
        self.assertIn("~/user-typed-path", err)

# ---------------------------------------------------------------------------
# _binary_still_contained (TOCTOU narrowing)
# ---------------------------------------------------------------------------


class BinaryStillContainedTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_exec(self.root / "tool")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_valid_local_binary_true(self) -> None:
        self.assertTrue(
            server._binary_still_contained(str((self.root / "tool").resolve()), self.root)
        )

    def test_removed_binary_false(self) -> None:
        (self.root / "tool").unlink()
        self.assertFalse(
            server._binary_still_contained(str((self.root / "tool").resolve()), self.root)
        )

    def test_outside_path_false(self) -> None:
        self.assertFalse(
            server._binary_still_contained("/bin/sh", self.root)
        )

    def test_swapped_to_symlink_escape_false(self) -> None:
        # Replace the real tool path with a symlink escaping the tree.
        (self.root / "tool").unlink()
        link = self.root / "tool"
        link.symlink_to("/bin")
        self.assertFalse(
            server._binary_still_contained(str(link.resolve()), self.root)
        )



if __name__ == "__main__":
    unittest.main()
