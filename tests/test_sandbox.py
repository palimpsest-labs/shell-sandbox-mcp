"""Tests for shell_sandbox_mcp.server security-critical helpers.

These cover path-containment, local-binary resolution, and cwd validation —
the code paths that decide what commands an agent may run. Run with the
venv python that has `mcp` installed:

    PYTHONPATH=src <venv>/bin/python -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server

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


# ---------------------------------------------------------------------------
# _split_command
# ---------------------------------------------------------------------------


class SplitCommandTest(unittest.TestCase):
    def test_no_operator_single_segment(self) -> None:
        self.assertEqual(
            server._split_command("ls -la"),
            [(None, "ls -la")],
        )

    def test_semicolon_splits(self) -> None:
        self.assertEqual(
            server._split_command("echo hi; echo bye"),
            [(None, "echo hi"), (";", "echo bye")],
        )

    def test_and_and_splits(self) -> None:
        self.assertEqual(
            server._split_command("make && make test"),
            [(None, "make"), ("&&", "make test")],
        )

    def test_or_or_splits(self) -> None:
        self.assertEqual(
            server._split_command("false || echo fallback"),
            [(None, "false"), ("||", "echo fallback")],
        )

    def test_mixed_operators(self) -> None:
        self.assertEqual(
            server._split_command("a && b; c || d"),
            [(None, "a"), ("&&", "b"), (";", "c"), ("||", "d")],
        )

    def test_operator_inside_quotes_preserved(self) -> None:
        self.assertEqual(
            server._split_command('echo "a; b"'),
            [(None, 'echo "a; b"')],
        )
        self.assertEqual(
            server._split_command("printf 'a && b'; ls"),
            [(None, "printf 'a && b'"), (";", "ls")],
        )

    def test_whitespace_and_empty_segments_dropped(self) -> None:
        self.assertEqual(
            server._split_command("  a   ;;  b  "),
            [(None, "a"), (";", "b")],
        )

    def test_empty_command(self) -> None:
        self.assertEqual(server._split_command(""), [])
        self.assertEqual(server._split_command("   "), [])

    def test_only_operator_is_empty(self) -> None:
        self.assertEqual(server._split_command(";"), [])

    def test_single_pipe_is_not_split(self) -> None:
        # A lone '|' is not a supported chaining operator, so it stays literal.
        self.assertEqual(
            server._split_command("ls | wc"),
            [(None, "ls | wc")],
        )


# ---------------------------------------------------------------------------
# _run_segment (chaining orchestration)
# ---------------------------------------------------------------------------


class ShellRunChainingTest(unittest.TestCase):
    """Exercise `shell_run` chaining semantics without invoking the real
    sandbox by stubbing `_run_segment`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        # /tmp is allowed by default config
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-chain-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_run_segment = server._run_segment
        self.calls: list[str] = []

    def tearDown(self) -> None:
        import shutil

        server._run_segment = self._orig_run_segment
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def _stub(self, rc_map: dict[str, int]) -> None:
        def fake(command: str, work_dir: Path, timeout: int) -> tuple[int, str]:
            self.calls.append(command)
            rc = rc_map.get(command, 0)
            return rc, f"out:{command}" if rc == 0 else f"err:{command}"

        server._run_segment = fake

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.allowed))

    def test_semicolon_runs_all_segments(self) -> None:
        self._stub({"a": 0, "b": 0})
        out = self._run("a ; b")
        self.assertEqual(self.calls, ["a", "b"])
        self.assertIn("out:a", out)
        self.assertIn("out:b", out)

    def test_andand_skips_after_failure(self) -> None:
        self._stub({"a": 1, "b": 0, "c": 0})
        out = self._run("a && b && c")
        self.assertEqual(self.calls, ["a"])
        self.assertIn("skipped", out)
        self.assertNotIn("out:b", out)
        self.assertNotIn("out:c", out)

    def test_andand_runs_after_success(self) -> None:
        self._stub({"a": 0, "b": 0})
        self._run("a && b")
        self.assertEqual(self.calls, ["a", "b"])

    def test_oror_runs_after_failure(self) -> None:
        self._stub({"a": 1, "b": 0})
        self._run("a || b")
        self.assertEqual(self.calls, ["a", "b"])

    def test_oror_skips_after_success(self) -> None:
        self._stub({"a": 0, "b": 0})
        out = self._run("a || b")
        self.assertEqual(self.calls, ["a"])
        self.assertIn("skipped", out)

    def test_resolution_failure_short_circuits_andand(self) -> None:
        # 'notallowed' fails resolution inside _run_segment (rc 1).
        self._stub({"notallowed": 1, "b": 0})
        out = self._run("notallowed && b")
        self.assertEqual(self.calls, ["notallowed"])


# ---------------------------------------------------------------------------
# _git_config_paths
# ---------------------------------------------------------------------------


class GitConfigPathsTest(unittest.TestCase):
    def test_paths_resolved(self) -> None:
        paths = server._git_config_paths()
        self.assertEqual(len(paths), 2)
        for p in paths:
            # must be absolute, canonical (no '..' or symlinked HOME remainder)
            self.assertTrue(Path(p).is_absolute())
            self.assertEqual(str(Path(p).resolve()), p)


# ---------------------------------------------------------------------------
# _cosmo_toolchain_paths
# ---------------------------------------------------------------------------


class CosmoToolchainPathsTest(unittest.TestCase):
    def test_paths_resolved(self) -> None:
        paths = server._cosmo_toolchain_paths()
        # toolchain tree + APE loader
        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(Path(p).is_absolute())
            self.assertEqual(str(Path(p).resolve()), p)
        # first path must be the vendored toolchain root
        self.assertEqual(Path(paths[0]), server.COSMO_TOOLCHAIN.resolve())

    def test_cosmocc_configured_with_local_toolchain(self) -> None:
        cfg = server.COMMANDS["cosmocc"]
        # binary must point inside the vendored toolchain, not the host install
        self.assertTrue(
            cfg["binary"].startswith(str(server.COSMO_TOOLCHAIN.resolve()))
        )
        self.assertEqual(cfg["extra_unveil_rx"], server._cosmo_toolchain_paths)


if __name__ == "__main__":
    unittest.main()
