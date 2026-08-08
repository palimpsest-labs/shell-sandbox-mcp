"""Tests for the builtin cd shell_run behaviour. Run with the venv python that has `mcp` installed:

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
# cd builtin tests — stub _run_segment / _run_pipeline and record work_dir
# ---------------------------------------------------------------------------


class ShellRunCdTest(unittest.TestCase):
    """Exercise the per-call ``cd`` builtin (AST primary path).

    Stubs ``_run_segment`` to record the ``work_dir`` argument so we can
    assert ``cd`` changes the working directory for subsequent segments
    within the same ``shell_run`` invocation.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # Create an allowed cwd under /tmp with a real subdirectory inside it.
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-cd-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = server._run_background
        self.segment_calls: list[tuple[str, str]] = []  # (cmd_str, work_dir)
        self.segment_rc_map: dict[str, int] = {}

    def tearDown(self) -> None:
        import shutil
        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        server._run_background = self._orig_background
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _cmd_str(self, command) -> str:
        """Convert a CommandNode or str to a display string for stub maps."""
        if isinstance(command, CommandNode):
            return _serialize_command(command)
        return str(command)

    def _stub_segments(self) -> None:
        """Stub _run_segment to record (cmd_str, work_dir) for every call."""
        rc_map = self.segment_rc_map

        def fake_segment(command, work_dir: Path, timeout: int,
                         expansion=None) -> tuple[int, str]:
            c = self._cmd_str(command)
            self.segment_calls.append((c, str(work_dir)))
            rc = rc_map.get(c, 0)
            return rc, f"out:{c}" if rc == 0 else f"err:{c}"

        server._run_segment = fake_segment

    def _stub_pipeline(self) -> None:
        def fake_pipeline(segments, work_dir: Path, timeout: int,
                          expansion=None) -> tuple[int, str]:
            return 0, "pipeline-ok"

        server._run_pipeline = fake_pipeline

    def _stub_background(self) -> None:
        def fake_background(segments, work_dir: Path,
                            expansion=None) -> tuple[int, str, int]:
            return 0, "bg", 0

        server._run_background = fake_background

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.allowed))

    # ------------------------------------------------------------------
    # cd sub && cmd  —  cmd sees the updated work_dir
    # ------------------------------------------------------------------

    def test_cd_sub_and_cmd_uses_updated_work_dir(self) -> None:
        """cd sub && echo hi  →  echo runs inside <allowed>/sub."""
        self._stub_segments()
        out = self._run("cd sub && echo hi")
        # echo should have been called with work_dir = <allowed>/sub
        self.assertEqual(len(self.segment_calls), 1,
                         f"Expected 1 segment call (echo), got {self.segment_calls}")
        cmd_str, wd = self.segment_calls[0]
        self.assertEqual(cmd_str, "echo hi")
        expected = str((self.allowed / "sub").resolve())
        self.assertEqual(wd, expected,
                         f"echo work_dir: expected {expected}, got {wd}")
        self.assertIn("out:echo hi", out)

    def test_cd_sub_semicolon_cmd_uses_updated_work_dir(self) -> None:
        """cd sub ; echo hi  →  echo runs inside <allowed>/sub."""
        self._stub_segments()
        out = self._run("cd sub ; echo hi")
        self.assertEqual(len(self.segment_calls), 1)
        cmd_str, wd = self.segment_calls[0]
        self.assertEqual(cmd_str, "echo hi")
        expected = str((self.allowed / "sub").resolve())
        self.assertEqual(wd, expected)
        self.assertIn("out:echo hi", out)

    # ------------------------------------------------------------------
    # cd nonexistent && cmd  →  cmd skipped (cd fails, rc=1)
    # ------------------------------------------------------------------

    def test_cd_nonexistent_skips_and(self) -> None:
        """cd nonexistent && echo hi  →  echo skipped, error reported."""
        self._stub_segments()
        out = self._run("cd nonexistent && echo hi")
        self.assertEqual(len(self.segment_calls), 0,
                         "echo should NOT have been called")
        self.assertIn("Directory not found", out)
        self.assertIn("skipped", out)

    def test_cd_nonexistent_skips_and_in_output(self) -> None:
        """Verify the skip message appears after a failed cd."""
        self._stub_segments()
        out = self._run("cd no_such_dir && echo x")
        self.assertIn("Directory not found", out)
        self.assertIn("skipped", out)

    # ------------------------------------------------------------------
    # cd /etc && cmd  →  rejected (not in allowed dirs), cmd skipped
    # ------------------------------------------------------------------

    def test_cd_etc_rejected(self) -> None:
        """cd /etc && echo hi  →  rejected, echo skipped."""
        self._stub_segments()
        out = self._run("cd /etc && echo hi")
        self.assertEqual(len(self.segment_calls), 0)
        self.assertIn("not in allowed paths", out)
        self.assertIn("skipped", out)

    # ------------------------------------------------------------------
    # cd .. from allowed root  →  escape rejected
    # ------------------------------------------------------------------

    def test_cd_dotdot_escapes_allowed_root(self) -> None:
        """cd .. from an allowed root (like /tmp) escapes → rejected."""
        self._stub_segments()
        # Run with cwd=/tmp (an allowed root). cd .. → /  (not allowed).
        out = server.shell_run("cd .. && echo hi", cwd="/tmp")
        self.assertIn("not in allowed paths", out)
        self.assertIn("skipped", out)

    # ------------------------------------------------------------------
    # bare cd  →  error, && chain skips
    # ------------------------------------------------------------------

    def test_bare_cd_error_and_skip(self) -> None:
        """Bare cd → 'cd: no directory', && chain skipped."""
        self._stub_segments()
        out = self._run("cd && echo hi")
        self.assertEqual(len(self.segment_calls), 0)
        self.assertIn("cd: no directory", out)
        self.assertIn("skipped", out)

    # ------------------------------------------------------------------
    # cd too many args
    # ------------------------------------------------------------------

    def test_cd_too_many_args(self) -> None:
        """cd a b → 'cd: too many arguments'."""
        self._stub_segments()
        out = self._run("cd a b && echo hi")
        self.assertIn("cd: too many arguments", out)
        self.assertEqual(len(self.segment_calls), 0)

    # ------------------------------------------------------------------
    # standalone cd sub  →  silent success
    # ------------------------------------------------------------------

    def test_standalone_cd_sub_silent(self) -> None:
        """cd sub alone returns '(no output)'."""
        self._stub_segments()
        out = self._run("cd sub")
        self.assertEqual(out, "(no output)")
        self.assertEqual(len(self.segment_calls), 0,
                         "No segment should have been dispatched for cd")

    def test_standalone_cd_sub_ast_fast_path(self) -> None:
        """cd sub alone via the AST fast path returns '(no output)'."""
        self._stub_segments()
        out = self._run("cd sub")
        self.assertEqual(out, "(no output)")

    # ------------------------------------------------------------------
    # cd with dot (self)
    # ------------------------------------------------------------------

    def test_cd_dot_is_noop(self) -> None:
        """cd . is a no-op — subsequent cmd runs in the same dir."""
        self._stub_segments()
        out = self._run("cd . && echo hi")
        self.assertEqual(len(self.segment_calls), 1)
        _cmd_str, wd = self.segment_calls[0]
        self.assertEqual(wd, str(self.allowed.resolve()))
        self.assertIn("out:echo hi", out)

    # ------------------------------------------------------------------
    # cd with ~ expansion
    # ------------------------------------------------------------------

    def test_cd_tilde_expands_then_rejected_not_in_allowed(self) -> None:
        """cd ~ expands HOME (not in allowed dirs) → 'not in allowed paths'.

        HOME is not in the test's allowed set, so ``cd ~`` must fail the
        containment check — proving ``expanduser`` ran on the target before
        joining with work_dir (previously it returned "Directory not found: ~"
        because ``~`` was never expanded).
        """
        self._stub_segments()
        out = server.shell_run("cd ~ && echo hi", cwd=str(self.allowed))
        self.assertEqual(len(self.segment_calls), 0)
        self.assertIn("not in allowed paths", out)
        self.assertNotIn("Directory not found: ~", out)

    # ------------------------------------------------------------------
    # cd -- <dir> (end-of-options)
    # ------------------------------------------------------------------

    def test_cd_dashdash_sub_and_cmd_uses_updated_work_dir(self) -> None:
        """cd -- sub && echo hi  →  echo runs inside <allowed>/sub."""
        self._stub_segments()
        out = self._run("cd -- sub && echo hi")
        self.assertEqual(len(self.segment_calls), 1,
                         f"Expected 1 segment call (echo), got {self.segment_calls}")
        cmd_str, wd = self.segment_calls[0]
        self.assertEqual(cmd_str, "echo hi")
        expected = str((self.allowed / "sub").resolve())
        self.assertEqual(wd, expected,
                         f"echo work_dir: expected {expected}, got {wd}")
        self.assertIn("out:echo hi", out)

    def test_cd_dashdash_alone_no_directory(self) -> None:
        """cd -- alone → 'cd: no directory' (like bare cd)."""
        self._stub_segments()
        out = self._run("cd -- && echo hi")
        self.assertEqual(len(self.segment_calls), 0)
        self.assertIn("cd: no directory", out)
        self.assertIn("skipped", out)



if __name__ == "__main__":
    unittest.main()
