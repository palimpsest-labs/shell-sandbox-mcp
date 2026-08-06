"""End-to-end smoke tests. Run with the venv python that has `mcp` installed:

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

class EndToEndSmokeTest(unittest.TestCase):
    """Real end-to-end smoke tests that go through shell_run.

    Note: these may fail when running inside a sandbox (sandbox-in-sandbox).
    The core logic is verified by the other test classes.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.work_dir))

    def test_heredoc_expansion_produces_correct_result(self) -> None:
        """Verify the full expansion pipeline without subprocess."""
        cmd = "cat <<EOF\nhello\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        # Verify the expanded command has a heredoc sentinel
        self.assertIn("<<", expanded)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn(sentinel, exp.heredoc_bodies)
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")
        args, redirs, err = _extract_redirects(
            "cat << " + sentinel, expansion=exp,
        )
        self.assertIsNone(err)
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].body, "hello\n")

    def test_heredoc_single_quoted_literal(self) -> None:
        """Verify single-quoted delimiters produce literal bodies."""
        cmd = "cat <<'EOF'\n$(echo hi)\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")

    def test_herestring_expansion(self) -> None:
        """Verify here-string produces correct body."""
        cmd = "cat <<<hello world"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello world\n")

    def test_command_substitution_single_arg(self) -> None:
        """Verify $(...) produces a sentinel with single-word value."""
        original = server._capture_stdout
        try:
            def fake(command, work_dir, timeout, depth, deadline=None, subst_count=None, env=None):
                return 0, b"a b"
            server._capture_stdout = fake
            cmd = "echo $(printf 'a b')"
            expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
            m = SENTINEL_ARG.search(expanded)
            self.assertIsNotNone(m)
            sentinel = f"\x01A{m.group(1)}\x01"
            self.assertEqual(exp.arg_values[sentinel], "a b")
            args, redirs, err = _extract_redirects(
                "echo " + sentinel, expansion=exp,
            )
            self.assertEqual(args, ["echo", "a b"])
        finally:
            server._capture_stdout = original

    def test_nested_heredoc_in_substitution(self) -> None:
        """Verify that _expand_command can handle $(cat <<EOF\nx\nEOF)."""
        original = server._capture_stdout
        try:
            def fake(command, work_dir, timeout, depth, deadline=None, subst_count=None, env=None):
                return 0, b"x"
            server._capture_stdout = fake
            cmd = "echo $(cat <<EOF\nx\nEOF)"
            expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
            m = SENTINEL_ARG.search(expanded)
            self.assertIsNotNone(m)
            sentinel = f"\x01A{m.group(1)}\x01"
            self.assertEqual(exp.arg_values.get(sentinel), "x")
        finally:
            server._capture_stdout = original



if __name__ == "__main__":
    unittest.main()
