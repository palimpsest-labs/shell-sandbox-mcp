"""Tests for the AST-native ``for`` loop via shell_run. Run with:

    PYTHONPATH=src python3 -m pytest tests/test_builtins_for.py -q
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.server import (
    CommandNode,
    _extract_redirects,
)


def _install_stubs() -> list[dict]:
    """Stub _run_segment/_run_pipeline to record resolved command args."""
    calls: list[dict] = []

    def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
        if isinstance(command, CommandNode):
            args, _, _ = _extract_redirects(command, expansion, work_dir)
            cmd_str = " ".join(args) if args else "<empty>"
        else:
            cmd_str = command
        calls.append({
            "args": cmd_str,
            "work_dir": str(work_dir),
        })
        return 0, ""

    def fake_pipeline(segments, work_dir, timeout, expansion=None, **kwargs):
        str_segs = []
        for s in segments:
            if isinstance(s, CommandNode):
                args, _, _ = _extract_redirects(s, expansion, work_dir)
                str_segs.append(" ".join(args) if args else "<empty>")
            else:
                str_segs.append(str(s))
        cmd_str = " | ".join(str_segs)
        calls.append({
            "args": cmd_str,
            "work_dir": str(work_dir),
        })
        return 0, ""

    server._run_segment = fake_segment
    server._run_pipeline = fake_pipeline
    return calls


def _remove_stubs(orig_segment, orig_pipeline) -> None:
    server._run_segment = orig_segment
    server._run_pipeline = orig_pipeline


class ForLoopTest(unittest.TestCase):
    """Test AST-native for-loop execution via shell_run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-for-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline

    def tearDown(self) -> None:
        _remove_stubs(self._orig_segment, self._orig_pipeline)
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # basic iteration
    # ------------------------------------------------------------------

    def test_basic_iteration(self) -> None:
        """for i in a b c; do echo $i; done → three iterations."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in a b c; do echo $i; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0]["args"], "echo a")
            self.assertEqual(calls[1]["args"], "echo b")
            self.assertEqual(calls[2]["args"], "echo c")
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_no_in_clause_zero_iterations(self) -> None:
        """for i; do echo $i; done → zero iterations."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i; do echo $i; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 0)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_quoted_empty_word_one_iteration(self) -> None:
        """for i in \"\"; do echo $i; done → one iteration (empty word)."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                'for i in ""; do echo $i; done',
                cwd=str(self.allowed), timeout=30,
            )
            # The quoted empty word still iterates once. Inside the loop
            # unquoted $i with an empty value expands to zero args → "echo".
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["args"], "echo")
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_no_in_clause_do_no_semicolon(self) -> None:
        """for i do echo hi; done → zero iterations."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i do echo hi; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 0)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_semicolon_before_do(self) -> None:
        """for i in x y; do echo $i; done → two iterations."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in x y; do echo $i; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["args"], "echo x")
            self.assertEqual(calls[1]["args"], "echo y")
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_no_semicolon_before_do(self) -> None:
        """for i in p q do echo $i; done → two iterations."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in p q do echo $i; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["args"], "echo p")
            self.assertEqual(calls[1]["args"], "echo q")
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # invalid var name
    # ------------------------------------------------------------------

    def test_invalid_var_name_numeric(self) -> None:
        """for 1x in a; do echo hi; done → error."""
        result = server.shell_run(
            "for 1x in a; do echo hi; done",
            cwd=str(self.allowed), timeout=30,
        )
        self.assertIn("invalid variable name", str(result))

    def test_invalid_var_name_dash(self) -> None:
        """for my-var in a; do echo hi; done → error."""
        result = server.shell_run(
            "for my-var in a; do echo hi; done",
            cwd=str(self.allowed), timeout=30,
        )
        self.assertIn("invalid variable name", str(result))

    # ------------------------------------------------------------------
    # body with multiple commands (; separated)
    # ------------------------------------------------------------------

    def test_body_with_semicolons(self) -> None:
        """Body can contain ';' between commands."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in 1 2; do echo a; echo b; done",
                cwd=str(self.allowed), timeout=30,
            )
            # Two iterations, each with two segments (echo a; echo b)
            self.assertGreaterEqual(len(calls), 4)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # $() inside in-words
    # ------------------------------------------------------------------

    def test_dollar_subst_in_in_words(self) -> None:
        """$() inside in-word list expands."""
        orig_capture = server._capture_stdout
        try:
            def fake_capture(command, work_dir, timeout, depth,
                           deadline=None, subst_count=None, env=None):
                return 0, b"EXPANDED"
            server._capture_stdout = fake_capture
            calls = _install_stubs()
            result = server.shell_run(
                "for i in $(echo hello); do echo $i; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["args"], "echo EXPANDED")
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
            server._capture_stdout = orig_capture

    # ------------------------------------------------------------------
    # done inside the body (quoted or in $()) should NOT terminate loop
    # ------------------------------------------------------------------

    def test_done_inside_single_quotes_not_terminator(self) -> None:
        """'done' inside single quotes is part of the body."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in a; do echo 'done'; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("done", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_done_inside_double_quotes_not_terminator(self) -> None:
        """'done' inside double quotes is part of the body."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                'for i in a; do echo "done"; done',
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("done", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_done_inside_dollar_subst_not_terminator(self) -> None:
        """'done' inside $() is part of the body."""
        orig_capture = server._capture_stdout
        try:
            def fake_capture(command, work_dir, timeout, depth,
                           deadline=None, subst_count=None, env=None):
                return 0, b"done-output"
            server._capture_stdout = fake_capture
            calls = _install_stubs()
            result = server.shell_run(
                "for i in a; do echo $(echo done); done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("done-output", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
            server._capture_stdout = orig_capture

    # ------------------------------------------------------------------
    # cd inside loop body does NOT persist across iterations
    # ------------------------------------------------------------------

    def test_cd_inside_body_does_not_persist(self) -> None:
        """cd inside the loop body does NOT persist across iterations."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in a b; do cd sub && echo $i; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 2)
            for call in calls:
                self.assertIn("sub", call["work_dir"])
                self.assertIn("echo", call["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # exit code propagation
    # ------------------------------------------------------------------

    def test_exit_code_last_iteration(self) -> None:
        """Exit code should be the last iteration's rc."""
        iteration = [0]

        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            iteration[0] += 1
            if iteration[0] == 1:
                return 0, "ok"
            return 1, "fail"

        server._run_segment = fake_segment
        server._run_pipeline = lambda segs, wd, to, **kw: (0, "pipe-ok")
        try:
            result = server.shell_run(
                "for i in a b; do echo $i; done",
                cwd=str(self.allowed), timeout=30,
                structured=True,
            )
            self.assertEqual(result["rc"], 1)  # last iteration failed
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_empty_body_no_output(self) -> None:
        """Empty in-list + empty body → (no output), rc 0."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in; do :; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 0)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # single-word in-list
    # ------------------------------------------------------------------

    def test_single_word_in_list(self) -> None:
        """for i in hello; do echo $i; done → one iteration."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in hello; do echo $i; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["args"], "echo hello")
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # for-loop mid-chain (via ;)
    # ------------------------------------------------------------------

    def test_for_mid_chain(self) -> None:
        """for-loop preceded by another command via ;"""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "echo hi; for i in a; do echo $i; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertIn("echo hi", calls[0]["args"])
            self.assertIn("echo a", calls[1]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # done as plain word argument
    # ------------------------------------------------------------------

    def test_done_as_plain_word_argument(self) -> None:
        """'echo done' in the body must not terminate the loop early."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in a; do echo done; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["args"], "echo done")
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_done_done_as_plain_words(self) -> None:
        """'echo done done' — both 'done's are arguments, neither terminates."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in a; do echo done done; done",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("done done", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_done_on_newline_in_body(self) -> None:
        """'done' on a line by itself in body is the terminator."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in a; do echo hi\ndone",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["args"], "echo hi")
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)


if __name__ == "__main__":
    unittest.main()
