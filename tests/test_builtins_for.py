"""Tests for the ``for`` loop builtin. Run with:

    PYTHONPATH=src <venv>/bin/python -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.server import (
    CommandNode,
    Expansion,
    _extract_redirects,
    _try_for_loop,
)


# ---------------------------------------------------------------------------
# Scanner unit tests — exercise _try_for_loop directly (no subprocess)
# ---------------------------------------------------------------------------

class ForLoopScannerTest(unittest.TestCase):
    """Test the for-loop scanner / grammar detection via _try_for_loop.

    Stubs _run_segment / _run_pipeline so we can verify scan results without
    launching real subprocesses.  The stubs resolve command args through the
    expansion so we can assert on the actual expanded argument values.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-for-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()  # for cd tests
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline

    def tearDown(self) -> None:
        import shutil
        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _install_stubs(self) -> list[dict]:
        """Stub _run_segment/_run_pipeline to record resolved command args.

        Each stub resolves the CommandNode through the Expansion and records
        the resolved args + work_dir.  Returns the calls list.
        """
        calls: list[dict] = []

        def fake_segment(command, work_dir, timeout, expansion=None):
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

        def fake_pipeline(segments, work_dir, timeout, expansion=None):
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

    # ------------------------------------------------------------------
    # detection / grammar
    # ------------------------------------------------------------------

    def test_not_a_for_loop_returns_none(self) -> None:
        """Plain echo is not a for-loop."""
        result = _try_for_loop("echo hello", self.allowed, 30)
        self.assertIsNone(result)

    def test_for_must_be_first_word(self) -> None:
        """'echo for i in a; do ...' is not a for-loop."""
        result = _try_for_loop("echo for i in a; do echo hi; done", self.allowed, 30)
        self.assertIsNone(result)

    def test_foreign_not_for(self) -> None:
        """'foreign i in a b; do echo hi; done' is not a for-loop."""
        result = _try_for_loop("foreign i in a b; do echo hi; done", self.allowed, 30)
        self.assertIsNone(result)

    def test_incomplete_missing_do(self) -> None:
        """'for i in a b' is not a for-loop (missing do...done)."""
        result = _try_for_loop("for i in a b", self.allowed, 30)
        self.assertIsNone(result)

    def test_incomplete_missing_done(self) -> None:
        """'for i in a; do echo hi' is not a for-loop (missing done)."""
        result = _try_for_loop("for i in a; do echo hi", self.allowed, 30)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # basic iteration
    # ------------------------------------------------------------------

    def test_basic_iteration(self) -> None:
        """for i in a b c; do echo $i; done → three iterations."""
        calls = self._install_stubs()
        result = _try_for_loop("for i in a b c; do echo $i; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["args"], "echo a")
        self.assertEqual(calls[1]["args"], "echo b")
        self.assertEqual(calls[2]["args"], "echo c")

    def test_no_in_clause_zero_iterations(self) -> None:
        """for i; do echo $i; done → zero iterations (no positional params)."""
        calls = self._install_stubs()
        result = _try_for_loop("for i; do echo $i; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 0)
        self.assertEqual(out, "(no output)")

    def test_no_in_clause_do_no_semicolon(self) -> None:
        """for i do echo hi; done → zero iterations."""
        calls = self._install_stubs()
        result = _try_for_loop("for i do echo hi; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 0)

    def test_semicolon_before_do(self) -> None:
        """for i in x y; do echo $i; done → two iterations."""
        calls = self._install_stubs()
        result = _try_for_loop("for i in x y; do echo $i; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["args"], "echo x")
        self.assertEqual(calls[1]["args"], "echo y")

    def test_no_semicolon_before_do(self) -> None:
        """for i in p q do echo $i; done → two iterations."""
        calls = self._install_stubs()
        result = _try_for_loop("for i in p q do echo $i; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["args"], "echo p")
        self.assertEqual(calls[1]["args"], "echo q")

    # ------------------------------------------------------------------
    # invalid var name
    # ------------------------------------------------------------------

    def test_invalid_var_name_numeric(self) -> None:
        """for 1x in a; do echo hi; done → error."""
        result = _try_for_loop("for 1x in a; do echo hi; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(rc, 1)
        self.assertIn("invalid variable name", out)

    def test_invalid_var_name_dash(self) -> None:
        """for my-var in a; do echo hi; done → error."""
        result = _try_for_loop("for my-var in a; do echo hi; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(rc, 1)

    # ------------------------------------------------------------------
    # body with multiple commands (; separated)
    # ------------------------------------------------------------------

    def test_body_with_semicolons(self) -> None:
        """Body can contain ';' between commands."""
        calls = self._install_stubs()
        result = _try_for_loop("for i in 1 2; do echo a; echo b; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        # Two iterations, each with two segments (echo a; echo b)
        self.assertGreaterEqual(len(calls), 4)

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
            calls = self._install_stubs()
            result = _try_for_loop(
                "for i in $(echo hello); do echo $i; done",
                self.allowed, 30,
            )
            self.assertIsNotNone(result)
            out, rc = result
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["args"], "echo EXPANDED")
        finally:
            server._capture_stdout = orig_capture

    # ------------------------------------------------------------------
    # done inside the body (quoted or in $()) should NOT terminate loop
    # ------------------------------------------------------------------

    def test_done_inside_single_quotes_not_terminator(self) -> None:
        """'done' inside single quotes is part of the body."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do echo 'done'; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)
        # Single-quoted 'done' stays as literal text
        self.assertIn("done", calls[0]["args"])

    def test_done_inside_double_quotes_not_terminator(self) -> None:
        """'done' inside double quotes is part of the body."""
        calls = self._install_stubs()
        result = _try_for_loop(
            'for i in a; do echo "done"; done',
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)
        self.assertIn("done", calls[0]["args"])

    def test_done_inside_dollar_subst_not_terminator(self) -> None:
        """'done' inside $() is part of the body."""
        orig_capture = server._capture_stdout
        try:
            def fake_capture(command, work_dir, timeout, depth,
                           deadline=None, subst_count=None, env=None):
                return 0, b"done-output"
            server._capture_stdout = fake_capture
            calls = self._install_stubs()
            result = _try_for_loop(
                "for i in a; do echo $(echo done); done",
                self.allowed, 30,
            )
            self.assertIsNotNone(result)
            out, rc = result
            self.assertEqual(len(calls), 1)
            # The $(...) resolves to "done-output" via the stub
            self.assertIn("done-output", calls[0]["args"])
        finally:
            server._capture_stdout = orig_capture

    # ------------------------------------------------------------------
    # cd inside loop body does NOT persist across iterations
    # ------------------------------------------------------------------

    def test_cd_inside_body_does_not_persist(self) -> None:
        """cd inside the loop body does NOT persist across iterations.

        Within a single iteration, ``cd sub && echo $i`` runs echo in the new
        directory.  But the next iteration starts fresh from the original
        work_dir (the Runner is re-created each time).
        """
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a b; do cd sub && echo $i; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        # Two iterations.  Within each, cd is intercepted by the builtin
        # (not dispatched to _run_segment), so we get one call per iteration
        # (the echo command).  Both echo commands run with work_dir=.../sub
        # because cd updated it within the chain.
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIn("sub", call["work_dir"])
            self.assertIn("echo", call["args"])

    # ------------------------------------------------------------------
    # exit code propagation
    # ------------------------------------------------------------------

    def test_exit_code_last_iteration(self) -> None:
        """Exit code should be the last iteration's rc."""
        iteration = [0]

        def fake_segment(command, work_dir, timeout, expansion=None):
            iteration[0] += 1
            if iteration[0] == 1:
                return 0, "ok"
            return 1, "fail"

        server._run_segment = fake_segment
        server._run_pipeline = lambda segs, wd, to, expansion=None: (0, "pipe-ok")
        try:
            result = _try_for_loop("for i in a b; do echo $i; done", self.allowed, 30)
            self.assertIsNotNone(result)
            out, rc = result
            self.assertEqual(rc, 1)  # last iteration failed
        finally:
            server._run_segment = self._orig_segment
            server._run_pipeline = self._orig_pipeline

    def test_empty_body_no_output(self) -> None:
        """Empty in-list + empty body → (no output), rc 0."""
        calls = self._install_stubs()
        result = _try_for_loop("for i in; do :; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(rc, 0)
        self.assertEqual(out, "(no output)")

    # ------------------------------------------------------------------
    # single-word in-list (no spaces)
    # ------------------------------------------------------------------

    def test_single_word_in_list(self) -> None:
        """for i in hello; do echo $i; done → one iteration."""
        calls = self._install_stubs()
        result = _try_for_loop("for i in hello; do echo $i; done", self.allowed, 30)
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["args"], "echo hello")

    # ------------------------------------------------------------------
    # $VAR outside the body (in the in-clause words) expands
    # ------------------------------------------------------------------

    def test_var_in_in_words_expands(self) -> None:
        """$VAR in the in-clause word list expands from base_env."""
        calls = self._install_stubs()
        # Use a var that's in _base_env (e.g. USER)
        result = _try_for_loop(
            "for i in $USER; do echo $i; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        # One iteration with the current user's name
        self.assertEqual(len(calls), 1)

    # ------------------------------------------------------------------
    # for-loop must be the entire command (not mid-chain via ; / && / ||)
    # ------------------------------------------------------------------

    def test_for_not_at_start_falls_through(self) -> None:
        """'echo hi; for i in a; do echo $i; done' — for not at start → None.

        The scanner only detects for-loops that start at position 0.  When the
        command starts with something else, _try_for_loop returns None and the
        caller falls through to normal parsing.
        """
        result = _try_for_loop(
            "echo hi; for i in a; do echo $i; done",
            self.allowed, 30,
        )
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Regression: multi-line for-loop with newlines (finding 3 — DoS)
    # ------------------------------------------------------------------

    def test_multiline_newlines_between_in_words(self) -> None:
        """Newlines between in-words must not hang the scanner."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a\ndo echo $i\ndone",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["args"], "echo a")

    def test_multiline_newlines_before_do(self) -> None:
        """Newline before do must parse correctly."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a b\n\ndo echo $i\n\ndone",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 2)

    def test_multiline_newlines_between_in_words_with_semicolon(self) -> None:
        """Newlines with semicolon before do must parse."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in x y\n;\ndo echo $i; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 2)

    def test_multiline_newlines_in_body(self) -> None:
        """Newlines in the body must not cause issues — 'done' as plain word."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do\n  echo $i\n  echo done\ndone",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        # Body is 'echo $i\n  echo done' — one pipeline (newline is ws).
        # The second 'done' on its own line is the terminator.
        self.assertEqual(len(calls), 1)
        # Verify 'done' was a plain word argument (not terminator)
        self.assertIn("echo done", calls[0]["args"])

    # ------------------------------------------------------------------
    # Regression: 'done' as a plain word in the body (finding 1)
    # ------------------------------------------------------------------

    def test_done_as_plain_word_argument(self) -> None:
        """'echo done' in the body must not terminate the loop early."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do echo done; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)
        # The body should be "echo done" — 'done' is an argument to echo
        self.assertEqual(calls[0]["args"], "echo done")

    def test_done_done_as_plain_words(self) -> None:
        """'echo done done' — both 'done's are arguments, neither terminates."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do echo done done; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)
        # Both 'done' words are arguments to echo
        self.assertIn("done", calls[0]["args"])
        # The second 'done' after echo is also an argument, not terminator
        self.assertIn("done done", calls[0]["args"])

    def test_done_after_semicolon_in_body(self) -> None:
        """'done' after ';' in the body IS the terminator."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do echo hi; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["args"], "echo hi")

    def test_done_on_newline_in_body(self) -> None:
        """'done' on a line by itself in body is the terminator."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do echo hi\ndone",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["args"], "echo hi")

    # ------------------------------------------------------------------
    # Regression: heredoc body containing 'done' (finding 2)
    # ------------------------------------------------------------------

    def test_heredoc_containing_done_not_terminator(self) -> None:
        """'done' inside a heredoc body must NOT terminate the loop."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do cat <<EOF\ndone\nEOF\n; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        # Body is 'cat <<EOF\ndone\nEOF\n;' — one command with heredoc.
        # The 'done' inside the heredoc body is NOT the terminator.
        self.assertEqual(len(calls), 1)

    def test_heredoc_strip_tabs_containing_done_not_terminator(self) -> None:
        """'done' inside a <<- heredoc body must NOT terminate the loop."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do cat <<-EOF\n\tdone\n\tEOF\n; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)

    def test_heredoc_with_quoted_delim_containing_done(self) -> None:
        """'done' inside a heredoc with quoted delimiter must NOT terminate."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do cat <<'EOF'\ndone\nEOF\n; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)

    def test_herestring_containing_done(self) -> None:
        """'done' in a here-string word should NOT terminate the loop."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do cat <<< done; done",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)
        # 'done' is the here-string stdin word (redirect), not an arg
        self.assertIn("cat", calls[0]["args"])

    # ------------------------------------------------------------------
    # Regression: trailing content after 'done' (finding 4)
    # ------------------------------------------------------------------

    def test_trailing_content_after_done_rejected(self) -> None:
        """Commands after 'done' must cause the scanner to reject the loop."""
        result = _try_for_loop(
            "for i in a; do echo $i; done ; rm -rf /",
            self.allowed, 30,
        )
        # Must return None so the caller falls through to normal parsing,
        # which will reject 'for' as an unknown command.
        self.assertIsNone(result)

    def test_trailing_content_after_done_with_newline_rejected(self) -> None:
        """Trailing cmd after done with newline must be rejected."""
        result = _try_for_loop(
            "for i in a; do echo $i; done\necho bad",
            self.allowed, 30,
        )
        self.assertIsNone(result)

    def test_no_trailing_content_after_done_ok(self) -> None:
        """Whitespace-only after done is fine (no trailing commands)."""
        calls = self._install_stubs()
        result = _try_for_loop(
            "for i in a; do echo $i; done   ",
            self.allowed, 30,
        )
        self.assertIsNotNone(result)
        out, rc = result
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
