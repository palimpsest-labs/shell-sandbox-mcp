"""Tests for break / continue builtins (Phase D)."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.parser import CommandNode
from shell_sandbox_mcp.server import _extract_redirects


def _install_stubs() -> list[dict]:
    """Stub _run_segment/_run_pipeline to record resolved command args."""
    calls: list[dict] = []

    def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
        if isinstance(command, CommandNode):
            args, _, _ = _extract_redirects(command, expansion, work_dir)
            cmd_str = " ".join(args) if args else "<empty>"
        else:
            cmd_str = str(command)
        calls.append({"args": cmd_str, "work_dir": str(work_dir)})
        return 0, ""

    server._run_segment = fake_segment

    def fake_pipeline(segments, work_dir, timeout, expansion=None, **kwargs):
        str_segs = []
        for s in segments:
            if isinstance(s, CommandNode):
                args, _, _ = _extract_redirects(s, expansion, work_dir)
                str_segs.append(" ".join(args) if args else "<empty>")
            else:
                str_segs.append(str(s))
        calls.append({"args": " | ".join(str_segs), "work_dir": str(work_dir)})
        return 0, ""

    server._run_pipeline = fake_pipeline
    return calls


def _remove_stubs(orig_segment, orig_pipeline) -> None:
    server._run_segment = orig_segment
    server._run_pipeline = orig_pipeline


class BreakContinueTest(unittest.TestCase):
    """Tests for break / continue in for/while loops."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-brk-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = server._run_background
        server._SESSION_FUNCTIONS.clear()

    def tearDown(self) -> None:
        _remove_stubs(self._orig_segment, self._orig_pipeline)
        server._run_background = self._orig_background
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()
        server._SESSION_FUNCTIONS.clear()

    # ------------------------------------------------------------------
    # break
    # ------------------------------------------------------------------

    def test_break_exits_while_loop(self) -> None:
        """break should exit the innermost while loop."""
        calls = _install_stubs()
        try:
            server.shell_run(
                "while true; do echo a; break; echo b; done",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
        # Should have echo a, but NOT echo b (break skipped it)
        self.assertTrue(any("echo a" in c.get("args", "") for c in calls))
        self.assertFalse(any("echo b" in c.get("args", "") for c in calls))

    def test_break_exits_for_loop(self) -> None:
        """break should exit the innermost for loop."""
        calls = _install_stubs()
        try:
            server.shell_run(
                "for x in 1 2 3; do echo $x; break; echo after; done",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
        # Should have echo 1 only (first iteration, then break)
        self.assertTrue(any("echo 1" in c.get("args", "") for c in calls))
        self.assertFalse(any("echo 2" in c.get("args", "") for c in calls))
        self.assertFalse(any("echo after" in c.get("args", "") for c in calls))

    def test_break_2_exits_two_loops(self) -> None:
        """break 2 should exit two nested loops."""
        calls = _install_stubs()
        try:
            server.shell_run(
                "for x in 1; do for y in a; do break 2; done; echo outer; done",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
        # Should NOT have echo outer (break 2 exited both loops)
        self.assertFalse(any("echo outer" in c.get("args", "") for c in calls))

    def test_break_outside_loop(self) -> None:
        """break outside a loop should emit diagnostic with rc=1."""
        result = server.shell_run(
            "break",
            cwd=str(self.allowed),
            timeout=30,
        )
        self.assertIn("only meaningful", result)

    def test_break_0_returns_error(self) -> None:
        """break 0 should produce rc=1 with diagnostic."""
        result = server.shell_run(
            "for x in 1; do break 0; done",
            cwd=str(self.allowed),
            timeout=30,
        )
        self.assertIn("invalid argument", result)

    def test_break_abc_returns_error(self) -> None:
        """break abc should produce rc=1 with diagnostic."""
        result = server.shell_run(
            "for x in 1; do break abc; done",
            cwd=str(self.allowed),
            timeout=30,
        )
        self.assertIn("invalid argument", result)

    # ------------------------------------------------------------------
    # continue
    # ------------------------------------------------------------------

    def test_continue_skips_iteration(self) -> None:
        """continue should skip the rest of the current iteration."""
        calls = _install_stubs()
        try:
            server.shell_run(
                "for x in 1 2; do echo a; continue; echo b; done",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
        # echo a should appear twice, echo b never
        count_a = sum(1 for c in calls if "echo a" in c.get("args", ""))
        self.assertEqual(count_a, 2, f"Expected 2 echo a, got {calls}")
        self.assertFalse(any("echo b" in c.get("args", "") for c in calls))

    def test_continue_2_skips_two_levels(self) -> None:
        """continue 2 should skip to next iteration of outer loop."""
        calls = _install_stubs()
        try:
            server.shell_run(
                "for x in 1; do for y in a; do continue 2; echo inner; done; echo outer; done",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
        # echo inner should NOT appear; echo outer SHOULD (continue 2 skips inner, goes to next outer iter)
        self.assertFalse(any("echo inner" in c.get("args", "") for c in calls))
        # Outer loop only has 1 iteration, so echo outer should appear 0 times
        # (continue 2 goes to next iteration of outer loop, but there's only 1)

    def test_continue_outside_loop(self) -> None:
        """continue outside a loop should emit diagnostic with rc=1."""
        result = server.shell_run(
            "continue",
            cwd=str(self.allowed),
            timeout=30,
        )
        self.assertIn("only meaningful", result)

    def test_continue_2_in_while_body(self) -> None:
        """continue 2 in a while body should skip to next iteration of outer loop."""
        calls = _install_stubs()
        try:
            server.shell_run(
                "for x in 1 2; do while true; do continue 2; echo inner; done; echo outer; done",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
        # echo inner should never appear (continue 2 skips it).
        self.assertFalse(any("echo inner" in c.get("args", "") for c in calls))
        # echo outer is also skipped: continue 2 sends control to the
        # for-loop's next iteration, which jumps over echo outer too.
        self.assertFalse(any("echo outer" in c.get("args", "") for c in calls))
        # true is called twice (once per iteration before continue 2 fires).
        count_true = sum(1 for c in calls if "true" in c.get("args", ""))
        self.assertEqual(count_true, 2, f"Expected 2 true calls, got {calls}")

    # ------------------------------------------------------------------
    # B1 regression: break N / continue N with N > nesting depth
    # ------------------------------------------------------------------

    def test_break_2_beyond_single_for_loop(self) -> None:
        """break 2 in a single for loop emits diagnostic, rc=1, no crash."""
        result = server.shell_run(
            "for x in 1; do break 2; done",
            cwd=str(self.allowed),
            timeout=30,
            structured=True,
        )
        self.assertEqual(result["rc"], 1)
        self.assertIn("only meaningful", result["output"])
        self.assertIn("break", result["output"])

    def test_break_3_beyond_single_for_loop(self) -> None:
        """break 3 in a single for loop emits diagnostic, rc=1, no crash."""
        result = server.shell_run(
            "for x in 1; do break 3; done",
            cwd=str(self.allowed),
            timeout=30,
            structured=True,
        )
        self.assertEqual(result["rc"], 1)
        self.assertIn("only meaningful", result["output"])
        self.assertIn("break", result["output"])

    def test_break_2_beyond_single_while_loop(self) -> None:
        """break 2 in a single while loop emits diagnostic, rc=1, no crash."""
        result = server.shell_run(
            "x=1; while x=1; do break 2; done",
            cwd=str(self.allowed),
            timeout=30,
            structured=True,
        )
        self.assertEqual(result["rc"], 1)
        self.assertIn("only meaningful", result["output"])
        self.assertIn("break", result["output"])

    def test_break_2_in_if_inside_for_loop(self) -> None:
        """break 2 in an if condition inside a single for loop: rc=1, no crash."""
        result = server.shell_run(
            "for x in 1; do if break 2; then echo hi; fi; echo after; done",
            cwd=str(self.allowed),
            timeout=30,
            structured=True,
        )
        self.assertEqual(result["rc"], 1)
        self.assertIn("only meaningful", result["output"])
        self.assertIn("break", result["output"])

    # ------------------------------------------------------------------
    # B2 regression: break / continue inside a while/until CONDITION
    # ------------------------------------------------------------------

    def test_break_in_while_condition(self) -> None:
        """break in while condition exits loop cleanly, no crash."""
        result = server.shell_run(
            "while break; do echo body; done",
            cwd=str(self.allowed),
            timeout=30,
            structured=True,
        )
        self.assertEqual(result["rc"], 0)
        self.assertNotIn("body", result["output"])

    def test_continue_in_while_condition(self) -> None:
        """continue in while condition does not crash (hits MAX_LOOP_ITER)."""
        result = server.shell_run(
            "while continue; do echo body; done",
            cwd=str(self.allowed),
            timeout=30,
            structured=True,
        )
        # Should not crash — either hits MAX_LOOP_ITER (rc=1) or completes
        self.assertIsNotNone(result["rc"])
        self.assertNotIn("body", result["output"])

    def test_break_2_in_while_condition_nested(self) -> None:
        """break 2 in while condition inside for loop exits both, no crash."""
        result = server.shell_run(
            "for x in 1 2; do while break 2; do echo inner; done; echo outer; done",
            cwd=str(self.allowed),
            timeout=30,
            structured=True,
        )
        self.assertEqual(result["rc"], 0)
        self.assertNotIn("inner", result["output"])
        self.assertNotIn("outer", result["output"])
