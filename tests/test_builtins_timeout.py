"""Tests for the timeout builtin shell_run behaviour. Run with the venv python that has `mcp` installed:

    PYTHONPATH=src <venv>/bin/python -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.parser import Word, WordPart
from shell_sandbox_mcp.server import (
    CommandNode,
    Expansion,
    _serialize_command,
    _apply_timeout_builtin,
    MAX_TIMEOUT,
)

# ---------------------------------------------------------------------------
# timeout builtin tests — stub _run_segment / _run_pipeline and record timeout
# ---------------------------------------------------------------------------


class ShellRunTimeoutTest(unittest.TestCase):
    """Exercise the per-pipeline ``timeout`` builtin (AST primary path).

    Stubs ``_run_segment`` and ``_run_pipeline`` to record the ``timeout``
    argument so we can assert ``timeout N`` overrides the default timeout.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-to-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = server._run_background
        self.segment_calls: list[tuple[str, int, str]] = []  # (cmd_str, timeout, work_dir)
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
        if isinstance(command, CommandNode):
            return _serialize_command(command)
        return str(command)

    def _stub_segments(self) -> None:
        rc_map = self.segment_rc_map

        def fake_segment(command, work_dir: Path, timeout: int,
                         expansion=None) -> tuple[int, str]:
            c = self._cmd_str(command)
            self.segment_calls.append((c, timeout, str(work_dir)))
            rc = rc_map.get(c, 0)
            return rc, f"out:{c}" if rc == 0 else f"err:{c}"

        server._run_segment = fake_segment

    def _stub_pipeline(self) -> None:
        pipeline_calls: list[tuple[list[str], int]] = []

        def fake_pipeline(segments, work_dir: Path, timeout: int,
                          expansion=None) -> tuple[int, str]:
            cmds = [self._cmd_str(s) for s in segments]
            pipeline_calls.append((cmds, timeout))
            return 0, "pipeline-ok"

        self._pipeline_calls = pipeline_calls
        server._run_pipeline = fake_pipeline

    def _stub_background(self) -> None:
        def fake_background(segments, work_dir: Path,
                            expansion=None) -> tuple[int, str, int]:
            return 0, "bg", 0

        server._run_background = fake_background

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.allowed))

    # ------------------------------------------------------------------
    # timeout N cmd  —  overrides segment timeout
    # ------------------------------------------------------------------

    def test_timeout_sets_segment_timeout(self) -> None:
        self._stub_segments()
        out = self._run("timeout 7 echo hi")
        self.assertEqual(len(self.segment_calls), 1)
        cmd_str, to_val, _wd = self.segment_calls[0]
        self.assertEqual(cmd_str, "echo hi")
        self.assertEqual(to_val, 7)
        self.assertIn("out:echo hi", out)

    # ------------------------------------------------------------------
    # clamping to MAX_TIMEOUT
    # ------------------------------------------------------------------

    def test_clamps_to_max_timeout(self) -> None:
        self._stub_segments()
        out = self._run("timeout 9999 echo hi")
        self.assertEqual(len(self.segment_calls), 1)
        cmd_str, to_val, _wd = self.segment_calls[0]
        self.assertEqual(cmd_str, "echo hi")
        self.assertEqual(to_val, MAX_TIMEOUT)
        self.assertIn("out:echo hi", out)

    # ------------------------------------------------------------------
    # reject zero
    # ------------------------------------------------------------------

    def test_rejects_zero(self) -> None:
        self._stub_segments()
        out = self._run("timeout 0 echo hi")
        self.assertIn("timeout: duration must be > 0", out)
        self.assertEqual(len(self.segment_calls), 0)

    # ------------------------------------------------------------------
    # reject negative
    # ------------------------------------------------------------------

    def test_rejects_negative(self) -> None:
        self._stub_segments()
        out = self._run("timeout -5 echo hi")
        self.assertIn("timeout: duration must be > 0", out)
        self.assertEqual(len(self.segment_calls), 0)

    # ------------------------------------------------------------------
    # reject non-number
    # ------------------------------------------------------------------

    def test_rejects_non_number(self) -> None:
        self._stub_segments()
        out = self._run("timeout abc echo hi")
        self.assertIn("timeout: invalid duration 'abc'", out)
        self.assertEqual(len(self.segment_calls), 0)

    # ------------------------------------------------------------------
    # missing command
    # ------------------------------------------------------------------

    def test_missing_command(self) -> None:
        self._stub_segments()
        out = self._run("timeout 5")
        self.assertIn("timeout: missing command", out)
        self.assertEqual(len(self.segment_calls), 0)

    # ------------------------------------------------------------------
    # missing duration
    # ------------------------------------------------------------------

    def test_missing_duration(self) -> None:
        self._stub_segments()
        out = self._run("timeout")
        self.assertIn("timeout: missing duration", out)
        self.assertEqual(len(self.segment_calls), 0)

    # ------------------------------------------------------------------
    # pipeline binds whole
    # ------------------------------------------------------------------

    def test_pipeline_binds_whole(self) -> None:
        self._stub_pipeline()
        out = self._run("timeout 3 a | b")
        self.assertTrue(hasattr(self, "_pipeline_calls"))
        self.assertEqual(len(self._pipeline_calls), 1)
        cmds, to_val = self._pipeline_calls[0]
        self.assertEqual(cmds, ["a", "b"])
        self.assertEqual(to_val, 3)
        self.assertIn("pipeline-ok", out)

    # ------------------------------------------------------------------
    # rejected in background
    # ------------------------------------------------------------------

    def test_rejected_in_background(self) -> None:
        self._stub_segments()
        self._stub_background()
        out = self._run("timeout 5 echo hi &")
        self.assertIn("timeout: not supported with background (&)", out)

    # ------------------------------------------------------------------
    # only at start of pipeline
    # ------------------------------------------------------------------

    def test_only_at_start_of_pipeline(self) -> None:
        """timeout on a non-first stage should fail with 'Command not allowed'."""
        self._stub_segments()
        out = self._run("echo hi | timeout 2 wc")
        # "timeout" is not in COMMANDS or BUSYBOX_APPLETS, so it'll be
        # 'Command not allowed' after the fallthrough
        self.assertIn("Command not allowed", out)

    # ------------------------------------------------------------------
    # timeout then cd still runs cd (timeout binds the pipeline,
    # cd is the command)
    # ------------------------------------------------------------------

    def test_timeout_then_cd_still_runs_cd(self) -> None:
        """timeout N cd sub — cd runs with overridden timeout (silent)."""
        out = self._run("timeout 5 cd sub")
        self.assertEqual(out, "(no output)")

    # ------------------------------------------------------------------
    # preserves redirects
    # ------------------------------------------------------------------

    def test_preserves_redirects(self) -> None:
        """timeout 2 ls > /tmp/out.txt should work (redirect preserved)."""
        self._stub_segments()
        out = self._run("timeout 2 ls > /tmp/out.txt")
        self.assertEqual(len(self.segment_calls), 1)
        cmd_str, to_val, _wd = self.segment_calls[0]
        # The redirect target should be part of the command display
        self.assertIn("ls", cmd_str)
        self.assertEqual(to_val, 2)

    # ------------------------------------------------------------------
    # not keyword mid-word: echo timeout unaffected
    # ------------------------------------------------------------------

    def test_not_keyword_mid_word(self) -> None:
        """echo timeout should NOT trigger the timeout builtin."""
        self._stub_segments()
        out = self._run("echo timeout hi")
        self.assertEqual(len(self.segment_calls), 1)
        cmd_str, to_val, _wd = self.segment_calls[0]
        self.assertEqual(cmd_str, "echo timeout hi")
        # Default timeout unchanged
        self.assertEqual(to_val, 30)
        self.assertIn("out:echo timeout hi", out)

    # ------------------------------------------------------------------
    # no timeout fallthrough: original timeout param passed unchanged
    # ------------------------------------------------------------------

    def test_no_timeout_fallthrough(self) -> None:
        """When no 'timeout' token, default timeout is passed unchanged."""
        self._stub_segments()
        out = self._run("echo hello")
        self.assertEqual(len(self.segment_calls), 1)
        _cmd_str, to_val, _wd = self.segment_calls[0]
        self.assertEqual(to_val, 30)  # DEFAULT_TIMEOUT
        self.assertIn("out:echo hello", out)


# ---------------------------------------------------------------------------
# _apply_timeout_builtin unit tests (direct function calls)
# ---------------------------------------------------------------------------


class ApplyTimeoutBuiltinUnitTest(unittest.TestCase):
    """Unit tests for the _apply_timeout_builtin function directly."""

    def _make_cmd(self, *words: str, backgrounded: bool = False) -> CommandNode:
        """Build a CommandNode with proper Word/WordPart objects."""
        word_objs = tuple(Word(parts=(WordPart(text=w, raw=w),)) for w in words)
        return CommandNode(words=word_objs, redirects=(), backgrounded=backgrounded)

    def test_fallthrough_when_not_timeout(self) -> None:
        nodes = [self._make_cmd("echo", "hi")]
        result, eff_to, err = _apply_timeout_builtin(nodes, None, False, 30)
        self.assertIs(err, None)
        self.assertEqual(eff_to, 30)
        self.assertEqual(result, nodes)  # unchanged

    def test_timeout_strips_first_two_words(self) -> None:
        nodes = [self._make_cmd("timeout", "5", "echo", "hi")]
        result, eff_to, err = _apply_timeout_builtin(nodes, None, False, 30)
        self.assertIs(err, None)
        self.assertEqual(eff_to, 5)
        self.assertEqual(len(result), 1)
        # Compare word texts, not the Word/WordPart objects
        self.assertEqual(tuple(w.text for w in result[0].words), ("echo", "hi"))

    def test_empty_nodes(self) -> None:
        result, eff_to, err = _apply_timeout_builtin([], None, False, 30)
        self.assertIs(err, None)
        self.assertEqual(eff_to, 30)
        self.assertEqual(result, [])

    def test_clamp_to_max(self) -> None:
        nodes = [self._make_cmd("timeout", "99999", "echo", "hi")]
        result, eff_to, err = _apply_timeout_builtin(nodes, None, False, 30)
        self.assertIs(err, None)
        self.assertEqual(eff_to, MAX_TIMEOUT)

    def test_reject_background(self) -> None:
        nodes = [self._make_cmd("timeout", "5", "echo", "hi")]
        result, eff_to, err = _apply_timeout_builtin(nodes, None, True, 30)
        self.assertIsNotNone(err)
        self.assertIn("background", err)
        self.assertIsNone(result)
        self.assertIsNone(eff_to)


if __name__ == "__main__":
    unittest.main()
