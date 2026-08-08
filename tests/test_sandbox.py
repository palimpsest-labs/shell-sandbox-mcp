"""Tests for shell_sandbox_mcp.server security-critical helpers. Run with the venv python that has `mcp` installed:

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
        def fake(command, work_dir: Path, timeout: int, expansion=None) -> tuple[int, str]:
            # command may be a str (legacy) or CommandNode (AST-native)
            cmd_str = (
                _serialize_command(command)
                if isinstance(command, CommandNode)
                else command
            )
            self.calls.append(cmd_str)
            rc = rc_map.get(cmd_str, 0)
            return rc, f"out:{cmd_str}" if rc == 0 else f"err:{cmd_str}"

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
# shell_run pipeline orchestration
# ---------------------------------------------------------------------------


class ShellRunPipelineTest(unittest.TestCase):
    """Exercise how `shell_run` routes pipe pipelines to `_run_pipeline`,
    and applies `&&`/`||` short-circuit to a pipeline's exit code, by stubbing
    `_run_segment`, `_run_pipeline`, and `_run_background`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-pipe-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = getattr(server, "_run_background", None)
        self.segment_calls: list[str] = []
        self.pipeline_calls: list[list[str]] = []
        self.background_calls: list[list[str]] = []

    def tearDown(self) -> None:
        import shutil

        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        if self._orig_background is not None:
            server._run_background = self._orig_background
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def _stub(
        self,
        pipeline_rc: dict[tuple[str, ...], int] | None = None,
        segment_rc: dict[str, int] | None = None,
        background_rc: dict[tuple[str, ...], int] | None = None,
    ) -> None:
        pipeline_rc = pipeline_rc or {}
        segment_rc = segment_rc or {}
        background_rc = background_rc or {}

        def fake_pipeline(stages, work_dir, timeout, expansion=None):
            # stages may be list[str] (legacy) or list[CommandNode] (AST-native)
            str_stages = [
                _serialize_command(s) if isinstance(s, CommandNode) else s
                for s in stages
            ]
            self.pipeline_calls.append(str_stages)
            rc = pipeline_rc.get(tuple(str_stages), 0)
            return rc, f"pipe:{'|'.join(str_stages)}" if rc == 0 else f"err-pipe:{'|'.join(str_stages)}"

        def fake_segment(command, work_dir, timeout, expansion=None):
            # command may be a str (legacy) or CommandNode (AST-native)
            cmd_str = (
                _serialize_command(command)
                if isinstance(command, CommandNode)
                else command
            )
            self.segment_calls.append(cmd_str)
            rc = segment_rc.get(cmd_str, 0)
            return rc, f"out:{cmd_str}" if rc == 0 else f"err:{cmd_str}"

        def fake_background(stages, work_dir, expansion=None):
            # stages may be list[str] (legacy) or list[CommandNode] (AST-native)
            str_stages = [
                _serialize_command(s) if isinstance(s, CommandNode) else s
                for s in stages
            ]
            self.background_calls.append(str_stages)
            rc = background_rc.get(tuple(str_stages), 0)
            return rc, f"bg:{'|'.join(str_stages)}", 0

        server._run_pipeline = fake_pipeline
        server._run_segment = fake_segment
        server._run_background = fake_background

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.allowed))

    def test_two_stage_pipe_routes_to_pipeline(self) -> None:
        self._stub()
        out = self._run("ls | wc")
        self.assertEqual(self.pipeline_calls, [["ls", "wc"]])
        self.assertEqual(self.segment_calls, [])
        self.assertIn("pipe:ls|wc", out)

    def test_three_stage_pipe(self) -> None:
        self._stub()
        self._run("a | b | c")
        self.assertEqual(self.pipeline_calls, [["a", "b", "c"]])

    def test_single_stage_uses_segment(self) -> None:
        self._stub()
        self._run("ls")
        self.assertEqual(self.segment_calls, ["ls"])
        self.assertEqual(self.pipeline_calls, [])

    def test_andand_skips_pipeline_after_failure(self) -> None:
        self._stub(pipeline_rc={("a", "b"): 1}, segment_rc={"c": 0})
        out = self._run("a | b && c")
        self.assertEqual(self.pipeline_calls, [["a", "b"]])
        self.assertEqual(self.segment_calls, [])
        self.assertIn("skipped", out)
        self.assertNotIn("out:c", out)

    def test_andand_runs_after_pipeline_success(self) -> None:
        self._stub(pipeline_rc={("a", "b"): 0}, segment_rc={"c": 0})
        out = self._run("a | b && c")
        self.assertEqual(self.pipeline_calls, [["a", "b"]])
        self.assertEqual(self.segment_calls, ["c"])
        self.assertIn("out:c", out)

    def test_oror_runs_after_pipeline_failure(self) -> None:
        self._stub(pipeline_rc={("a", "b"): 1}, segment_rc={"c": 0})
        self._run("a | b || c")
        self.assertEqual(self.pipeline_calls, [["a", "b"]])
        self.assertEqual(self.segment_calls, ["c"])

    def test_oror_skips_after_pipeline_success(self) -> None:
        self._stub(pipeline_rc={("a", "b"): 0}, segment_rc={"c": 0})
        out = self._run("a | b || c")
        self.assertEqual(self.pipeline_calls, [["a", "b"]])
        self.assertEqual(self.segment_calls, [])
        self.assertIn("skipped", out)

    def test_pipeline_resolution_failure_short_circuits_andand(self) -> None:
        # A denied command inside a pipeline surfaces as rc 1 from _run_pipeline.
        self._stub(pipeline_rc={("nope", "b"): 1}, segment_rc={"c": 0})
        out = self._run("nope | b && c")
        self.assertEqual(self.pipeline_calls, [["nope", "b"]])
        self.assertEqual(self.segment_calls, [])
        self.assertIn("skipped", out)

    def test_bare_ampersand_returns_immediately_via_background(self) -> None:
        self._stub()
        out = self._run("echo hi & ls")
        # 'echo hi' is backgrounded, 'ls' is a normal segment
        self.assertEqual(self.background_calls, [["echo hi"]])
        self.assertEqual(self.segment_calls, ["ls"])
        self.assertEqual(self.pipeline_calls, [])
        self.assertIn("bg:echo hi", out)
        self.assertIn("out:ls", out)



if __name__ == "__main__":
    unittest.main()
