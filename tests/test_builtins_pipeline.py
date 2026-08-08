"""Tests for per-stage builtin handling in multi-stage pipelines.

Run with:

    PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.builtins import _classify_builtin, _BUILTIN_STAGE_NAMES
from shell_sandbox_mcp.parser import CommandNode
from shell_sandbox_mcp.variables import VariableStore


# ============================================================================
# _classify_builtin unit tests
# ============================================================================


class ClassifyBuiltinTest(unittest.TestCase):
    """Unit tests for _classify_builtin on various CommandNode inputs."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _cmd_node(self, text: str) -> CommandNode:
        from shell_sandbox_mcp.parser import program_to_chain
        from shell_sandbox_mcp.server import _expand_command
        _, expansion, program = _expand_command(text, self.work_dir, 30, 0)
        self.assertIsNotNone(program, f"Parse failed for: {text}")
        chains = program_to_chain(program)
        self.assertTrue(chains)
        return chains[0][1][0]

    def test_export_positive(self) -> None:
        node = self._cmd_node("export FOO=bar")
        self.assertEqual(_classify_builtin(node, None, self.work_dir), "export")

    def test_unset_positive(self) -> None:
        node = self._cmd_node("unset FOO")
        self.assertEqual(_classify_builtin(node, None, self.work_dir), "unset")

    def test_set_positive(self) -> None:
        node = self._cmd_node("set FOO=bar")
        self.assertEqual(_classify_builtin(node, None, self.work_dir), "set")

    def test_shift_positive(self) -> None:
        node = self._cmd_node("shift")
        self.assertEqual(_classify_builtin(node, None, self.work_dir), "shift")

    def test_source_positive(self) -> None:
        node = self._cmd_node("source file.sh")
        self.assertEqual(_classify_builtin(node, None, self.work_dir), "source")

    def test_dot_positive(self) -> None:
        node = self._cmd_node(". file.sh")
        self.assertEqual(_classify_builtin(node, None, self.work_dir), ".")

    def test_echo_negative(self) -> None:
        node = self._cmd_node("echo export")
        self.assertIsNone(_classify_builtin(node, None, self.work_dir))

    def test_cd_negative(self) -> None:
        node = self._cmd_node("cd /tmp")
        self.assertIsNone(_classify_builtin(node, None, self.work_dir))

    def test_cat_negative(self) -> None:
        node = self._cmd_node("cat file")
        self.assertIsNone(_classify_builtin(node, None, self.work_dir))

    def test_empty_negative(self) -> None:
        """Empty command produces no args → _classify_builtin returns None."""
        from shell_sandbox_mcp.parser import CommandNode
        node = CommandNode(words=())
        self.assertIsNone(_classify_builtin(node, None, self.work_dir))

    def test_export_with_redirect(self) -> None:
        """export > f should still classify as 'export' (redirects ignored)."""
        node = self._cmd_node("export > f")
        self.assertEqual(_classify_builtin(node, None, self.work_dir), "export")

    def test_builtin_stage_names_constant(self) -> None:
        """Verify _BUILTIN_STAGE_NAMES contains the expected 6 names."""
        self.assertEqual(
            set(_BUILTIN_STAGE_NAMES),
            {"export", "unset", "set", "shift", "source", "."},
        )


# ============================================================================
# Mixed-pipeline end-to-end tests via shell_run (stubbed execution)
# ============================================================================


class MixedPipelineTest(unittest.TestCase):
    """End-to-end mixed-pipeline tests via shell_run with stubbed execution.

    Stubs _run_pipeline_core to capture injected_first_stdin_bytes and
    start_index, and _build_invocation to avoid real sandbox resolution.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-mix-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_pipeline_core = server._run_pipeline_core
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = server._run_background
        self._orig_build = server._build_invocation
        self.pipeline_calls: list[dict] = []

    def tearDown(self) -> None:
        import shutil
        server._run_pipeline_core = self._orig_pipeline_core
        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        server._run_background = self._orig_background
        server._build_invocation = self._orig_build
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def _stub_execution(self):
        """Stub _run_pipeline_core, _run_segment, _run_pipeline, _run_background,
        and _build_invocation to capture args and avoid real subprocesses."""
        calls = self.pipeline_calls

        def fake_pipeline_core(segments, work_dir, timeout, expansion=None,
                                *, shell_env=None, stage_env_overrides=None,
                                injected_first_stdin_bytes=None, start_index=0):
            calls.append({
                "segments": segments,
                "injected_first_stdin_bytes": injected_first_stdin_bytes,
                "start_index": start_index,
                "shell_env": shell_env,
                "stage_env_overrides": stage_env_overrides,
            })
            if injected_first_stdin_bytes:
                return 0, injected_first_stdin_bytes, b"", []
            return 0, b"ok", b"", []

        server._run_pipeline_core = fake_pipeline_core

        def fake_segment(command, work_dir, timeout, expansion=None,
                          *, shell_env=None, stage_env_overrides=None):
            calls.append({
                "command": command,
                "shell_env": shell_env,
                "stage_env_overrides": stage_env_overrides,
            })
            return 0, "ok"

        server._run_segment = fake_segment

        def fake_pipeline(segments, work_dir, timeout, expansion=None,
                           *, shell_env=None, stage_env_overrides=None,
                           injected_first_stdin_bytes=None, start_index=0):
            calls.append({
                "segments": segments,
                "shell_env": shell_env,
                "stage_env_overrides": stage_env_overrides,
                "injected_first_stdin_bytes": injected_first_stdin_bytes,
                "start_index": start_index,
            })
            return 0, "pipeline-ok"

        server._run_pipeline = fake_pipeline

        def fake_background(segments, work_dir, expansion=None,
                            *, shell_env=None, stage_env_overrides=None):
            calls.append({"segments": segments, "bg": True})
            return 0, "bg", 0

        server._run_background = fake_background

        from shell_sandbox_mcp.executor import Invocation

        def fake_build(command, work_dir, expansion=None, *, shell_env=None):
            return Invocation(
                binary="/bin/echo",
                sandbox_args=["/bin/echo"],
                env=shell_env,
                cfg={},
                redirects=[],
            )

        server._build_invocation = fake_build

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.allowed))

    # ------------------------------------------------------------------
    # export X=1 | echo hi
    # ------------------------------------------------------------------

    def test_export_pipes_to_echo(self) -> None:
        """export X=1 | echo hi → echo mini-pipeline gets empty stdin,
        start_index=1."""
        self._stub_execution()
        out = self._run("export X=1 | echo hi")
        found = False
        for c in self.pipeline_calls:
            if c.get("start_index") == 1:
                self.assertEqual(c.get("injected_first_stdin_bytes"), b"")
                found = True
                break
        self.assertTrue(found,
                        f"No subprocess call with start_index=1: {self.pipeline_calls}")

    # ------------------------------------------------------------------
    # source s.sh | wc -l  (stubbed execution)
    # ------------------------------------------------------------------

    def test_source_pipes_to_wc(self) -> None:
        """source s.sh | wc -l → wc mini-pipeline gets source output as stdin."""
        script = self.allowed / "s.sh"
        script.write_text("echo a\necho b\necho c\n")
        self._stub_execution()
        out = self._run(f"source s.sh | wc -l")
        found = False
        for c in self.pipeline_calls:
            if c.get("start_index") == 1:
                injected = c.get("injected_first_stdin_bytes")
                self.assertIsNotNone(injected)
                self.assertIn(b"ok", injected)
                found = True
                break
        self.assertTrue(found,
                        f"No subprocess call with start_index=1: {self.pipeline_calls}")

    # ------------------------------------------------------------------
    # export | cat
    # ------------------------------------------------------------------

    def test_export_no_args_pipes_to_cat(self) -> None:
        """export | cat → cat gets sorted exported-vars text as stdin."""
        self._stub_execution()
        out = self._run("export | cat")
        for c in self.pipeline_calls:
            if c.get("start_index") == 1:
                injected = c.get("injected_first_stdin_bytes")
                self.assertIsNotNone(injected)
                self.assertIn(b"PATH=", injected)

    # ------------------------------------------------------------------
    # cmd1 | set A=x | cmd2
    # ------------------------------------------------------------------

    def test_cmd_set_cmd_chain(self) -> None:
        """echo hello | set A=x | cat → cat got empty stdin (start_index=2)."""
        self._stub_execution()
        out = self._run("echo hello | set A=x | cat")
        for c in self.pipeline_calls:
            if c.get("start_index") == 2:
                injected = c.get("injected_first_stdin_bytes")
                self.assertEqual(injected, b"")

    # ------------------------------------------------------------------
    # set A=x | set B=y
    # ------------------------------------------------------------------

    def test_set_pipes_to_set(self) -> None:
        """set A=x | set B=y → both builtins, no subprocess; output empty."""
        self._stub_execution()
        out = self._run("set A=x | set B=y")
        self.assertIn("(no output)", out)

    # ------------------------------------------------------------------
    # false | export X=1
    # ------------------------------------------------------------------

    def test_false_pipes_to_export(self) -> None:
        """false | export X=1 → rc 0 (last stage is export, which succeeds)."""
        self._stub_execution()
        out = self._run("false | export X=1")
        # export with args produces no stdout → builtin-last output empty
        self.assertIn("(no output)", out)

    # ------------------------------------------------------------------
    # source s.sh & and export X=1 & → rejected
    # ------------------------------------------------------------------

    def test_source_background_rejected(self) -> None:
        """source s.sh & → rejected with 'builtin not supported in
        backgrounded pipeline (&)', rc 1."""
        script = self.allowed / "s2.sh"
        script.write_text("echo bg")
        out = self._run("source s2.sh &")
        self.assertIn("builtin not supported in backgrounded pipeline", out)

    def test_export_background_rejected(self) -> None:
        """export X=1 & → rejected, rc 1."""
        out = self._run("export X=1 &")
        self.assertIn("builtin not supported in backgrounded pipeline", out)

    # ------------------------------------------------------------------
    # cd /tmp | echo hi → still rejected
    # ------------------------------------------------------------------

    def test_cd_in_pipeline_still_rejected(self) -> None:
        """cd /tmp | echo hi → still 'Command not allowed: cd'."""
        out = self._run("cd /tmp | echo hi")
        self.assertIn("not allowed", out.lower())

    # ------------------------------------------------------------------
    # timeout 5 source s.sh | wc -l
    # ------------------------------------------------------------------

    def test_timeout_with_source_pipeline(self) -> None:
        """timeout 5 source s.sh | wc -l → timeout stripped, source output
        feeds wc."""
        script = self.allowed / "t.sh"
        script.write_text("echo one\necho two\n")
        self._stub_execution()
        out = self._run("timeout 5 source t.sh | wc -l")
        for c in self.pipeline_calls:
            if c.get("start_index") == 1:
                injected = c.get("injected_first_stdin_bytes")
                self.assertIsNotNone(injected)
                self.assertIn(b"ok", injected)

    def test_timeout_export_pipeline(self) -> None:
        """timeout 3 export X=1 | cat → timeout stripped, export builtin
        resolves and feeds cat (start_index=1 subprocess)."""
        self._stub_execution()
        out = self._run("timeout 3 export X=1 | cat")
        found = False
        for c in self.pipeline_calls:
            if c.get("start_index") == 1:
                found = True
                break
        self.assertTrue(found,
                        f"No subprocess call with start_index=1: {self.pipeline_calls}")

    def test_timeout_pure_subprocess_pipeline_fastpath(self) -> None:
        """timeout 3 echo hi | wc -l → pure subprocess fast path preserved."""
        self._stub_execution()
        out = self._run("timeout 3 echo hi | wc -l")
        self.assertIn("pipeline-ok", out.lower())

    # ------------------------------------------------------------------
    # X=1 export Y=2 | echo hi
    # ------------------------------------------------------------------

    def test_env_prefix_export_pipeline(self) -> None:
        """X=1 export Y=2 | echo hi → mixed pipeline runs."""
        self._stub_execution()
        out = self._run("X=1 export Y=2 | echo hi")
        found = False
        for c in self.pipeline_calls:
            if c.get("start_index") == 1:
                found = True
                break
        self.assertTrue(found,
                        f"No subprocess call with start_index=1: {self.pipeline_calls}")

    # ------------------------------------------------------------------
    # Containment: source /etc/passwd | wc -l
    # ------------------------------------------------------------------

    def test_source_escapes_sandbox_in_pipeline(self) -> None:
        """source /etc/passwd | wc -l → 'escapes sandbox'."""
        self._stub_execution()
        out = self._run("source /etc/passwd | wc -l")
        self.assertIn("escapes sandbox", out)

    # ------------------------------------------------------------------
    # Byte-for-byte regression: single-stage builtins unchanged
    # ------------------------------------------------------------------

    def test_single_stage_export_unchanged(self) -> None:
        """Single-stage export should still work (byte-for-byte)."""
        self._stub_execution()
        out = self._run("export FOO=bar")
        self.assertEqual(out, "(no output)")

    def test_single_stage_echo_unchanged(self) -> None:
        """Single-stage echo should still work."""
        self._stub_execution()
        out = self._run("echo hello")
        self.assertIn("ok", out)

    # ------------------------------------------------------------------
    # Pure subprocess pipeline unchanged
    # ------------------------------------------------------------------

    def test_pure_subprocess_pipeline_unchanged(self) -> None:
        """A pipeline with no builtins should still work normally."""
        self._stub_execution()
        out = self._run("echo hello | cat")
        self.assertIn("pipeline-ok", out.lower())


if __name__ == "__main__":
    unittest.main()
