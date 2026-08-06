"""Tests for pipeline/expand/capture/fd-resolution executor helpers. Run with the venv python that has `mcp` installed:

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
# _run_pipeline real-subprocess orchestration
# ---------------------------------------------------------------------------


class RunPipelineIntegrationTest(unittest.TestCase):
    """Drive `_run_pipeline` with real subprocesses to exercise the Popen
    chaining, stderr-draining threads, reaping, and timeout paths. The sandbox
    wrapper is bypassed by stubbing `_build_invocation` to emit plain system
    commands, so the orchestration logic is what's under test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)
        self._orig_build = server._build_invocation

    def tearDown(self) -> None:
        server._build_invocation = self._orig_build
        self._tmp.cleanup()

    def _fake_build(self, mapping: dict[str, server.Invocation]):
        def fake(command: str, work_dir, expansion=None):
            return mapping.get(command, server.EmptyInvocation())

        server._build_invocation = fake

    def test_real_two_stage_pipe(self) -> None:
        self._fake_build({
            "producer": server.Invocation("/bin/echo", ["/bin/echo", "hello"], None, {}, []),
            "consumer": server.Invocation("/usr/bin/wc", ["/usr/bin/wc", "-c"], None, {}, []),
        })
        rc, out = server._run_pipeline(["producer", "consumer"], self.work_dir, 10)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "6")  # "hello\n" is 6 bytes

    def test_upstream_keeps_running_is_reaped(self) -> None:
        # The last stage (head) exits immediately, but the producer loops
        # writing to stderr forever. The orchestration must kill+reap the
        # upstream stage and still return promptly without hanging or racing
        # on the stderr buffer.
        self._fake_build({
            "producer": server.Invocation(
                "/bin/sh",
                ["/bin/sh", "-c", "echo out; while true; do echo err >&2; done"],
                None,
                {},
                [],
            ),
            "consumer": server.Invocation("/usr/bin/head", ["/usr/bin/head", "-n1"], None, {}, []),
        })
        rc, out = server._run_pipeline(["producer", "consumer"], self.work_dir, 10)
        self.assertEqual(rc, 0)
        self.assertIn("out", out)
        self.assertIn("[stderr]", out)

    def test_pipeline_timeout_kills_stages(self) -> None:
        # A last stage that never exits must be killed and reported as a
        # timeout rather than hanging the tool.
        self._fake_build({
            "producer": server.Invocation("/bin/echo", ["/bin/echo", "hi"], None, {}, []),
            "consumer": server.Invocation(
                "/bin/sh",
                ["/bin/sh", "-c", "while true; do :; done"],
                None,
                {},
                [],
            ),
        })
        rc, out = server._run_pipeline(["producer", "consumer"], self.work_dir, 1)
        self.assertEqual(rc, 1)
        self.assertIn("timed out", out)

    def test_three_stage_real_pipe(self) -> None:
        import os as _os

        grep = "/usr/bin/grep" if _os.path.exists("/usr/bin/grep") else "/bin/grep"
        self._fake_build({
            "a": server.Invocation("/bin/echo", ["/bin/echo", "one\ntwo\nthree"], None, {}, []),
            "b": server.Invocation(grep, [grep, "two"], None, {}, []),
            "c": server.Invocation("/usr/bin/wc", ["/usr/bin/wc", "-l"], None, {}, []),
        })
        rc, out = server._run_pipeline(["a", "b", "c"], self.work_dir, 10)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "1")

class ExpandCommandTest(unittest.TestCase):
    """Test _expand_command with stubbed _capture_stdout."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)
        self._orig_capture = server._capture_stdout
        self.captures: list[str] = []

    def tearDown(self) -> None:
        server._capture_stdout = self._orig_capture
        self._tmp.cleanup()

    def _stub_capture(self, outputs: dict[str, str]) -> None:
        def fake(command, work_dir, timeout, depth, deadline=None, subst_count=None, env=None):
            self.captures.append(command)
            val = outputs.get(command, "")
            return 0, val.encode("utf-8")

        server._capture_stdout = fake

    def test_unquoted_heredoc(self) -> None:
        cmd = "cat <<EOF\nhello\nworld\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        # Should contain << + sentinel
        self.assertIn("<<", expanded)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn(sentinel, exp.heredoc_bodies)
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\nworld\n")

    def test_single_quoted_delimiter_no_expansion(self) -> None:
        self._stub_capture({"echo hi": "hi"})
        cmd = "cat <<'EOF'\n$(echo hi)\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        # Body should be literal $(echo hi), not expanded
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")
        self.assertEqual(len(self.captures), 0)  # no $() expansion triggered

    def test_unquoted_heredoc_expands_dollar_paren(self) -> None:
        self._stub_capture({"echo hello": "hello"})
        cmd = "cat <<EOF\n$(echo hello)\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")

    def test_escaped_dollar_paren_in_heredoc_not_expanded(self) -> None:
        """A backslash-escaped $() in an unquoted heredoc body stays literal."""
        self._stub_capture({"echo hi": "hi"})
        cmd = "cat <<EOF\n\\$(echo hi)\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "\\$(echo hi)\n")
        self.assertEqual(len(self.captures), 0)  # no $() expansion triggered

    def test_heredoc_tab_strip(self) -> None:
        cmd = "cat <<-EOF\n\t\thello\n\tEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn("<<-", expanded)
        # Body should have ALL leading tabs stripped
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")

    def test_herestring_unquoted(self) -> None:
        cmd = "cat <<<hello"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")
        self.assertIn("<<<", expanded)

    def test_herestring_quoted(self) -> None:
        cmd = "cat <<<'hello world'"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello world\n")

    def test_herestring_expands_dollar_paren_unless_single_quoted(self) -> None:
        self._stub_capture({"echo hi": "hi"})
        # Unquoted here-string with $()
        cmd = "cat <<<$(echo hi)"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hi\n")

        # Single-quoted here-string with $() — no expansion
        self.captures.clear()
        cmd2 = "cat <<<'$(echo hi)'"
        expanded2, exp2, _program2 = _expand_command(cmd2, self.work_dir, 30, 0)
        m2 = SENTINEL_HD.search(expanded2)
        self.assertIsNotNone(m2)
        sentinel2 = f"\x01H{m2.group(1)}\x01"
        self.assertEqual(exp2.heredoc_bodies[sentinel2], "$(echo hi)\n")
        self.assertEqual(len(self.captures), 0)

    def test_command_substitution_sentinel(self) -> None:
        self._stub_capture({"echo hello": "hello"})
        cmd = "echo $(echo hello)"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        # Should contain arg sentinel
        m = SENTINEL_ARG.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01A{m.group(1)}\x01"
        self.assertIn(sentinel, exp.arg_values)
        self.assertEqual(exp.arg_values[sentinel], "hello")
        # The expanded command should have "echo <sentinel>"
        self.assertTrue(expanded.startswith("echo "))

    def test_nested_command_substitution(self) -> None:
        outputs = {"echo inner": "inner", "echo $(echo inner)": "outer"}
        self._stub_capture(outputs)
        cmd = "echo $(echo $(echo inner))"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        # The outer $() captures "echo $(echo inner)" since the inner $()
        # is inside the outer $(), and the outer _capture_stdout call is what
        # triggers the recursive expansion (via its own _expand_command call).
        # The stub short-circuits that.
        self.assertIn("echo $(echo inner)", self.captures)

    def test_unbalanced_dollar_paren_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _expand_command("echo $(unclosed", self.work_dir, 30, 0)
        self.assertIn("Unbalanced", str(ctx.exception))

    def test_missing_heredoc_terminator_error(self) -> None:
        cmd = "cat <<EOF\nhello\nworld\n"
        with self.assertRaises(ValueError) as ctx:
            _expand_command(cmd, self.work_dir, 30, 0)
        self.assertIn("not found", str(ctx.exception))

    def test_depth_limit(self) -> None:
        # Do NOT stub _capture_stdout — the depth check must fire.
        # Pass depth=MAX_SUBST_DEPTH so that when _expand_command encounters $()
        # it calls _capture_stdout with depth+1 = MAX_SUBST_DEPTH+1 which
        # triggers the limit.
        with self.assertRaises(ValueError) as ctx:
            _expand_command("echo $(echo A)", self.work_dir, 30, MAX_SUBST_DEPTH)
        self.assertIn("depth", str(ctx.exception).lower())

    def test_count_limit(self) -> None:
        # Do NOT stub _capture_stdout — the count check must fire.
        # Create many $() substitutions to exceed MAX_SUBST_COUNT.
        parts = ["$(echo {})".format(i) for i in range(MAX_SUBST_COUNT + 5)]
        cmd = "echo " + " ".join(parts)
        with self.assertRaises(ValueError) as ctx:
            _expand_command(cmd, self.work_dir, 30, 0)
        self.assertIn("count", str(ctx.exception).lower())

    def test_quotes_inside_heredoc_body_preserved(self) -> None:
        cmd = "cat <<EOF\nline with \"quotes\" and 'apostrophes'\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "line with \"quotes\" and 'apostrophes'\n")

    def test_double_quoted_delimiter_no_expansion(self) -> None:
        self._stub_capture({"echo hi": "hi"})
        cmd = 'cat <<"EOF"\n$(echo hi)\nEOF'
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        sentinel = f"\x01H{m.group(1)}\x01"
        # Body should be literal $(echo hi), not expanded
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")

class CaptureStdoutTest(unittest.TestCase):
    """Test _capture_stdout with stubbed segment/pipeline cores."""

    def setUp(self) -> None:
        self._orig_segment_core = server._run_segment_core
        self._orig_pipeline_core = server._run_pipeline_core
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        server._run_segment_core = self._orig_segment_core
        server._run_pipeline_core = self._orig_pipeline_core
        self._tmp.cleanup()

    def _stub_segment(self, mapping: dict[str, tuple[int, bytes, bytes, list]]) -> None:
        def fake(command, work_dir, timeout, expansion=None):
            key = command if isinstance(command, str) else server._serialize_command(command)
            if key in mapping:
                return mapping[key]
            return (0, b"", b"", [])
        server._run_segment_core = fake

    def _stub_pipeline(self, mapping: dict[tuple, tuple[int, bytes, bytes, list]]) -> None:
        def fake(segments, work_dir, timeout, expansion=None):
            key = tuple(
                s if isinstance(s, str) else server._serialize_command(s)
                for s in segments
            )
            if key in mapping:
                return mapping[key]
            return (0, b"", b"", [])
        server._run_pipeline_core = fake

    def test_single_segment_stdout(self) -> None:
        self._stub_segment({"echo hi": (0, b"hi\n", b"", [])})
        rc, stdout = _capture_stdout("echo hi", self.work_dir, 30, 1)
        self.assertEqual(rc, 0)
        self.assertEqual(stdout, b"hi\n")

    def test_pipeline_last_stage_stdout(self) -> None:
        self._stub_pipeline({("a", "b"): (0, b"result\n", b"", [])})
        rc, stdout = _capture_stdout("a | b", self.work_dir, 30, 1)
        self.assertEqual(rc, 0)
        self.assertEqual(stdout, b"result\n")

    def test_andand_short_circuit(self) -> None:
        self._stub_segment({
            "fail": (1, b"", b"fail", []),
            "succeed": (0, b"good", b"", []),
        })
        rc, stdout = _capture_stdout("fail && succeed", self.work_dir, 30, 1)
        self.assertEqual(rc, 1)
        self.assertEqual(stdout, b"")  # succeed skipped

    def test_background_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _capture_stdout("echo hi &", self.work_dir, 30, 1)
        self.assertIn("background", str(ctx.exception).lower())

    def test_output_truncated(self) -> None:
        long_output = b"x" * (MAX_SUBST_OUTPUT + 100)
        self._stub_segment({"big": (0, long_output, b"", [])})
        rc, stdout = _capture_stdout("big", self.work_dir, 30, 1)
        self.assertLessEqual(len(stdout), MAX_SUBST_OUTPUT)

    def test_empty_command(self) -> None:
        rc, stdout = _capture_stdout("", self.work_dir, 30, 1)
        self.assertEqual(rc, 0)
        self.assertEqual(stdout, b"")

class ResolveFdTargetsStdinTest(unittest.TestCase):
    """Test that _resolve_fd_targets returns stdin_bytes for heredoc/here-string."""

    def test_herestring_returns_stdin_bytes(self) -> None:
        redirs = [Redirect(fd=0, op="<<<", body="hello\n")]
        result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertIsInstance(result, FdPlan)
        self.assertEqual(result.stdin_bytes, b"hello\n")
        self.assertIsNone(result.stdin_file)
        self.assertIn("[stdin <<<]", result.report)

    def test_heredoc_returns_stdin_bytes(self) -> None:
        redirs = [Redirect(fd=0, op="<<", body="line1\nline2\n")]
        result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertEqual(result.stdin_bytes, b"line1\nline2\n")
        self.assertIsNone(result.stdin_file)
        self.assertIn("[stdin <<]", result.report)

    def test_heredoc_tab_returns_stdin_bytes(self) -> None:
        redirs = [Redirect(fd=0, op="<<-", body="tabbed\n", strip_tabs=True)]
        result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertEqual(result.stdin_bytes, b"tabbed\n")
        self.assertIn("[stdin <<-]", result.report)

    def test_no_stdin_redirect_returns_none(self) -> None:
        redirs = [Redirect(fd=1, op=">", raw_target="out.txt", target_path="/tmp/out.txt")]
        result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertIsInstance(result, FdPlan)
        self.assertIsNone(result.stdin_bytes)
        self.assertIsNone(result.stdin_file)

    def test_multiple_stdin_rejected_by_resolve(self) -> None:
        redirs = [
            Redirect(fd=0, op="<<", body="a\n"),
            Redirect(fd=0, op="<<", body="b\n"),
        ]
        with self.assertRaises(ValueError) as ctx:
            _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertIn("Multiple stdin redirects", str(ctx.exception))

    def test_input_redirect_returns_stdin_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            infile = Path(tmp) / "in.txt"
            infile.write_text("data\n")
            redirs = [Redirect(fd=0, op="<", raw_target=str(infile), target_path=str(infile))]
            result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
            self.assertIsInstance(result, FdPlan)
            self.assertIsNone(result.stdin_bytes)
            self.assertIsNotNone(result.stdin_file)
            self.assertIn(f"[stdin <- {infile}]", result.report)
            self.assertIn(result.stdin_file, result.to_close)
            # Sanity: the file object actually reads the file content.
            self.assertEqual(result.stdin_file.read(), b"data\n")
            result.stdin_file.close()

    def test_input_redirect_missing_file_raises(self) -> None:
        redirs = [Redirect(fd=0, op="<", raw_target="nope.txt", target_path="/nonexistent/nope.txt")]
        with self.assertRaises(ValueError) as ctx:
            _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertIn("Input redirect file not found", str(ctx.exception))

    def test_input_redirect_conflicts_with_heredoc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            infile = Path(tmp) / "in.txt"
            infile.write_text("x\n")
            redirs = [
                Redirect(fd=0, op="<", raw_target=str(infile), target_path=str(infile)),
                Redirect(fd=0, op="<<", body="x\n"),
            ]
            with self.assertRaises(ValueError) as ctx:
                _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
            self.assertIn("Multiple stdin redirects", str(ctx.exception))

class BuildInvocationHeredocTest(unittest.TestCase):
    """Test that _build_invocation threads expansion correctly."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_heredoc_redirect_passed_through(self) -> None:
        expansion = Expansion(
            arg_values={},
            heredoc_bodies={"\x01H0\x01": "body\n"},
        )
        inv = _build_invocation(
            "cat << \x01H0\x01", self.root, expansion=expansion,
        )
        self.assertIsInstance(inv, Invocation)
        self.assertEqual(len(inv.redirects), 1)
        self.assertEqual(inv.redirects[0].fd, 0)
        self.assertEqual(inv.redirects[0].op, "<<")
        self.assertEqual(inv.redirects[0].body, "body\n")

class RunSegmentCoreStdinTest(unittest.TestCase):
    """Test that _run_segment_core passes stdin_bytes via input= to subprocess.run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_build = server._build_invocation
        self._orig_run = server.subprocess.run

    def tearDown(self) -> None:
        server._build_invocation = self._orig_build
        server.subprocess.run = self._orig_run
        self._tmp.cleanup()

    def test_stdin_bytes_passed_to_subprocess_run(self) -> None:
        import subprocess as _sp

        captured_input = []

        def fake_run(args, **kwargs):
            captured_input.append(kwargs.get("input"))
            return _sp.CompletedProcess(args, 0, stdout=b"ok\n", stderr=b"")

        server.subprocess.run = fake_run

        def fake_build(command, work_dir, expansion=None):
            return Invocation(
                "/usr/bin/cat",
                ["/usr/bin/cat"],
                None,
                {},
                [Redirect(fd=0, op="<<", body="hello\n")],
            )

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_segment_core(
            "cat << DUMMY", self.root, 30,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(stdout_b, b"ok\n")
        self.assertEqual(captured_input[0], b"hello\n")

    def test_no_stdin_when_no_heredoc(self) -> None:
        import subprocess as _sp

        captured_input = []

        def fake_run(args, **kwargs):
            captured_input.append(kwargs.get("input"))
            return _sp.CompletedProcess(args, 0, stdout=b"", stderr=b"")

        server.subprocess.run = fake_run

        def fake_build(command, work_dir, expansion=None):
            return Invocation(
                "/usr/bin/echo",
                ["/usr/bin/echo", "hi"],
                None,
                {},
                [],
            )

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_segment_core(
            "echo hi", self.root, 30,
        )
        self.assertIsNone(captured_input[0])

    def test_input_redirect_passes_stdin_file_not_input(self) -> None:
        import subprocess as _sp

        infile = self.root / "in.txt"
        infile.write_text("data\n")
        captured = {}

        def fake_run(args, **kwargs):
            captured["input"] = kwargs.get("input")
            captured["stdin"] = kwargs.get("stdin")
            return _sp.CompletedProcess(args, 0, stdout=b"ok\n", stderr=b"")

        server.subprocess.run = fake_run

        def fake_build(command, work_dir, expansion=None):
            return Invocation(
                "/usr/bin/cat",
                ["/usr/bin/cat"],
                None,
                {},
                [Redirect(fd=0, op="<", raw_target=str(infile), target_path=str(infile))],
            )

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_segment_core(
            f"cat < {infile}", self.root, 30,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(stdout_b, b"ok\n")
        # Input redirect must go via stdin=<file>, NOT input=<bytes>.
        self.assertIsNone(captured.get("input"))
        self.assertIsNotNone(captured.get("stdin"))
        self.assertIn(f"[stdin <- {infile}]", report)

class RunPipelineCoreStdinTest(unittest.TestCase):
    """Test _run_pipeline_core stdin plumbing and non-first-stage rejection."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_build = server._build_invocation
        self._orig_popen = server.subprocess.Popen

    def tearDown(self) -> None:
        server._build_invocation = self._orig_build
        server.subprocess.Popen = self._orig_popen
        self._tmp.cleanup()

    def test_heredoc_on_first_stage_uses_stdin_pipe(self) -> None:
        """First stage with heredoc gets stdin=PIPE with writer thread."""
        import io

        captured_stdins = []
        stdin_buffers = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                captured_stdins.append(kwargs.get("stdin"))
                # When stdin=subprocess.PIPE, provide a writable BytesIO buffer
                if kwargs.get("stdin") is subprocess.PIPE:
                    self.stdin = io.BytesIO()
                    stdin_buffers.append(self.stdin)
                else:
                    self.stdin = None
                self.stdout = _FakePipe()
                self.stderr = _FakePipe()
                self.pid = 9999
                self.returncode = 0

            def poll(self):
                return 0
            def wait(self):
                return 0
            def communicate(self, timeout=None):
                return (b"output\n", b"")
            def kill(self):
                pass

        class _FakePipe:
            def close(self):
                pass
            def read(self):
                return b""

        def fake_build(command, work_dir, expansion=None):
            if "cat" in command:
                return Invocation(
                    "/usr/bin/cat",
                    ["/usr/bin/cat"],
                    None,
                    {},
                    [Redirect(fd=0, op="<<", body="hello\n")],
                )
            if "grep" in command:
                return Invocation(
                    "/usr/bin/grep",
                    ["/usr/bin/grep", "x"],
                    None,
                    {},
                    [],
                )
            return EmptyInvocation()

        server._build_invocation = fake_build
        server.subprocess.Popen = FakePopen

        rc, stdout_b, stderr_b, report = _run_pipeline_core(
            ["cat << H0", "grep x"], self.root, 30,
        )
        self.assertEqual(rc, 0)
        # First stage should have gotten subprocess.PIPE for stdin
        self.assertIs(captured_stdins[0], subprocess.PIPE)
        # Second stage should have gotten the prev.stdout (which is the FakePipe)
        self.assertIsNotNone(captured_stdins[1])

    def test_heredoc_on_non_first_stage_rejected(self) -> None:
        """Non-first stage with heredoc should be rejected."""
        def fake_build(command, work_dir, expansion=None):
            if "echo" in command:
                return Invocation(
                    "/usr/bin/echo",
                    ["/usr/bin/echo", "hi"],
                    None,
                    {},
                    [],
                )
            if "cat" in command:
                return Invocation(
                    "/usr/bin/cat",
                    ["/usr/bin/cat"],
                    None,
                    {},
                    [Redirect(fd=0, op="<<", body="hello\n")],
                )
            return EmptyInvocation()

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_pipeline_core(
            ["echo hi", "cat << H0"], self.root, 30,
        )
        self.assertEqual(rc, 1)
        self.assertIn(b"not allowed on non-first", stdout_b)

    def test_input_redirect_on_first_stage_passes_file_stdin(self) -> None:
        """First-stage < file passes the file object as stdin= (not PIPE)."""
        infile = self.root / "in.txt"
        infile.write_text("data\n")

        captured_stdins = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                captured_stdins.append(kwargs.get("stdin"))
                self.stdin = None
                self.stdout = _FakePipe()
                self.stderr = _FakePipe()
                self.pid = 9999
                self.returncode = 0

            def poll(self):
                return 0
            def wait(self):
                return 0
            def communicate(self, timeout=None):
                return (b"output\n", b"")
            def kill(self):
                pass

        class _FakePipe:
            def close(self):
                pass
            def read(self):
                return b""

        def fake_build(command, work_dir, expansion=None):
            if "cat" in command:
                return Invocation(
                    "/usr/bin/cat",
                    ["/usr/bin/cat"],
                    None,
                    {},
                    [Redirect(fd=0, op="<", raw_target=str(infile), target_path=str(infile))],
                )
            if "grep" in command:
                return Invocation(
                    "/usr/bin/grep",
                    ["/usr/bin/grep", "x"],
                    None,
                    {},
                    [],
                )
            return EmptyInvocation()

        server._build_invocation = fake_build
        server.subprocess.Popen = FakePopen

        rc, stdout_b, stderr_b, report = _run_pipeline_core(
            [f"cat < {infile}", "grep x"], self.root, 30,
        )
        self.assertEqual(rc, 0)
        # First stage stdin must be the open file object, not a pipe.
        self.assertIsNot(captured_stdins[0], subprocess.PIPE)
        self.assertIsNotNone(captured_stdins[0])
        # The returned report is the LAST stage's; the first stage's stdin
        # report line must still have been produced internally (checked by
        # the file-object stdin above).

    def test_input_redirect_on_non_first_stage_rejected(self) -> None:
        """< file on a non-first stage is rejected like a heredoc."""
        def fake_build(command, work_dir, expansion=None):
            if "echo" in command:
                return Invocation(
                    "/usr/bin/echo",
                    ["/usr/bin/echo", "hi"],
                    None,
                    {},
                    [],
                )
            if "cat" in command:
                return Invocation(
                    "/usr/bin/cat",
                    ["/usr/bin/cat"],
                    None,
                    {},
                    [Redirect(fd=0, op="<", raw_target="in.txt", target_path="/tmp/in.txt")],
                )
            return EmptyInvocation()

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_pipeline_core(
            ["echo hi", "cat < in.txt"], self.root, 30,
        )
        self.assertEqual(rc, 1)
        self.assertIn(b"not allowed on non-first", stdout_b)



if __name__ == "__main__":
    unittest.main()
