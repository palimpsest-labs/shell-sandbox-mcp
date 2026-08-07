"""Tests for redirect extraction and validation. Run with the venv python that has `mcp` installed:

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
# _extract_redirects
# ---------------------------------------------------------------------------


class ExtractRedirectsTest(unittest.TestCase):
    """Pure unit tests for ``_extract_redirects`` — no subprocess calls."""

    def _extract(self, segment: str):
        return server._extract_redirects(segment)

    def test_simple_stdout_redirect(self) -> None:
        args, redirs, err = self._extract("echo hi > out.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        r = redirs[0]
        self.assertEqual(r.fd, 1)
        self.assertEqual(r.op, ">")
        self.assertEqual(r.raw_target, "out.txt")
        self.assertIsNone(r.target_path)

    def test_stdout_append(self) -> None:
        args, redirs, err = self._extract("echo hi >> log.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">>")

    def test_stderr_redirect(self) -> None:
        args, redirs, err = self._extract("cmd 2> err.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "err.txt")

    def test_stderr_append(self) -> None:
        args, redirs, err = self._extract("cmd 2>> err.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">>")

    def test_2gt1_fd_dup(self) -> None:
        args, redirs, err = self._extract("cmd 2>&1")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">&")
        self.assertEqual(redirs[0].target_fd, 1)

    def test_1gt2_fd_dup(self) -> None:
        args, redirs, err = self._extract("cmd 1>&2")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">&")
        self.assertEqual(redirs[0].target_fd, 2)

    def test_2gt1x_not_fd_dup(self) -> None:
        # `2>&1x` — the `x` after `1` means this is a `2>` redirect to file
        # `&1x`, NOT an fd-dup operator.
        args, redirs, err = self._extract("cmd 2>&1x")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "&1x")

    def test_1gt2y_not_fd_dup(self) -> None:
        # `1>&2y` — the `y` after `2` means this is a `1>` redirect to file
        # `&2y`, NOT an fd-dup operator.
        args, redirs, err = self._extract("cmd 1>&2y")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "&2y")

    def test_quoted_operator_not_redirect(self) -> None:
        args, redirs, err = self._extract('echo ">" hello')
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ">", "hello"])
        self.assertEqual(len(redirs), 0)

    def test_quoted_operator_single_quote(self) -> None:
        args, redirs, err = self._extract("echo '>' hello")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ">", "hello"])
        self.assertEqual(len(redirs), 0)

    def test_redirect_leading(self) -> None:
        args, redirs, err = self._extract(">out echo x")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "x"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out")

    def test_redirect_middle(self) -> None:
        args, redirs, err = self._extract("echo a > f b")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "f")

    def test_multiple_redirects(self) -> None:
        args, redirs, err = self._extract("cmd 2>e 1>&2")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 2)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "e")
        self.assertEqual(redirs[1].fd, 1)
        self.assertEqual(redirs[1].op, ">&")
        self.assertEqual(redirs[1].target_fd, 2)

    def test_glued_not_redirect(self) -> None:
        # foo>bar — > is not at word boundary, treated as literal
        args, redirs, err = self._extract("echo foo>bar")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "foo>bar"])
        self.assertEqual(len(redirs), 0)

    def test_glued_target_ok(self) -> None:
        # >out.txt — > is at word start, out.txt is glued target
        args, redirs, err = self._extract(">out.txt echo hi")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out.txt")

    def test_missing_target_error(self) -> None:
        args, redirs, err = self._extract("echo >")
        self.assertEqual(err, "Redirect operator missing target file")

    def test_missing_target_2gt_error(self) -> None:
        args, redirs, err = self._extract("echo 2>")
        self.assertEqual(err, "Redirect operator missing target file")

    def test_fd_gt_2_error(self) -> None:
        args, redirs, err = self._extract("echo 3> f")
        self.assertEqual(err, "Redirects only support fds 1 and 2 (got 3)")

    def test_fd_0_error(self) -> None:
        args, redirs, err = self._extract("echo 0> f")
        self.assertEqual(err, "Redirects only support fds 1 and 2 (got 0)")

    def test_2gt3_error(self) -> None:
        # 2>&3 — only 1 and 2 are valid dup target fds.
        args, redirs, err = self._extract("cmd 2>&3")
        self.assertEqual(err, "Redirect dup target fd must be 1 or 2")

    def test_input_redirect_error(self) -> None:
        args, redirs, err = self._extract("cmd < file")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 0)
        self.assertEqual(redirs[0].op, "<")
        self.assertEqual(redirs[0].raw_target, "file")

    def test_input_redirect_glued(self) -> None:
        args, redirs, err = self._extract("cmd <file")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 0)
        self.assertEqual(redirs[0].op, "<")
        self.assertEqual(redirs[0].raw_target, "file")

    def test_input_redirect_missing_target(self) -> None:
        args, redirs, err = self._extract("cmd <")
        self.assertEqual(err, "Input redirect missing target file")

    def test_input_redirect_then_heredoc_rejected(self) -> None:
        expansion = Expansion(arg_values={}, heredoc_bodies={"\x01H0\x01": "body\n"})
        args, redirs, err = server._extract_redirects(
            "cmd < file << \x01H0\x01", expansion=expansion,
        )
        self.assertIsNotNone(err)
        self.assertIn("Multiple stdin redirects", err)

    def test_heredoc_then_input_rejected(self) -> None:
        expansion = Expansion(arg_values={}, heredoc_bodies={"\x01H0\x01": "body\n"})
        args, redirs, err = server._extract_redirects(
            "cmd << \x01H0\x01 < file", expansion=expansion,
        )
        self.assertIsNotNone(err)
        self.assertIn("Multiple stdin redirects", err)

    def test_input_heredoc_error(self) -> None:
        args, redirs, err = self._extract("cmd << EOF")
        # Without expansion, bare << with non-sentinel target cannot resolve
        self.assertIn("not found", err)

    def test_unbalanced_quotes_error(self) -> None:
        args, redirs, err = self._extract('echo "hi')
        self.assertEqual(err, "Unbalanced quotes in command")

    def test_1gt_redirect(self) -> None:
        args, redirs, err = self._extract("cmd 1> out.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">")

    def test_1gtgt_redirect(self) -> None:
        args, redirs, err = self._extract("cmd 1>> out.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">>")

    def test_no_args_only_redirect(self) -> None:
        args, redirs, err = self._extract("> out.txt")
        self.assertIsNone(err)
        self.assertEqual(args, [])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out.txt")

# ---------------------------------------------------------------------------
# _validate_redirect_paths
# ---------------------------------------------------------------------------


class ValidateRedirectPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_valid_path(self) -> None:
        redirs = [server.Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target='out.txt')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNone(err)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].target_path, str((self.root / "out.txt").resolve()))

    def test_dotdot_escape(self) -> None:
        redirs = [server.Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target='../escape')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNotNone(err)
        self.assertIn("escapes allowed roots", err)

    def test_absolute_escape(self) -> None:
        redirs = [server.Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target='/etc/passwd')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNotNone(err)
        self.assertIn("escapes allowed roots", err)

    def test_symlink_escape(self) -> None:
        (self.root / "evil").symlink_to("/etc")
        redirs = [server.Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target='evil/hostname')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNotNone(err)
        self.assertIn("escapes allowed roots", err)

    def test_dev_null_output(self) -> None:
        redirs = [server.Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target='/dev/null')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNone(err)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].target_path, str(Path("/dev/null").resolve()))

    def test_dev_null_input(self) -> None:
        redirs = [server.Redirect(fd=0, op='<', target_path=None, target_fd=None, raw_target='/dev/null')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNone(err)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].target_path, str(Path("/dev/null").resolve()))

    def test_dev_null_2gt(self) -> None:
        redirs = [server.Redirect(fd=2, op='>', target_path=None, target_fd=None, raw_target='/dev/null')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNone(err)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].target_path, str(Path("/dev/null").resolve()))

    def test_2gt1_passes_through(self) -> None:
        redirs = [server.Redirect(fd=2, op='>&', target_path=None, target_fd=1, raw_target='1')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNone(err)
        self.assertEqual(validated, redirs)

    def test_input_redirect_in_workdir(self) -> None:
        (self.root / "in.txt").write_text("data\n")
        redirs = [server.Redirect(fd=0, op='<', target_path=None, target_fd=None, raw_target='in.txt')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNone(err)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].target_path, str((self.root / "in.txt").resolve()))

    def test_input_redirect_under_tmp(self) -> None:
        redirs = [server.Redirect(fd=0, op='<', target_path=None, target_fd=None, raw_target='/tmp/input-redir-test')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNone(err)
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].target_path, str(Path("/tmp/input-redir-test").resolve()))

    def test_input_redirect_symlink_escape(self) -> None:
        # A symlink under /tmp pointing outside /tmp must be rejected.
        link = Path("/tmp/input-redir-symlink-test")
        try:
            link.symlink_to("/etc/passwd")
            redirs = [server.Redirect(fd=0, op='<', target_path=None, target_fd=None, raw_target=str(link))]
            validated, err = server._validate_redirect_paths(redirs, self.root)
            self.assertIsNotNone(err)
            self.assertIn("escapes allowed roots", err)
        finally:
            if link.exists() or link.is_symlink():
                try:
                    link.unlink()
                except OSError:
                    pass

# ---------------------------------------------------------------------------
# _build_invocation with redirects
# ---------------------------------------------------------------------------


class BuildInvocationRedirectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_redirect_resolved_to_work_dir(self) -> None:
        inv = server._build_invocation("echo hi > out.txt", self.root)
        self.assertIsInstance(inv, server.Invocation)
        self.assertEqual(len(inv.redirects), 1)
        self.assertEqual(inv.redirects[0].target_path, str((self.root / "out.txt").resolve()))
        # Ensure > and out.txt are NOT in sandbox_args
        self.assertNotIn(">", inv.sandbox_args)
        self.assertNotIn("out.txt", inv.sandbox_args)

    def test_escape_path_error(self) -> None:
        inv = server._build_invocation("echo > ../escape", self.root)
        self.assertIsInstance(inv, server.InvocationError)
        self.assertIn("escapes allowed roots", inv.message)

    def test_invalid_fd_error(self) -> None:
        inv = server._build_invocation("echo 3> f", self.root)
        self.assertIsInstance(inv, server.InvocationError)
        self.assertIn("only support fds 1 and 2", inv.message)

# ---------------------------------------------------------------------------
# _run_segment with redirects (stubbed subprocess — sandbox python can't fork)
# ---------------------------------------------------------------------------


class RunSegmentRedirectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_build = server._build_invocation
        self._orig_run = server.subprocess.run

    def tearDown(self) -> None:
        server._build_invocation = self._orig_build
        server.subprocess.run = self._orig_run
        self._tmp.cleanup()

    def _stub_build(self, redirects):
        """Stub _build_invocation to return a busybox echo invocation with given redirects."""
        def fake_build(command, work_dir, expansion=None):
            return server.Invocation(
                str(server.BUSYBOX_BIN.resolve()),
                [str(server.BUSYBOX_BIN.resolve()), "echo", "hi"],
                None,
                {},
                redirects,
            )
        server._build_invocation = fake_build

    def _stub_run_writing_stdout(self):
        """Stub subprocess.run to write 'hi\n' to whatever stdout target is passed."""
        import subprocess as _sp

        def fake_run(args, **kwargs):
            stdout_target = kwargs.get("stdout", _sp.PIPE)
            stderr_target = kwargs.get("stderr", _sp.PIPE)
            # Write "hi\n" to the stdout target
            if hasattr(stdout_target, 'write'):
                stdout_target.write(b"hi\n")
                stdout_target.flush()
            elif isinstance(stdout_target, int):
                os.write(stdout_target, b"hi\n")
            # Return a CompletedProcess
            return _sp.CompletedProcess(args, 0, stdout=None, stderr=None)

        server.subprocess.run = fake_run

    def test_stdout_redirect_opens_file(self) -> None:
        outfile = self.root / "out.txt"
        self._stub_build([
            server.Redirect(fd=1, op='>', target_path=str(outfile), target_fd=None, raw_target='out.txt'),
        ])
        self._stub_run_writing_stdout()
        rc, out = server._run_segment("testcmd", self.root, 10)
        self.assertEqual(rc, 0)
        self.assertIn("[stdout -> out.txt]", out)
        # File should have "hi"
        self.assertEqual(outfile.read_text(), "hi\n")

    def test_stderr_redirect_opens_file(self) -> None:
        errfile = self.root / "err.txt"
        self._stub_build([
            server.Redirect(fd=2, op='>', target_path=str(errfile), target_fd=None, raw_target='err.txt'),
        ])
        # subprocess.run writes to stdout (PIPE) and we capture it
        import subprocess as _sp
        def fake_run(args, **kwargs):
            return _sp.CompletedProcess(args, 0, stdout=b"hi\n", stderr=None)
        server.subprocess.run = fake_run
        rc, out = server._run_segment("testcmd", self.root, 10)
        self.assertEqual(rc, 0)
        self.assertIn("hi", out)
        self.assertIn("[stderr -> err.txt]", out)
        # err.txt should exist but be empty (since run didn't write to it)
        self.assertTrue(errfile.exists())

    def test_truncate_behavior(self) -> None:
        outfile = self.root / "out.txt"
        outfile.write_text("old content")
        self._stub_build([
            server.Redirect(fd=1, op='>', target_path=str(outfile), target_fd=None, raw_target='out.txt'),
        ])
        self._stub_run_writing_stdout()
        rc, out = server._run_segment("testcmd", self.root, 10)
        self.assertEqual(rc, 0)
        self.assertEqual(outfile.read_text(), "hi\n")

    def test_append_behavior(self) -> None:
        outfile = self.root / "out.txt"
        outfile.write_text("line1\n")
        self._stub_build([
            server.Redirect(fd=1, op='>>', target_path=str(outfile), target_fd=None, raw_target='out.txt'),
        ])
        self._stub_run_writing_stdout()
        rc, out = server._run_segment("testcmd", self.root, 10)
        self.assertEqual(rc, 0)
        content = outfile.read_text()
        self.assertIn("line1", content)
        self.assertIn("hi", content)

    def test_repeated_same_fd_last_wins(self) -> None:
        f1 = self.root / "f1"
        f2 = self.root / "f2"
        self._stub_build([
            server.Redirect(fd=1, op='>', target_path=str(f1), target_fd=None, raw_target='f1'),
            server.Redirect(fd=1, op='>', target_path=str(f2), target_fd=None, raw_target='f2'),
        ])
        self._stub_run_writing_stdout()
        rc, out = server._run_segment("testcmd", self.root, 10)
        self.assertEqual(rc, 0)
        # f1 truncated/empty, f2 gets "hi\n"
        self.assertEqual(f1.read_text(), "")
        self.assertEqual(f2.read_text(), "hi\n")

    def test_2gt1_report_line(self) -> None:
        import subprocess as _sp

        self._stub_build([
            server.Redirect(fd=2, op='>&', target_path=None, target_fd=1, raw_target='1'),
        ])
        def fake_run(args, **kwargs):
            # Assert stderr=subprocess.STDOUT was passed
            self.assertIs(kwargs.get("stderr"), _sp.STDOUT)
            return _sp.CompletedProcess(args, 0, stdout=b"hi\n", stderr=None)
        server.subprocess.run = fake_run
        rc, out = server._run_segment("testcmd", self.root, 10)
        self.assertEqual(rc, 0)
        self.assertIn("[stderr -> stdout]", out)

    def test_2gt1_then_stdout_redirect_snapshots(self) -> None:
        # `2>&1 >file`: stderr must be bound to the ORIGINAL stdout (a shared
        # pipe), not dragged into `file` by the later stdout redirect.
        import subprocess as _sp

        outfile = self.root / "out.txt"
        self._stub_build([
            server.Redirect(fd=2, op='>&', target_path=None, target_fd=1, raw_target='1'),
            server.Redirect(fd=1, op='>', target_path=str(outfile), target_fd=None, raw_target='out.txt'),
        ])
        captured = {}
        def fake_run(args, **kwargs):
            captured["stdout"] = kwargs.get("stdout")
            captured["stderr"] = kwargs.get("stderr")
            # Simulate writing to both redirected targets
            if hasattr(kwargs.get("stdout"), "write"):
                kwargs["stdout"].write(b"to-file\n")
            elif isinstance(kwargs.get("stdout"), int):
                os.write(kwargs["stdout"], b"to-file\n")
            if hasattr(kwargs.get("stderr"), "write"):
                kwargs["stderr"].write(b"to-stderr\n")
            elif isinstance(kwargs.get("stderr"), int):
                os.write(kwargs["stderr"], b"to-stderr\n")
            return _sp.CompletedProcess(args, 0, stdout=None, stderr=None)
        server.subprocess.run = fake_run
        rc, out = server._run_segment("testcmd", self.root, 10)
        self.assertEqual(rc, 0)
        # stdout target is the file; stderr target is a shared pipe fd (int)
        self.assertTrue(hasattr(captured["stdout"], "write"))
        self.assertIsInstance(captured["stderr"], int)
        # stderr must NOT have gone to out.txt
        self.assertEqual(outfile.read_text(), "to-file\n")
        self.assertIn("[stderr -> stdout]", out)

    def test_redirect_target_symlink_rejected(self) -> None:
        # O_NOFOLLOW: a redirect target that is a symlink (even inside the
        # work dir) must not be followed when opening for output.
        target = self.root / "real.txt"
        link = self.root / "out.txt"
        link.symlink_to(target)
        self._stub_build([
            server.Redirect(fd=1, op='>', target_path=str(link), target_fd=None, raw_target='out.txt'),
        ])
        def fake_run(args, **kwargs):
            return _sp.CompletedProcess(args, 1, stdout=None, stderr=None)
        server.subprocess.run = fake_run
        rc, out = server._run_segment("testcmd", self.root, 10)
        # The open should raise (ELOOP), surfaced as a clean error -> rc 1.
        self.assertEqual(rc, 1)
        self.assertIn("Error opening redirect target", out)
        # The symlink target must NOT have been created/truncated.
        self.assertFalse(target.exists())

# ---------------------------------------------------------------------------
# _run_pipeline with redirects
# ---------------------------------------------------------------------------


class RunPipelineRedirectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_build = server._build_invocation
        self._orig_popen = server.subprocess.Popen

    def tearDown(self) -> None:
        server._build_invocation = self._orig_build
        server.subprocess.Popen = self._orig_popen
        self._tmp.cleanup()

    def test_intermediate_stdout_redirect_rejected(self) -> None:
        def fake_build(command, work_dir, expansion=None):
            if command == "producer > f":
                return server.Invocation(
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "echo", "hi"],
                    None,
                    {},
                    [server.Redirect(fd=1, op='>', target_path=str(work_dir / "f"), target_fd=None, raw_target="f")],
                )
            if command == "consumer":
                return server.Invocation(
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "cat"],
                    None,
                    {},
                    [],
                )
            return server.EmptyInvocation()

        server._build_invocation = fake_build
        rc, out = server._run_pipeline(["producer > f", "consumer"], self.root, 10)
        self.assertEqual(rc, 1)
        self.assertIn("Cannot redirect stdout of intermediate pipe stage", out)

    def test_last_stage_stdout_redirect(self) -> None:
        outfile = self.root / "out.txt"

        def fake_build(command, work_dir, expansion=None):
            if command == "producer":
                return server.Invocation(
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "echo", "hello"],
                    None,
                    {},
                    [],
                )
            if command == f"consumer > {outfile}":
                return server.Invocation(
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "cat"],
                    None,
                    {},
                    [server.Redirect(fd=1, op='>', target_path=str(outfile), target_fd=None, raw_target="out.txt")],
                )
            return server.EmptyInvocation()

        server._build_invocation = fake_build

        # Stub Popen to simulate a successful pipeline.
        # Both stdout and stderr are fake pipes with close()/read().
        class _FakePipe:
            def close(self):
                pass
            def read(self):
                return b""

        class FakePopen:
            def __init__(self, args, **kwargs):
                self.args = args
                self._stdout_target = kwargs.get("stdout")
                self.stdin = kwargs.get("stdin")
                self.returncode = 0
                self.pid = 9999
                self.stdout = _FakePipe()
                self.stderr = _FakePipe()

            def poll(self):
                return 0
            def wait(self):
                return 0
            def communicate(self, timeout=None):
                if hasattr(self._stdout_target, 'write'):
                    self._stdout_target.write(b"hello\n")
                return (None, b"")

        server.subprocess.Popen = FakePopen
        rc, out = server._run_pipeline(["producer", f"consumer > {outfile}"], self.root, 10)
        self.assertEqual(rc, 0)
        self.assertIn("[stdout -> out.txt]", out)
        self.assertEqual(outfile.read_text().strip(), "hello")

# ---------------------------------------------------------------------------
# _run_background with redirects
# ---------------------------------------------------------------------------


class RunBackgroundRedirectTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_build = server._build_invocation
        self._orig_popen = server.subprocess.Popen
        self._orig_start_reaper = server._start_reaper

    def tearDown(self) -> None:
        server._build_invocation = self._orig_build
        server.subprocess.Popen = self._orig_popen
        server._start_reaper = self._orig_start_reaper
        self._tmp.cleanup()

    def _stub_popen(self):
        """Stub Popen to return a fake process without forking."""
        class FakePopen:
            def __init__(self, args, **kwargs):
                self.args = args
                self.stdout = kwargs.get("stdout")
                self.stderr = kwargs.get("stderr")
                self.stdin = kwargs.get("stdin")
                self.pid = 9999

            def poll(self):
                return 0

            def wait(self):
                return 0

        server.subprocess.Popen = FakePopen
        server._start_reaper = lambda: None  # no-op

    def test_stdout_redirect_writes_file_and_log(self) -> None:
        self._stub_popen()
        outfile = self.root / "out.txt"

        def fake_build(command, work_dir, expansion=None):
            return server.Invocation(
                str(server.BUSYBOX_BIN.resolve()),
                [str(server.BUSYBOX_BIN.resolve()), "echo", "hi"],
                None,
                {},
                [server.Redirect(fd=1, op='>', target_path=str(outfile), target_fd=None, raw_target='out.txt')],
            )

        server._build_invocation = fake_build
        rc, out = server._run_background(
            [f"echo hi > {outfile}"], self.root,
        )
        self.assertEqual(rc, 0)
        self.assertIn("Backgrounded PID", out)
        self.assertIn("[stdout -> out.txt]", out)

    def test_stderr_redirect_report_line(self) -> None:
        self._stub_popen()
        errfile = self.root / "err.txt"

        def fake_build(command, work_dir, expansion=None):
            return server.Invocation(
                str(server.BUSYBOX_BIN.resolve()),
                [str(server.BUSYBOX_BIN.resolve()), "echo", "hi"],
                None,
                {},
                [server.Redirect(fd=2, op='>', target_path=str(errfile), target_fd=None, raw_target='err.txt')],
            )

        server._build_invocation = fake_build
        rc, out = server._run_background(
            [f"echo hi 2> {errfile}"], self.root,
        )
        self.assertEqual(rc, 0)
        self.assertIn("Backgrounded PID", out)
        self.assertIn("[stderr -> err.txt]", out)

    def test_background_heredoc_threads_expansion(self) -> None:
        """_run_background passes expansion through to _build_invocation.

        Without this, backgrounded heredocs resolve their sentinel body to None
        and fail with "Heredoc body not found".
        """
        self._stub_popen()
        expansion = Expansion(
            arg_values={},
            heredoc_bodies={"\x01H0\x01": "hello\n"},
        )
        received = {}

        def fake_build(command, work_dir, expansion=None):
            received["expansion"] = expansion
            return server.Invocation(
                str(server.BUSYBOX_BIN.resolve()),
                [str(server.BUSYBOX_BIN.resolve()), "cat"],
                None,
                {},
                [server.Redirect(fd=0, op="<<", body="hello\n")],
            )

        server._build_invocation = fake_build
        rc, out = server._run_background(
            [f"cat << \x01H0\x01"], self.root, expansion=expansion,
        )
        self.assertEqual(rc, 0)
        self.assertIn("Backgrounded PID", out)
        self.assertIs(received["expansion"], expansion)

    def test_background_command_substitution_resolves(self) -> None:
        """$() sentinels resolve via expansion in backgrounded commands."""
        launched = []

        class RecPopen:
            def __init__(self, args, **kwargs):
                launched.append(args)
                self.stdout = kwargs.get("stdout")
                self.stderr = kwargs.get("stderr")
                self.stdin = kwargs.get("stdin")
                self.pid = 9001

            def poll(self):
                return 0

            def wait(self):
                return 0

        server.subprocess.Popen = RecPopen
        server._start_reaper = lambda: None

        expansion = Expansion(arg_values={"\x01A0\x01": "world"}, heredoc_bodies={})
        rc, out = server._run_background(
            [f"echo \x01A0\x01"], self.root, expansion=expansion,
        )
        self.assertEqual(rc, 0)
        self.assertIn("Backgrounded PID", out)
        # Real _build_invocation should have resolved the sentinel to "world".
        self.assertTrue(any("world" in a for a in launched))

    def test_background_heredoc_on_non_first_stage_rejected(self) -> None:
        """Non-first pipeline stage heredoc is rejected in background mode."""
        def fake_build(command, work_dir, expansion=None):
            if "echo" in command:
                return server.Invocation(
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "echo", "hi"],
                    None,
                    {},
                    [],
                )
            if "cat" in command:
                return server.Invocation(
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "cat"],
                    None,
                    {},
                    [server.Redirect(fd=0, op="<<", body="hello\n")],
                )
            return server.EmptyInvocation()

        server._build_invocation = fake_build
        rc, out = server._run_background(
            ["echo hi", "cat << H0"], self.root,
        )
        self.assertEqual(rc, 1)
        self.assertIn("not allowed on non-first", out)

    def test_background_input_redirect_sets_first_stdin(self) -> None:
        """First-stage < file passes the file object as stdin= on the first Popen."""
        infile = self.root / "in.txt"
        infile.write_text("data\n")
        captured_stdins = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                captured_stdins.append(kwargs.get("stdin"))
                self.stdout = kwargs.get("stdout")
                self.stderr = kwargs.get("stderr")
                self.stdin = kwargs.get("stdin")
                self.pid = 9999

            def poll(self):
                return 0
            def wait(self):
                return 0

        server.subprocess.Popen = FakePopen
        server._start_reaper = lambda: None

        def fake_build(command, work_dir, expansion=None):
            return server.Invocation(
                str(server.BUSYBOX_BIN.resolve()),
                [str(server.BUSYBOX_BIN.resolve()), "cat"],
                None,
                {},
                [server.Redirect(fd=0, op="<", raw_target=str(infile), target_path=str(infile))],
            )

        server._build_invocation = fake_build
        rc, out = server._run_background(
            [f"cat < {infile}"], self.root,
        )
        self.assertEqual(rc, 0)
        self.assertIn("Backgrounded PID", out)
        # The first (only) stage's stdin must be the open file object.
        self.assertIsNot(captured_stdins[0], subprocess.PIPE)
        self.assertIsNotNone(captured_stdins[0])
        self.assertIn(f"[stdin <- {infile}]", out)


# ---------------------------------------------------------------------------
# Heredoc / here-string / command substitution tests (moved from test_expand.py)
# ---------------------------------------------------------------------------

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

class ExtractRedirectsHeredocTest(unittest.TestCase):
    """Test heredoc/here-string sentinel resolution in _extract_redirects."""

    def _extract(self, segment, expansion=None):
        return server._extract_redirects(segment, expansion)

    def test_herestring_sentinel(self) -> None:
        expansion = Expansion(
            arg_values={},
            heredoc_bodies={"\x01H0\x01": "hello\n"},
        )
        args, redirs, err = self._extract("cat <<< \x01H0\x01", expansion)
        self.assertIsNone(err)
        self.assertEqual(args, ["cat"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 0)
        self.assertEqual(redirs[0].op, "<<<")
        self.assertEqual(redirs[0].body, "hello\n")

    def test_heredoc_sentinel(self) -> None:
        expansion = Expansion(
            arg_values={},
            heredoc_bodies={"\x01H0\x01": "line1\nline2\n"},
        )
        args, redirs, err = self._extract("cat << \x01H0\x01", expansion)
        self.assertIsNone(err)
        self.assertEqual(args, ["cat"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 0)
        self.assertEqual(redirs[0].op, "<<")
        self.assertEqual(redirs[0].body, "line1\nline2\n")

    def test_heredoc_tab_strip_sentinel(self) -> None:
        expansion = Expansion(
            arg_values={},
            heredoc_bodies={"\x01H0\x01": "line1\n"},
        )
        args, redirs, err = self._extract("cat <<- \x01H0\x01", expansion)
        self.assertIsNone(err)
        self.assertEqual(args, ["cat"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 0)
        self.assertEqual(redirs[0].op, "<<-")
        self.assertEqual(redirs[0].body, "line1\n")
        self.assertTrue(redirs[0].strip_tabs)

    def test_arg_sentinel_resolved(self) -> None:
        expansion = Expansion(
            arg_values={"\x01A0\x01": "hello world"},
            heredoc_bodies={},
        )
        args, redirs, err = self._extract("echo \x01A0\x01", expansion)
        self.assertIsNone(err)
        # Arg sentinel should be resolved to the single word "hello world"
        self.assertEqual(args, ["echo", "hello world"])
        self.assertEqual(len(redirs), 0)

    def test_compound_word_sentinel_resolved(self) -> None:
        """A sentinel embedded mid-word is resolved: echo a$(echo b)c -> abc."""
        expansion = Expansion(
            arg_values={"\x01A0\x01": "b"},
            heredoc_bodies={},
        )
        args, redirs, err = self._extract("echo a\x01A0\x01c", expansion)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "abc"])

    def test_compound_word_multiple_sentinels_resolved(self) -> None:
        """Multiple sentinels in one word are each substituted in place."""
        expansion = Expansion(
            arg_values={"\x01A0\x01": "x", "\x01A1\x01": "y"},
            heredoc_bodies={},
        )
        args, redirs, err = self._extract("echo \x01A0\x01-\x01A1\x01", expansion)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "x-y"])

    def test_arg_sentinel_spaces_preserved(self) -> None:
        expansion = Expansion(
            arg_values={"\x01A0\x01": "a b c"},
            heredoc_bodies={},
        )
        args, redirs, err = self._extract("printf %s \x01A0\x01", expansion)
        self.assertIsNone(err)
        self.assertEqual(args, ["printf", "%s", "a b c"])

    def test_arg_sentinel_in_redirect_target(self) -> None:
        expansion = Expansion(
            arg_values={"\x01A0\x01": "out.txt"},
            heredoc_bodies={},
        )
        args, redirs, err = self._extract("echo hi > \x01A0\x01", expansion)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out.txt")

    def test_multiple_stdin_redirects_rejected(self) -> None:
        expansion = Expansion(
            arg_values={},
            heredoc_bodies={
                "\x01H0\x01": "body1\n",
                "\x01H1\x01": "body2\n",
            },
        )
        args, redirs, err = self._extract("cat << \x01H0\x01 << \x01H1\x01", expansion)
        self.assertIsNotNone(err)
        self.assertIn("Multiple stdin redirects", err)

    def test_bare_lt_still_rejected(self) -> None:
        args, redirs, err = self._extract("cmd < file")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 0)
        self.assertEqual(redirs[0].op, "<")
        self.assertEqual(redirs[0].raw_target, "file")

    def test_heredoc_body_not_found_error(self) -> None:
        args, redirs, err = self._extract("cat << \x01H99\x01")
        self.assertIsNotNone(err)
        self.assertIn("Heredoc body not found", err)

    def test_herestring_body_not_found_error(self) -> None:
        args, redirs, err = self._extract("cat <<< \x01H99\x01")
        self.assertIsNotNone(err)
        self.assertIn("Here-string body not found", err)

    def test_arg_sentinel_not_in_expansion_returns_literal(self) -> None:
        # Without expansion, sentinel passes through as literal
        args, redirs, err = self._extract("echo \x01A0\x01")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "\x01A0\x01"])



if __name__ == "__main__":
    unittest.main()
