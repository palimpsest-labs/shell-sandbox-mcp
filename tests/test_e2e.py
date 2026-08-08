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


def _find_hd_sentinel(prog):
    """Return the first heredoc-sentinel WordPart in the first command."""
    cmd = prog.chains[0].pipeline.commands[0]
    for rs in cmd.redirects:
        for p in rs.target.parts:
            if p.is_hd_sentinel:
                return p
    return None


def _find_arg_sentinel(prog):
    """Return the first arg-sentinel WordPart in the first command."""
    cmd = prog.chains[0].pipeline.commands[0]
    for w in cmd.words:
        for p in w.parts:
            if p.is_arg_sentinel:
                return p
    return None


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
        expanded, exp, prog = _expand_command(cmd, self.work_dir, 30, 0)
        # Verify the expanded command has a heredoc sentinel
        self.assertIn("<<", expanded)
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "hello\n")
        # Use AST path for extract_redirects — no sentinel reconstruction
        cmd_node = prog.chains[0].pipeline.commands[0]
        args, redirs, err = _extract_redirects(cmd_node, expansion=exp)
        self.assertIsNone(err)
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].body, "hello\n")

    def test_heredoc_single_quoted_literal(self) -> None:
        """Verify single-quoted delimiters produce literal bodies."""
        cmd = "cat <<'EOF'\n$(echo hi)\nEOF"
        expanded, exp, prog = _expand_command(cmd, self.work_dir, 30, 0)
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "$(echo hi)\n")

    def test_herestring_expansion(self) -> None:
        """Verify here-string produces correct body."""
        cmd = "cat <<<hello world"
        expanded, exp, prog = _expand_command(cmd, self.work_dir, 30, 0)
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "hello world\n")

    def test_command_substitution_single_arg(self) -> None:
        """Verify $(...) with space-containing output is field-split (IFS)."""
        original = server._capture_stdout
        try:
            def fake(command, work_dir, timeout, depth, deadline=None, subst_count=None, env=None):
                return 0, b"a b"
            server._capture_stdout = fake
            cmd = "echo $(printf 'a b')"
            expanded, exp, prog = _expand_command(cmd, self.work_dir, 30, 0)
            part = _find_arg_sentinel(prog)
            self.assertIsNotNone(part)
            self.assertEqual(exp.arg_for(part), "a b")
            # Unquoted $(...) → field-split by default IFS.
            cmd_node = prog.chains[0].pipeline.commands[0]
            args, redirs, err = _extract_redirects(cmd_node, expansion=exp)
            self.assertEqual(args, ["echo", "a", "b"])
        finally:
            server._capture_stdout = original

    def test_command_substitution_single_arg_quoted(self) -> None:
        """Quoted "$(printf 'a b')" is NOT field-split (single arg)."""
        original = server._capture_stdout
        try:
            def fake(command, work_dir, timeout, depth, deadline=None, subst_count=None, env=None):
                return 0, b"a b"
            server._capture_stdout = fake
            cmd = 'echo "$(printf \'a b\')"'
            expanded, exp, prog = _expand_command(cmd, self.work_dir, 30, 0)
            part = _find_arg_sentinel(prog)
            self.assertIsNotNone(part)
            self.assertEqual(exp.arg_for(part), "a b")
            cmd_node = prog.chains[0].pipeline.commands[0]
            args, redirs, err = _extract_redirects(cmd_node, expansion=exp)
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
            expanded, exp, prog = _expand_command(cmd, self.work_dir, 30, 0)
            part = _find_arg_sentinel(prog)
            self.assertIsNotNone(part)
            self.assertEqual(exp.arg_for(part), "x")
        finally:
            server._capture_stdout = original


class TestSubprocessGit(unittest.TestCase):
    """End-to-end tests: subprocess-spawned git through xargs/find/python3/scripts.

    These go through the real sandbox (shell_run) and verify that commands
    which spawn subprocesses can run git successfully, while never being able
    to read ~/.git-credentials.

    When running inside a sandbox (sandbox-in-sandbox), the inner sandbox
    binary cannot start — all tests are skipped in that environment.  The
    real verification is done via the MCP shell_run tool from the host."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)
        # Detect sandbox-in-sandbox: if the inner sandbox binary cannot start
        # (APE loader EACCES), skip all tests.
        probe = server.shell_run(
            "echo __sbx_probe__", cwd=str(self.work_dir),
            timeout=5, structured=True,
        )
        assert isinstance(probe, dict)
        self._sandbox_broken = (
            probe["rc"] != 0
            and "sandbox: line 12: /home/arch/.ape" in probe.get("output", "")
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _require_sandbox(self) -> None:
        if self._sandbox_broken:
            self.skipTest("sandbox-in-sandbox: inner sandbox binary cannot start")

    def _run(self, command: str, timeout: int = 30) -> str:
        return server.shell_run(command, cwd=str(self.work_dir), timeout=timeout)

    def _run_structured(self, command: str, timeout: int = 30) -> dict:
        result = server.shell_run(command, cwd=str(self.work_dir),
                                  timeout=timeout, structured=True)
        assert isinstance(result, dict)
        return result

    def _init_git_repo(self) -> None:
        """Create a git repo in the temp work dir so git status succeeds."""
        out = self._run("git init")
        # git init may print "Initialized empty Git repository in ..."
        self.assertIn("Initialized", out)

    # ------------------------------------------------------------------
    # Positive: git status works through subprocess-spawning wrappers
    # ------------------------------------------------------------------

    def test_xargs_git_status(self) -> None:
        self._require_sandbox()
        self._init_git_repo()
        out = self._run("printf '.' | xargs git status")
        # rc 0 (no "Exit code:" line), and output mentions a known phrase
        self.assertNotIn("Exit code:", out)
        self.assertTrue(
            "nothing to commit" in out or "No commits yet" in out or
            "On branch" in out,
            f"unexpected git status output: {out}",
        )

    def test_find_exec_git_status(self) -> None:
        self._require_sandbox()
        self._init_git_repo()
        out = self._run("find . -maxdepth 1 -exec git status \\;")
        self.assertNotIn("Exit code:", out)
        self.assertTrue(
            "nothing to commit" in out or "No commits yet" in out or
            "On branch" in out,
            f"unexpected git status output: {out}",
        )

    def test_python_subprocess_git_status(self) -> None:
        self._require_sandbox()
        self._init_git_repo()
        out = self._run(
            "python3 -c "
            "'import subprocess,sys; sys.exit(subprocess.run([\"git\",\"status\"]).returncode)'"
        )
        # rc 0 -> no "Exit code:" line (git status succeeded)
        self.assertNotIn("Exit code:", out)

    def test_shell_script_git_status(self) -> None:
        self._require_sandbox()
        self._init_git_repo()
        script = self.work_dir / "s.sh"
        script.write_text("#!/bin/sh\ngit status\n")
        script.chmod(0o755)
        out = self._run("./s.sh")
        self.assertNotIn("Exit code:", out)
        self.assertTrue(
            "nothing to commit" in out or "No commits yet" in out or
            "On branch" in out,
            f"unexpected git status output: {out}",
        )

    # ------------------------------------------------------------------
    # Negative: ~/.git-credentials is NEVER readable by subprocess spawners
    # ------------------------------------------------------------------

    _CRED_PATH = str((Path.home().resolve() / ".git-credentials"))

    def test_xargs_cannot_read_credentials(self) -> None:
        """xargs cat on ~/.git-credentials must fail (unveil blocks it)."""
        self._require_sandbox()
        cred_file = Path(self._CRED_PATH)
        if not cred_file.exists():
            self.skipTest(f"{self._CRED_PATH} does not exist")
        result = self._run_structured(
            f"echo {self._CRED_PATH} | xargs cat"
        )
        self.assertNotEqual(result["rc"], 0,
                            "xargs cat on ~/.git-credentials must be blocked")

    def test_python_cannot_read_credentials(self) -> None:
        """python3 open(~/.git-credentials) must fail (unveil blocks it)."""
        self._require_sandbox()
        cred_file = Path(self._CRED_PATH)
        if not cred_file.exists():
            self.skipTest(f"{self._CRED_PATH} does not exist")
        result = self._run_structured(
            "python3 -c 'open(\"/home/arch/.git-credentials\").read()'"
        )
        self.assertNotEqual(result["rc"], 0,
                            "python open of ~/.git-credentials must be blocked")


if __name__ == "__main__":
    unittest.main()
