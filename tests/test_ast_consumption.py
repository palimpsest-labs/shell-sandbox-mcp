"""Tests that the live path parses once and threads the AST. Run with the venv python that has `mcp` installed:

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
# AST consumption tests — prove the live path parses once and threads the
# CommandNode through to _extract_redirects / _build_invocation without
# re-lexing.
# ---------------------------------------------------------------------------


class ASTConsumptionTest(unittest.TestCase):
    """Prove the live shell_run path consumes the AST without double-lex.

    Uses monkey-patching to spy on internal functions and assert that:
    1. ``_build_invocation`` receives a ``CommandNode`` (not a ``str``) from
       the live path.
    2. ``_extract_redirects`` is called with a ``CommandNode`` via the AST
       projection path (``_extract_from_node``).
    3. The ``split_legacy`` function is NOT called from the live path
       (proving the double-lex is eliminated).
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-ast-" + os.urandom(4).hex())
        self.allowed.mkdir()
        # Save originals
        self._orig_build = server._build_invocation
        self._orig_extract = server._extract_redirects
        self._orig_split_legacy = server.split_legacy

    def tearDown(self) -> None:
        import shutil
        server._build_invocation = self._orig_build
        server._extract_redirects = self._orig_extract
        server.split_legacy = self._orig_split_legacy
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def test_build_invocation_receives_commandnode_from_live_path(self) -> None:
        """shell_run threads CommandNode through to _build_invocation."""
        received_types: list[type] = []

        def spy_build(command, work_dir, expansion=None):
            received_types.append(type(command))
            # Return error to short-circuit (avoid actual sandbox)
            return server.InvocationError("spy")

        server._build_invocation = spy_build
        server.shell_run("echo hi", cwd=str(self.allowed))

        self.assertTrue(
            len(received_types) > 0,
            "_build_invocation was never called",
        )
        self.assertIn(
            CommandNode,
            received_types,
            f"Expected CommandNode in received_types, got {received_types}",
        )

    def test_extract_redirects_receives_commandnode_from_live_path(self) -> None:
        """_extract_redirects receives CommandNode via AST projection."""
        received_types: list[type] = []

        def spy_extract(segment, expansion=None, work_dir=None):
            received_types.append(type(segment))
            # Return valid empty result
            return ["echo", "hi"], [], None

        server._extract_redirects = spy_extract

        # Also stub _resolve_command to avoid the real allowlist path
        orig_resolve = server._resolve_command
        try:
            def fake_resolve(args, work_dir):
                return "/bin/echo", ["/bin/echo", "hi"], server.COMMANDS.get("echo", {"promises": "stdio"})
            server._resolve_command = fake_resolve
            server.shell_run("echo hi", cwd=str(self.allowed))
        finally:
            server._resolve_command = orig_resolve

        self.assertTrue(
            len(received_types) > 0,
            "_extract_redirects was never called",
        )
        self.assertIn(
            CommandNode,
            received_types,
            f"Expected CommandNode in received_types, got {received_types}",
        )

    def test_split_legacy_not_called_from_live_path(self) -> None:
        """split_legacy is NOT invoked from the AST live path."""
        call_count = [0]
        orig_split = server.split_legacy

        def counting_split(command):
            call_count[0] += 1
            return orig_split(command)

        server.split_legacy = counting_split
        # Stub _build_invocation to short-circuit
        def _fake_build(command, work_dir, expansion=None):
            return server.InvocationError("spy")
        server._build_invocation = _fake_build

        try:
            server.shell_run("echo hi", cwd=str(self.allowed))
        finally:
            server.split_legacy = orig_split

        # split_legacy should NOT be called from the live AST path
        self.assertEqual(
            call_count[0], 0,
            f"split_legacy was called {call_count[0]} times from the live path; "
            "double-lex is still present!",
        )

    def test_expand_command_returns_programnode(self) -> None:
        """_expand_command returns a non-None ProgramNode for valid input."""
        from shell_sandbox_mcp.server import _expand_command

        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            # Stub capture to avoid real subprocess
            orig_capture = server._capture_stdout
            server._capture_stdout = lambda cmd, wd2, to, d, dl=None, sc=None: (0, b"test")
            try:
                _cleaned, _exp, program = _expand_command(
                    "echo hi", wd, 30, 0,
                )
                self.assertIsNotNone(
                    program,
                    "ProgramNode should not be None for valid input",
                )
                self.assertIsInstance(
                    program, ProgramNode,
                    "Returned object should be a ProgramNode",
                )
            finally:
                server._capture_stdout = orig_capture

    def test_program_to_chain_projection(self) -> None:
        """program_to_chain correctly projects AST to legacy chain format."""
        from shell_sandbox_mcp.parser import (
            AndOrNode,
            CommandNode as PCmd,
            PipelineNode,
            ProgramNode,
            Word,
            WordPart,
            program_to_chain,
        )
        cmd = PCmd(words=(Word(parts=(WordPart(text="ls", raw="ls"),)),), redirects=())
        pipeline = PipelineNode(commands=(cmd,))
        chain = AndOrNode(operator=None, pipeline=pipeline, backgrounded=False)
        program = ProgramNode(chains=(chain,))

        result = program_to_chain(program)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], None)  # operator
        self.assertEqual(result[0][2], False)  # backgrounded
        self.assertEqual(len(result[0][1]), 1)  # one CommandNode



class TildeExpansionTest(unittest.TestCase):
    """Tilde expansion of command args and file-redirect targets.

    Expansion happens in ``parser._extract_from_node``: command words and
    ``>``/``>>``/``<`` redirect targets that begin with ``~`` are expanded
    via ``Path.expanduser``.  Heredoc/here-string delimiters and ``>&`` fd
    targets are left untouched.
    """

    def _expand(self, command: str) -> "tuple[list[str], list[Redirect], str | None]":
        """Parse *command* and run it through ``_extract_from_node``."""
        from shell_sandbox_mcp.parser import (
            parse_command,
            program_to_chain,
            _extract_from_node,
        )
        cleaned, _exp, program = parse_command(
            command, lambda i: (0, b""), Path("."), 30, 0,
        )
        self.assertIsNotNone(program, f"parse_command rejected: {command!r}")
        chain = program_to_chain(program)
        self.assertTrue(chain and chain[0][1], f"no command chain for {command!r}")
        return _extract_from_node(chain[0][1][0], _exp)

    def test_tilde_arg_alone(self) -> None:
        args, redirects, err = self._expand("echo ~")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", str(Path.home())])
        self.assertEqual(redirects, [])

    def test_tilde_slash_arg(self) -> None:
        args, redirects, err = self._expand("echo ~/x")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", str(Path.home() / "x")])
        self.assertEqual(redirects, [])

    def test_tilde_current_user_arg(self) -> None:
        args, redirects, err = self._expand("echo ~arch")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", str(Path.home())])
        self.assertEqual(redirects, [])

    def test_tilde_redirect_target(self) -> None:
        args, redirects, err = self._expand("cat < ~/x")
        self.assertIsNone(err)
        self.assertEqual(args, ["cat"])
        self.assertEqual(len(redirects), 1)
        self.assertEqual(redirects[0].op, "<")
        self.assertEqual(redirects[0].raw_target, str(Path.home() / "x"))

    def test_tilde_output_redirect_target(self) -> None:
        args, redirects, err = self._expand("cmd > ~/out.txt")
        self.assertIsNone(err)
        self.assertEqual(len(redirects), 1)
        self.assertEqual(redirects[0].op, ">")
        self.assertEqual(redirects[0].raw_target, str(Path.home() / "out.txt"))

    def test_heredoc_delimiter_not_expanded(self) -> None:
        args, redirects, err = self._expand("cat << EOF\nbody\nEOF")
        self.assertIsNone(err)
        self.assertEqual(len(redirects), 1)
        self.assertEqual(redirects[0].op, "<<")
        self.assertEqual(redirects[0].body, "body\n")

    def test_herestring_not_expanded(self) -> None:
        args, redirects, err = self._expand("cmd <<< ~/x")
        self.assertIsNone(err)
        self.assertEqual(len(redirects), 1)
        self.assertEqual(redirects[0].op, "<<<")
        self.assertEqual(redirects[0].body, "~/x\n")

    def test_fd_redirect_target_not_expanded(self) -> None:
        from shell_sandbox_mcp.parser import (
            CommandNode as ParserCmd,
            RedirectSpec,
            Word,
            WordPart,
            _extract_from_node,
        )
        # Build a >& redirect with a tilde target manually — the lexer only
        # accepts digit fd targets for >&, so a manual node is needed.
        cmd = ParserCmd(
            words=(Word(parts=(WordPart(text="cmd", raw="cmd"),)),),
            redirects=(RedirectSpec(
                fd=2, op=">&",
                target=Word(parts=(WordPart(text="~foo", raw="~foo"),)),
            ),),
        )
        args, redirects, err = _extract_from_node(cmd)
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirects), 1)
        self.assertEqual(redirects[0].op, ">&")
        self.assertEqual(redirects[0].raw_target, "~foo")
        self.assertIsNone(redirects[0].target_fd)

    def test_middle_of_word_tilde_not_expanded(self) -> None:
        args, redirects, err = self._expand("echo foo~bar")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "foo~bar"])
        self.assertEqual(redirects, [])

    def test_tilde_not_expanded_in_redirect_middle(self) -> None:
        args, redirects, err = self._expand("cmd > a~b")
        self.assertIsNone(err)
        self.assertEqual(len(redirects), 1)
        self.assertEqual(redirects[0].op, ">")
        self.assertEqual(redirects[0].raw_target, "a~b")


if __name__ == "__main__":
    unittest.main()
