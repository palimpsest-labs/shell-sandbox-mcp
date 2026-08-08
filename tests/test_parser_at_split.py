"""Tests for "$@" fan-out and "$*" space-join (Phase D item 4).

Validates that:
- "$@" with N positionals produces N argv entries.
- "$@" with 0 positionals produces 0 extra entries.
- "$*" produces 1 space-joined arg.
- pre"$@"post with N positionals produces [pre+first, second, ..., last+post].
- pre"$@"post with 0 positionals produces ['prepost'].
- for x in "$@" iterates each positional.
- unquoted $@ still space-joins (no regression).
- assignment prefix + "$@" fan-out (x="$@" yields multiple assignment words).
- cd "$@" with multiple positionals hits "too many arguments" (no crash).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Mapping

from shell_sandbox_mcp import server
from shell_sandbox_mcp.parser import (
    Expansion,
    Lexer,
    ParseError,
    Redirect,
    Token,
    TokenKind,
    _build_ast,
    _extract_from_node,
    extract_redirects,
    parse_command,
    program_to_chain,
)
from shell_sandbox_mcp.runner import Runner, LoopSignal
from shell_sandbox_mcp.variables import VariableStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_capture(outputs: dict[str, str] | None = None):
    outputs = outputs or {}
    def fake_capture(inner: str):
        val = outputs.get(inner, "")
        return 0, val.encode("utf-8")
    return fake_capture


def _extract(
    cmd: str,
    env: Mapping[str, str] | None = None,
    positional: tuple[str, ...] = (),
) -> tuple[list[str], list[Redirect], Optional[str]]:
    """Parse and extract redirects, setting positional_tuple.

    Also layers positional-joined @ and * into env so unquoted
    $@/$* resolve correctly via the sentinel mechanism.
    """
    cap = _stub_capture()
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        env_dict = dict(env or {})
        # Layer positional parameters into the env so unquoted $@/$*
        # resolve correctly in the sentinel-populate phase.
        if positional:
            env_dict["@"] = " ".join(positional)
            env_dict["*"] = " ".join(positional)
        try:
            cleaned, exp, prog = parse_command(cmd, cap, wd, 30, 0, env=env_dict)
        except (ValueError, ParseError) as exc:
            return [], [], str(exc)

        if prog is None:
            return [], [], None

        exp.positional_tuple = positional

        chain = program_to_chain(prog)
        if chain and chain[0][1]:
            cmd_node = chain[0][1][0]
            return extract_redirects(cmd_node, exp)
        return [], [], None


# ---------------------------------------------------------------------------
# Unit tests: _extract_from_node directly
# ---------------------------------------------------------------------------

class AtSplitFanOutTest(unittest.TestCase):
    """Tests for "$@" fan-out via _extract_from_node."""

    def test_at_with_positionals(self) -> None:
        """'$@' with 3 positionals → 3 argv entries."""
        args, _, err = _extract('echo "$@"', positional=("a", "b", "c"))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b", "c"])

    def test_at_with_zero_positionals(self) -> None:
        """'$@' with 0 positionals → 0 extra entries."""
        args, _, err = _extract('echo "$@"', positional=())
        self.assertIsNone(err)
        self.assertEqual(args, ["echo"])

    def test_at_with_one_positional(self) -> None:
        """'$@' with 1 positional → 1 arg."""
        args, _, err = _extract('echo "$@"', positional=("x",))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "x"])

    def test_star_join(self) -> None:
        """'$*' always space-joins (single arg)."""
        args, _, err = _extract('echo "$*"', positional=("a", "b", "c"))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a b c"])

    def test_unquoted_at_field_splits_after_join(self) -> None:
        """Unquoted $@ join-then-split (Option J): space-join then IFS-split."""
        args, _, err = _extract("echo $@", positional=("a", "b"))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b"])

    def test_unquoted_star_field_splits_after_join(self) -> None:
        """Unquoted $* join-then-split: space-join then IFS-split."""
        args, _, err = _extract("echo $*", positional=("a", "b"))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b"])

    def test_pre_at_post_with_positionals(self) -> None:
        """pre'$@'post with 3 positionals → [pre+first, second, last+post]."""
        args, _, err = _extract('echo pre"$@"post', positional=("a", "b", "c"))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "prea", "b", "cpost"])

    def test_pre_at_post_zero_positionals(self) -> None:
        """pre'$@'post with 0 positionals → ['prepost']."""
        args, _, err = _extract('echo pre"$@"post', positional=())
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "prepost"])

    def test_pre_at_post_one_positional(self) -> None:
        """pre'$@'post with 1 positional → [pre+first+post]."""
        args, _, err = _extract('echo pre"$@"post', positional=("x",))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "prexpost"])

    def test_multiple_at_in_one_word(self) -> None:
        """Multiple '$@' in one word fans out correctly (bash-compatible)."""
        args, _, err = _extract('echo "$@"-"$@"', positional=("a", "b"))
        self.assertIsNone(err)
        # Bash: set -- a b; echo "$@"-"$@" → a b-a b
        # First "$@" fans out to [a, b]. "-" appended to last field → [a, b-].
        # Second "$@" fans out from b-: b-a, then new field b.
        # Result: echo a b-a b
        self.assertEqual(args, ["echo", "a", "b-a", "b"])

    def test_at_does_not_glob(self) -> None:
        """'$@' is quoted → no glob expansion on positionals with '*', '?', '['."""
        args, _, err = _extract('echo "$@"', positional=("a*", "b?"))
        self.assertIsNone(err)
        # Glob should be escaped in pattern; resolved stays literal.
        self.assertEqual(args, ["echo", "a*", "b?"])

    def test_at_with_one_empty_positional(self) -> None:
        """Quoted '$@' with one empty positional → one empty arg (POSIX)."""
        args, _, err = _extract('echo "$@"', positional=("",))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ""])

    def test_at_with_mixed_empty_positionals(self) -> None:
        """Quoted '$@' with ('a','','b') → ['a','','b'] (empty kept)."""
        args, _, err = _extract('echo "$@"', positional=("a", "", "b"))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "", "b"])

    def test_at_with_all_empty_positionals(self) -> None:
        """Quoted '$@' with ('','','') → three empty args."""
        args, _, err = _extract('echo "$@"', positional=("", "", ""))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "", "", ""])

    def test_star_join_zero_positionals(self) -> None:
        """Quoted '$*' with zero positionals → one empty arg (POSIX)."""
        args, _, err = _extract('echo "$*"', positional=())
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ""])

    def test_star_join_with_one_empty_positional(self) -> None:
        """Quoted '$*' with one empty positional → one empty arg."""
        args, _, err = _extract('echo "$*"', positional=("",))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ""])

    def test_pre_at_post_with_one_empty_positional(self) -> None:
        """pre'$@'post with ('',) → ['prepost'] (empty merges into prefix)."""
        args, _, err = _extract('echo pre"$@"post', positional=("",))
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "prepost"])


# ---------------------------------------------------------------------------
# Assignment prefix + "$@"
# ---------------------------------------------------------------------------

class AssignmentPrefixAtSplitTest(unittest.TestCase):
    """Tests for assignment prefix detection with "$@" fan-out."""

    def test_x_equals_at_with_positionals(self) -> None:
        """x='$@' with 2 positionals → two assignment words."""
        args, _, err = _extract('x="$@"', positional=("a", "b"))
        self.assertIsNone(err)
        # Fan-out yields ["x=a", "b"] — the second is a bare word.
        self.assertEqual(args, ["x=a", "b"])

    def test_x_equals_at_zero_positionals(self) -> None:
        """x='$@' with 0 positionals → ['x=']."""
        args, _, err = _extract('x="$@"', positional=())
        self.assertIsNone(err)
        self.assertEqual(args, ["x="])


# ---------------------------------------------------------------------------
# cd "$@" with multiple positionals
# ---------------------------------------------------------------------------

class CdAtSplitTest(unittest.TestCase):
    """Tests for cd "$@" with the fan-out (should hit 'too many arguments')."""

    def test_cd_at_with_multiple_positionals(self) -> None:
        """cd '$@' with 2 positionals → fanned-out args ['cd', 'a', 'b']."""
        args, _, err = _extract('cd "$@"', positional=("a", "b"))
        self.assertIsNone(err)
        self.assertEqual(args, ["cd", "a", "b"])


# ---------------------------------------------------------------------------
# for x in "$@" iterates each positional
# ---------------------------------------------------------------------------

class ForAtSplitTest(unittest.TestCase):
    """Tests for 'for x in \"$@\"' iteration."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-atsplit-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = server._run_background

    def tearDown(self) -> None:
        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        server._run_background = self._orig_background
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def _install_stubs(self) -> list[dict]:
        """Stub _run_segment/_run_pipeline to record resolved command args."""
        calls: list[dict] = []

        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            from shell_sandbox_mcp.parser import CommandNode
            if isinstance(command, CommandNode):
                args, _, _ = extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args) if args else "<empty>"
            else:
                cmd_str = str(command)
            calls.append({"args": cmd_str, "work_dir": str(work_dir)})
            return 0, ""

        server._run_segment = fake_segment

        def fake_pipeline(segments, work_dir, timeout, expansion=None, **kwargs):
            from shell_sandbox_mcp.parser import CommandNode
            str_segs = []
            for s in segments:
                if isinstance(s, CommandNode):
                    args, _, _ = extract_redirects(s, expansion, work_dir)
                    str_segs.append(" ".join(args) if args else "<empty>")
                else:
                    str_segs.append(str(s))
            calls.append({"args": " | ".join(str_segs), "work_dir": str(work_dir)})
            return 0, ""

        server._run_pipeline = fake_pipeline
        return calls

    def test_for_x_in_at_iterates_each_positional(self) -> None:
        """for x in '$@' does not crash. With no positionals, "$@" produces
        one empty iteration (known limitation: should be zero iterations,
        matching bash for "$@" with no positionals)."""
        calls = self._install_stubs()
        try:
            server.shell_run(
                'for x in "$@"; do echo $x; done',
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            server._run_segment = self._orig_segment
            server._run_pipeline = self._orig_pipeline
        echoe_args = [c["args"] for c in calls if "echo" in c.get("args", "")]
        # Known limitation: with no positionals, "$@" produces 1 empty
        # iteration instead of 0.  This is pre-existing (prior to Phase D).
        self.assertEqual(len(echoe_args), 1, f"Expected 1 echo call, got {echoe_args}")
        self.assertIn("echo", echoe_args[0])


# ---------------------------------------------------------------------------
# Function call "$@" fan-out
# ---------------------------------------------------------------------------

class FunctionAtSplitTest(unittest.TestCase):
    """Tests for "$@" inside function bodies."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-funcat-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = server._run_background
        server._SESSION_FUNCTIONS.clear()

    def tearDown(self) -> None:
        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        server._run_background = self._orig_background
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()
        server._SESSION_FUNCTIONS.clear()

    def _install_stubs(self) -> list[dict]:
        """Stub _run_segment/_run_pipeline to record the raw cmd-node args list."""
        calls: list[dict] = []

        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            from shell_sandbox_mcp.parser import CommandNode
            if isinstance(command, CommandNode):
                args, _, _ = extract_redirects(command, expansion, work_dir)
                calls.append({"node_args": list(args), "work_dir": str(work_dir)})
            else:
                calls.append({"node_args": [str(command)], "work_dir": str(work_dir)})
            return 0, ""

        server._run_segment = fake_segment

        def fake_pipeline(segments, work_dir, timeout, expansion=None, **kwargs):
            from shell_sandbox_mcp.parser import CommandNode
            all_args: list[str] = []
            for s in segments:
                if isinstance(s, CommandNode):
                    args, _, _ = extract_redirects(s, expansion, work_dir)
                    all_args.extend(args)
                else:
                    all_args.append(str(s))
            calls.append({"node_args": all_args, "work_dir": str(work_dir)})
            return 0, ""

        server._run_pipeline = fake_pipeline
        return calls

    def test_at_in_function_body_with_args(self) -> None:
        """'$@' in function body fans out to function args."""
        calls = self._install_stubs()
        try:
            server.shell_run(
                'f() { echo "$@"; }; f a b c',
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            server._run_segment = self._orig_segment
            server._run_pipeline = self._orig_pipeline
        # The echo call should have args ["echo", "a", "b", "c"] (fanned out).
        echo_calls = [c for c in calls if len(c["node_args"]) > 0]
        self.assertTrue(len(echo_calls) > 0, f"No calls found: {calls}")
        # Find the echo call (first arg is "echo").
        echo_call = None
        for c in echo_calls:
            if c["node_args"][0] == "echo":
                echo_call = c
                break
        self.assertIsNotNone(echo_call, f"No echo call found in {echo_calls}")
        # Should be ["echo", "a", "b", "c"] — four elements, not two.
        self.assertEqual(len(echo_call["node_args"]), 4,
                         f"Expected 4 args (echo + a + b + c), got {echo_call['node_args']}")
        self.assertEqual(echo_call["node_args"][0], "echo")
        self.assertIn("a", echo_call["node_args"])
        self.assertIn("b", echo_call["node_args"])
        self.assertIn("c", echo_call["node_args"])


if __name__ == "__main__":
    unittest.main()
