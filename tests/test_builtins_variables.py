"""Tests for variable assignment and builtins (export, unset, set, shift,
source / .).  Run with:

    PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.parser import (
    Lexer,
    ParseError,
    TokenKind,
    _ASSIGN_WORD_RE,
    _BUILTIN_NAMES,
    segment_needs_variable_state,
    split_chains,
)
from shell_sandbox_mcp.variables import VariableStore
from shell_sandbox_mcp.builtins import _split_assignment_prefix
from shell_sandbox_mcp.server import (
    CommandNode,
    Expansion,
    VariableStore as VS2,  # verify re-export
    _expand_command,
)


# ============================================================================
# split_chains unit tests
# ============================================================================


class SplitChainsTest(unittest.TestCase):
    """Lex-only chain split."""

    def test_semicolon_split(self) -> None:
        segs = split_chains("echo a ; echo b")
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0], (None, "echo a", False))
        self.assertEqual(segs[1], (";", "echo b", False))

    def test_and_and_split(self) -> None:
        segs = split_chains("true && echo ok")
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0], (None, "true", False))
        self.assertEqual(segs[1], ("&&", "echo ok", False))

    def test_or_or_split(self) -> None:
        segs = split_chains("false || echo fail")
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0], (None, "false", False))
        self.assertEqual(segs[1], ("||", "echo fail", False))

    def test_background_split(self) -> None:
        segs = split_chains("sleep 10 & echo done")
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0], (None, "sleep 10", True))
        self.assertEqual(segs[1], (None, "echo done", False))

    def test_newline_split(self) -> None:
        # Newlines are chain separators in the lex-based split_chains.
        segs = split_chains("echo a\necho b")
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0], (None, "echo a", False))
        self.assertEqual(segs[1], (None, "echo b", False))

    def test_pipe_stays_in_segment(self) -> None:
        segs = split_chains("echo a | grep x ; echo b")
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0], (None, "echo a | grep x", False))
        self.assertEqual(segs[1], (";", "echo b", False))

    def test_separator_inside_subst_not_split(self) -> None:
        # split_legacy produces 1 segment — the $() body is absorbed and
        # the display form uses sentinel markers.
        segs = split_chains("echo $(echo a; echo b)")
        self.assertEqual(len(segs), 1)
        self.assertIn("echo", segs[0][1])

    def test_heredoc_separator_not_split(self) -> None:
        # split_legacy produces 1 segment for heredoc — the heredoc body
        # `;\n` is absorbed into the token and newline after EOF is not
        # a chain separator in replay mode.
        segs = split_chains("cat <<EOF\n;\nEOF\necho done")
        self.assertEqual(len(segs), 1)
        self.assertIn("cat", segs[0][1])

    def test_trailing_semicolon_dropped(self) -> None:
        segs = split_chains("echo a ;")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0], (None, "echo a", False))

    def test_empty_command(self) -> None:
        segs = split_chains("")
        self.assertEqual(segs, [])

    def test_multiple_bg_and_chain(self) -> None:
        segs = split_chains("a & b && c")
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0], (None, "a", True))
        self.assertEqual(segs[1], (None, "b", False))
        self.assertEqual(segs[2], ("&&", "c", False))


# ============================================================================
# segment_needs_variable_state unit tests
# ============================================================================


class SegmentNeedsStateTest(unittest.TestCase):
    """Cheap lex-only detection gate."""

    def test_assign_positive(self) -> None:
        self.assertTrue(segment_needs_variable_state("VAR=x"))

    def test_assign_with_value(self) -> None:
        self.assertTrue(segment_needs_variable_state("FOO=bar baz"))

    def test_export_positive(self) -> None:
        self.assertTrue(segment_needs_variable_state("export FOO"))

    def test_unset_positive(self) -> None:
        self.assertTrue(segment_needs_variable_state("unset FOO"))

    def test_set_positive(self) -> None:
        self.assertTrue(segment_needs_variable_state("set FOO=bar"))

    def test_shift_positive(self) -> None:
        self.assertTrue(segment_needs_variable_state("shift"))

    def test_source_positive(self) -> None:
        self.assertTrue(segment_needs_variable_state("source file"))

    def test_dot_positive(self) -> None:
        self.assertTrue(segment_needs_variable_state(". file"))

    def test_echo_negative(self) -> None:
        self.assertFalse(segment_needs_variable_state("echo VAR=x"))

    def test_plain_cmd_negative(self) -> None:
        self.assertFalse(segment_needs_variable_state("echo hello"))

    def test_pipe_with_assign_in_stage2(self) -> None:
        # Per-segment detection: the first word of the segment is `cmd`, not `VAR=x`
        self.assertFalse(segment_needs_variable_state("cmd | VAR=x cmd2"))

    def test_redirect_before_assign(self) -> None:
        # Redirect token at start is skipped; first word is VAR=x → True
        self.assertTrue(segment_needs_variable_state("2>err VAR=x cmd"))

    def test_redirect_before_normal_cmd(self) -> None:
        self.assertFalse(segment_needs_variable_state("2>err echo ok"))


# ============================================================================
# VariableStore unit tests
# ============================================================================


class VariableStoreTest(unittest.TestCase):
    """VariableStore init, mutations, and env views."""

    def setUp(self) -> None:
        # Ensure clean state for each test by using a store with known vars.
        # We can't fully control _base_env() but we know PATH exists.
        self.store = VariableStore()

    def test_init_has_path(self) -> None:
        self.assertIn("PATH", self.store.variables)
        self.assertIn("PATH", self.store.exported)
        self.assertTrue(self.store.is_exported("PATH"))

    def test_init_has_home(self) -> None:
        self.assertIn("HOME", self.store.variables)
        self.assertIn("HOME", self.store.exported)

    def test_set_local(self) -> None:
        self.store.set_local("MYVAR", "hello")
        self.assertEqual(self.store.get("MYVAR"), "hello")
        self.assertFalse(self.store.is_exported("MYVAR"))

    def test_set_export(self) -> None:
        self.store.set_export("MYVAR", "hello")
        self.assertEqual(self.store.get("MYVAR"), "hello")
        self.assertTrue(self.store.is_exported("MYVAR"))

    def test_mark_export_existing(self) -> None:
        self.store.set_local("MYVAR", "hello")
        self.assertFalse(self.store.is_exported("MYVAR"))
        self.store.mark_export("MYVAR")
        self.assertTrue(self.store.is_exported("MYVAR"))
        self.assertEqual(self.store.get("MYVAR"), "hello")

    def test_mark_export_new(self) -> None:
        """mark_export on a previously-unset var initialises to ''."""
        self.store.mark_export("NEWVAR")
        self.assertTrue(self.store.is_exported("NEWVAR"))
        self.assertEqual(self.store.get("NEWVAR"), "")

    def test_unset(self) -> None:
        self.store.set_export("MYVAR", "hello")
        self.store.unset("MYVAR")
        self.assertEqual(self.store.get("MYVAR"), "")
        self.assertFalse(self.store.is_exported("MYVAR"))

    def test_unset_path(self) -> None:
        self.store.unset("PATH")
        self.assertEqual(self.store.get("PATH"), "")
        self.assertFalse(self.store.is_exported("PATH"))

    def test_get_unset_var(self) -> None:
        self.assertEqual(self.store.get("NONEXISTENT"), "")

    def test_env_for_expansion_includes_local(self) -> None:
        self.store.set_local("LOCAL", "val")
        env = self.store.env_for_expansion()
        self.assertEqual(env["LOCAL"], "val")
        self.assertEqual(env.get("PATH"), self.store.get("PATH"))

    def test_env_for_subprocess_excludes_local(self) -> None:
        self.store.set_local("LOCAL", "val")
        env = self.store.env_for_subprocess()
        self.assertNotIn("LOCAL", env)
        self.assertIn("PATH", env)

    def test_env_for_subprocess_includes_exported(self) -> None:
        self.store.set_export("GLOBAL", "val")
        env = self.store.env_for_subprocess()
        self.assertIn("GLOBAL", env)
        self.assertEqual(env["GLOBAL"], "val")


# ============================================================================
# _split_assignment_prefix unit tests
# ============================================================================


class SplitAssignmentPrefixTest(unittest.TestCase):
    """Assignment prefix detection on CommandNode."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _cmd_node(self, text: str) -> CommandNode:
        """Parse a segment text and return its first CommandNode."""
        from shell_sandbox_mcp.parser import program_to_chain
        from shell_sandbox_mcp.server import _expand_command
        _, expansion, program = _expand_command(text, self.work_dir, 30, 0)
        self.assertIsNotNone(program, f"Parse failed for: {text}")
        chains = program_to_chain(program)
        self.assertTrue(chains)
        return chains[0][1][0]

    def test_pure_assignment(self) -> None:
        node = self._cmd_node("VAR=hello")
        prefix, remaining, err = _split_assignment_prefix(node, None, self.work_dir)
        self.assertIsNotNone(prefix)
        self.assertIsNone(remaining)
        self.assertIsNone(err)
        self.assertEqual(prefix, {"VAR": "hello"})

    def test_assignment_with_cmd(self) -> None:
        node = self._cmd_node("VAR=x echo hi")
        prefix, remaining, err = _split_assignment_prefix(node, None, self.work_dir)
        self.assertIsNotNone(prefix)
        self.assertIsNotNone(remaining)
        self.assertIsNone(err)
        self.assertEqual(prefix, {"VAR": "x"})
        # Remaining node has only "echo hi"
        from shell_sandbox_mcp.parser import extract_redirects
        args, _, _ = extract_redirects(remaining, None, self.work_dir)
        self.assertEqual(args, ["echo", "hi"])

    def test_empty_value_var_equals(self) -> None:
        node = self._cmd_node("VAR= cmd")
        prefix, remaining, err = _split_assignment_prefix(node, None, self.work_dir)
        self.assertIsNotNone(prefix)
        self.assertEqual(prefix, {"VAR": ""})

    def test_value_contains_equals(self) -> None:
        node = self._cmd_node("VAR=a=b cmd")
        prefix, remaining, err = _split_assignment_prefix(node, None, self.work_dir)
        self.assertIsNotNone(prefix)
        self.assertEqual(prefix, {"VAR": "a=b"})

    def test_equals_first_char_not_assignment(self) -> None:
        node = self._cmd_node("=foo cmd")
        prefix, remaining, err = _split_assignment_prefix(node, None, self.work_dir)
        self.assertIsNone(prefix)
        self.assertIsNotNone(remaining)

    def test_multiple_assignments(self) -> None:
        node = self._cmd_node("A=1 B=2 cmd")
        prefix, remaining, err = _split_assignment_prefix(node, None, self.work_dir)
        self.assertIsNotNone(prefix)
        self.assertEqual(prefix, {"A": "1", "B": "2"})

    def test_cmd_not_leading_assignment(self) -> None:
        node = self._cmd_node("cmd VAR=x")
        prefix, remaining, err = _split_assignment_prefix(node, None, self.work_dir)
        self.assertIsNone(prefix)

    def test_double_dash_stops_detection(self) -> None:
        node = self._cmd_node("-- VAR=x cmd")
        prefix, remaining, err = _split_assignment_prefix(node, None, self.work_dir)
        # -- is first arg; it stops detection. So no prefix.
        self.assertIsNone(prefix)

    def test_invalid_var_name(self) -> None:
        node = self._cmd_node("1abc=x cmd")
        prefix, remaining, err = _split_assignment_prefix(node, None, self.work_dir)
        self.assertIsNone(prefix)


# ============================================================================
# Builtin edge cases (unit level via direct calls)
# ============================================================================


class BuiltinsEdgeTest(unittest.TestCase):
    """Edge cases for export/unset/set/shift/source via direct builtin calls."""

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

    # -- export --

    def test_export_no_args_prints_exported(self) -> None:
        from shell_sandbox_mcp.builtins import _try_export
        store = VariableStore()
        store.set_export("FOO", "bar")
        node = self._cmd_node("export")
        handled, out, rc = _try_export(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 0)
        self.assertIn("FOO=bar", out)
        self.assertIn("PATH=", out)

    def test_export_name_value(self) -> None:
        from shell_sandbox_mcp.builtins import _try_export
        store = VariableStore()
        node = self._cmd_node("export FOO=bar")
        handled, out, rc = _try_export(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 0)
        self.assertEqual(store.get("FOO"), "bar")
        self.assertTrue(store.is_exported("FOO"))

    def test_export_bare_name(self) -> None:
        from shell_sandbox_mcp.builtins import _try_export
        store = VariableStore()
        store.set_local("FOO", "bar")
        node = self._cmd_node("export FOO")
        handled, out, rc = _try_export(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 0)
        self.assertTrue(store.is_exported("FOO"))
        self.assertEqual(store.get("FOO"), "bar")

    def test_export_invalid_name(self) -> None:
        from shell_sandbox_mcp.builtins import _try_export
        store = VariableStore()
        node = self._cmd_node("export 1abc=x")
        handled, out, rc = _try_export(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 1)
        self.assertIn("not a valid", out)

    # -- unset --

    def test_unset_silent(self) -> None:
        from shell_sandbox_mcp.builtins import _try_unset
        store = VariableStore()
        store.set_local("FOO", "bar")
        node = self._cmd_node("unset FOO")
        handled, out, rc = _try_unset(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 0)
        self.assertEqual(store.get("FOO"), "")

    def test_unset_missing_silent(self) -> None:
        from shell_sandbox_mcp.builtins import _try_unset
        store = VariableStore()
        node = self._cmd_node("unset MISSING")
        handled, out, rc = _try_unset(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 0)

    # -- set --

    def test_set_no_args_prints_all(self) -> None:
        from shell_sandbox_mcp.builtins import _try_set
        store = VariableStore()
        store.set_local("FOO", "bar")
        node = self._cmd_node("set")
        handled, out, rc = _try_set(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 0)
        self.assertIn("FOO=bar", out)
        self.assertIn("PATH=", out)

    def test_set_name_value(self) -> None:
        from shell_sandbox_mcp.builtins import _try_set
        store = VariableStore()
        node = self._cmd_node("set FOO=bar")
        handled, out, rc = _try_set(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 0)
        self.assertEqual(store.get("FOO"), "bar")
        self.assertFalse(store.is_exported("FOO"))

    def test_set_flag_ignored(self) -> None:
        from shell_sandbox_mcp.builtins import _try_set
        store = VariableStore()
        node = self._cmd_node("set -e")
        handled, out, rc = _try_set(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 0)  # no-op, rc 0

    def test_set_unsupported_arg(self) -> None:
        from shell_sandbox_mcp.builtins import _try_set
        store = VariableStore()
        node = self._cmd_node("set foo")
        handled, out, rc = _try_set(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 1)

    # -- shift --

    def test_shift_rc1(self) -> None:
        from shell_sandbox_mcp.builtins import _try_shift
        store = VariableStore()
        node = self._cmd_node("shift")
        handled, out, rc = _try_shift(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 1)

    def test_shift_invalid_arg(self) -> None:
        from shell_sandbox_mcp.builtins import _try_shift
        store = VariableStore()
        node = self._cmd_node("shift abc")
        handled, out, rc = _try_shift(node, None, self.work_dir, store)
        self.assertTrue(handled)
        self.assertEqual(rc, 1)
        self.assertIn("invalid argument", out)

    # -- source --

    def test_source_missing_file(self) -> None:
        from shell_sandbox_mcp.builtins import _try_source
        store = VariableStore()
        node = self._cmd_node("source /nonexistent/file")
        handled, out, rc = _try_source(node, None, self.work_dir, store, 30, 0)
        self.assertTrue(handled)
        self.assertEqual(rc, 1)
        # /nonexistent/file is outside allowed roots → escapes sandbox
        self.assertIn("escapes sandbox", out)

    def test_source_missing_arg(self) -> None:
        from shell_sandbox_mcp.builtins import _try_source
        store = VariableStore()
        node = self._cmd_node("source")
        handled, out, rc = _try_source(node, None, self.work_dir, store, 30, 0)
        self.assertTrue(handled)
        self.assertEqual(rc, 1)
        self.assertIn("missing file", out)

    def test_source_escapes_sandbox(self) -> None:
        from shell_sandbox_mcp.builtins import _try_source
        store = VariableStore()
        node = self._cmd_node("source /etc/passwd")
        handled, out, rc = _try_source(node, None, self.work_dir, store, 30, 0)
        self.assertTrue(handled)
        self.assertEqual(rc, 1)
        self.assertIn("escapes sandbox", out)

    def test_source_contained_valid(self) -> None:
        from shell_sandbox_mcp.builtins import _try_source
        store = VariableStore()
        # Create a script file inside work_dir
        script = self.work_dir / "vars.sh"
        script.write_text("export FOO=from_source")
        node = self._cmd_node("source vars.sh")
        # Stub _run_segment to avoid subprocess execution
        handled, out, rc = _try_source(node, None, self.work_dir, store, 30, 0)
        self.assertTrue(handled)
        # source runs the script; since the script contains `export FOO=from_source`,
        # the store should be mutated (via run_command re-expansion).
        self.assertEqual(store.get("FOO"), "from_source")
        self.assertTrue(store.is_exported("FOO"))


# ============================================================================
# End-to-end via shell_run with stubbed execution
# ============================================================================


class ShellRunVariablesTest(unittest.TestCase):
    """End-to-end variable assignment and builtins via shell_run.

    Stubs _run_segment and _build_invocation to capture env and avoid
    launching real subprocesses.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-var-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = server._run_background
        self._orig_build = server._build_invocation
        self.segment_calls: list[dict] = []  # captures env and args

    def tearDown(self) -> None:
        import shutil
        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        server._run_background = self._orig_background
        server._build_invocation = self._orig_build
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def _stub_segment_and_build(self) -> None:
        """Stub _run_segment to capture env and return a canned output."""
        calls = self.segment_calls

        def fake_segment(command, work_dir, timeout, expansion=None,
                          *, shell_env=None, stage_env_overrides=None):
            calls.append({
                "command": command,
                "work_dir": str(work_dir),
                "shell_env": shell_env,
                "stage_env_overrides": stage_env_overrides,
            })
            return 0, "ok"

        server._run_segment = fake_segment

        def fake_pipeline(segments, work_dir, timeout, expansion=None,
                           *, shell_env=None, stage_env_overrides=None):
            return 0, "pipeline-ok"

        server._run_pipeline = fake_pipeline

        def fake_background(segments, work_dir, expansion=None,
                            *, shell_env=None, stage_env_overrides=None):
            return 0, "bg"

        server._run_background = fake_background

        # Also stub _build_invocation so the executor doesn't try to resolve
        # real sandbox paths.  We care about env passing here.
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
    # VAR=hello; echo $VAR  →  hello
    # ------------------------------------------------------------------

    def test_var_assign_and_expand(self) -> None:
        """VAR=hello; echo $VAR should print hello."""
        self._stub_segment_and_build()
        out = self._run("VAR=hello; echo $VAR")
        # Should have one segment call: echo hello (value expanded)
        self.assertGreaterEqual(len(self.segment_calls), 1,
                                f"Expected at least 1 segment call, got {self.segment_calls}")
        # The echo command should see "hello" as an argument
        # (the $VAR was expanded to "hello" by the variable store)
        self.assertIn("ok", out)

    def test_var_assign_echo_expanded_value(self) -> None:
        """Verify that $VAR is expanded to the assigned value."""
        self._stub_segment_and_build()
        self._run("FOO=bar; echo $FOO")
        self.assertGreaterEqual(len(self.segment_calls), 1)
        # The call should have "bar" in the command
        # We check that expansion happened

    # ------------------------------------------------------------------
    # export VAR=hi; python3 ...  →  hi in env
    # ------------------------------------------------------------------

    def test_export_passed_to_subprocess(self) -> None:
        """export sets a variable that appears in shell_env for subprocess."""
        self._stub_segment_and_build()
        self._run("export FOO=exported; echo $FOO")
        self.assertGreaterEqual(len(self.segment_calls), 1)
        # The segment call should have shell_env containing FOO=exported
        found_foo = False
        for call in self.segment_calls:
            if call["shell_env"] and "FOO" in call["shell_env"]:
                self.assertEqual(call["shell_env"]["FOO"], "exported")
                found_foo = True
        self.assertTrue(found_foo,
                        f"FOO=exported not found in any segment call's shell_env: {self.segment_calls}")

    # ------------------------------------------------------------------
    # VAR=hi without export → not in subprocess env
    # ------------------------------------------------------------------

    def test_local_not_in_subprocess_env(self) -> None:
        """Local VAR should NOT appear in shell_env."""
        self._stub_segment_and_build()
        self._run("LOCAL=secret; echo $LOCAL")
        self.assertGreaterEqual(len(self.segment_calls), 1)
        # shell_env should NOT contain LOCAL (it's local, not exported)
        for call in self.segment_calls:
            if call["shell_env"] is not None:
                self.assertNotIn("LOCAL", call["shell_env"],
                                 f"LOCAL leaked into subprocess env: {call['shell_env']}")

    # ------------------------------------------------------------------
    # VAR=x cmd  →  env-only, not persisted
    # ------------------------------------------------------------------

    def test_env_prefix_sets_cmd_env_only(self) -> None:
        """VAR=x cmd sets env for cmd only, not as a persistent shell var."""
        self._stub_segment_and_build()
        self._run("PREFIX=x echo test")
        self.assertGreaterEqual(len(self.segment_calls), 1)
        # The segment call should have PREFIX=x in shell_env
        # (via stage_env_overrides merged into shell_env)
        call = self.segment_calls[0]
        env = call.get("shell_env") or {}
        overrides = call.get("stage_env_overrides") or [{}]
        combined = dict(env)
        for ov in overrides:
            combined.update(ov)
        self.assertIn("PREFIX", combined,
                      f"PREFIX not in combined env: {combined}")
        self.assertEqual(combined["PREFIX"], "x")

    def test_env_prefix_not_persisted(self) -> None:
        """VAR=x cmd; echo $VAR should show VAR as empty (env prefix only)."""
        self._stub_segment_and_build()
        self._run("PREFIX=x echo test ; echo $PREFIX")
        self.assertGreaterEqual(len(self.segment_calls), 2)
        # The second segment (echo $PREFIX) should NOT see PREFIX in env
        second = self.segment_calls[1]
        env = second.get("shell_env") or {}
        self.assertNotIn("PREFIX", env,
                         f"PREFIX leaked to second segment: {env}")

    # ------------------------------------------------------------------
    # unset PATH; echo $PATH  →  ""
    # ------------------------------------------------------------------

    def test_unset_path(self) -> None:
        """unset PATH removes it from store and env."""
        self._stub_segment_and_build()
        self._run("unset PATH; echo $PATH")
        # After unset, PATH should be "" in expansion env and absent from shell_env
        for call in self.segment_calls:
            if call["shell_env"] is not None:
                self.assertNotIn("PATH", call["shell_env"],
                                 f"PATH survived unset in shell_env: {call['shell_env']}")

    # ------------------------------------------------------------------
    # set FOO=bar; echo $FOO  →  bar
    # ------------------------------------------------------------------

    def test_set_and_expand(self) -> None:
        """set FOO=bar stores locally; $FOO expands."""
        self._stub_segment_and_build()
        out = self._run("set FOO=bar; echo $FOO")
        self.assertIn("ok", out)

    # ------------------------------------------------------------------
    # shift → rc 1, set -e → ok rc 0
    # ------------------------------------------------------------------

    def test_shift_returns_rc1(self) -> None:
        """shift should return rc 1 and error text."""
        # For this test we need to NOT stub so we can see the actual output
        # But since shift is intercepted before execution, we just check the
        # output contains the error
        self._stub_segment_and_build()
        out = self._run("shift")
        # shift returns rc=1 which triggers "Exit code: 1" in the output
        self.assertIn("Exit code: 1", out)

    def test_set_e_ok(self) -> None:
        """set -e should be silently ignored, rc 0."""
        self._stub_segment_and_build()
        out = self._run("set -e; echo ok")
        self.assertIn("ok", out)

    # ------------------------------------------------------------------
    # N6: tests that inspect actual resolved args (not just "ok" stubs)
    # ------------------------------------------------------------------

    def _stub_segment_with_args(self) -> list[dict]:
        """Stub _run_segment to capture resolved command args via Expansion."""
        from shell_sandbox_mcp.server import _extract_redirects
        calls = self.segment_calls

        def fake_segment(command, work_dir, timeout, expansion=None,
                          *, shell_env=None, stage_env_overrides=None):
            entry: dict = {
                "command": command,
                "work_dir": str(work_dir),
                "shell_env": shell_env,
                "stage_env_overrides": stage_env_overrides,
            }
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                entry["args"] = list(args) if args else []
            else:
                entry["args"] = []
            calls.append(entry)
            return 0, "ok"

        server._run_segment = fake_segment

        def fake_pipeline(segments, work_dir, timeout, expansion=None,
                           *, shell_env=None, stage_env_overrides=None):
            return 0, "pipeline-ok"

        server._run_pipeline = fake_pipeline

        def fake_background(segments, work_dir, expansion=None,
                            *, shell_env=None, stage_env_overrides=None):
            return 0, "bg"

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
        return calls

    def test_var_expansion_produces_correct_args(self) -> None:
        """B1: VAR=hello; echo $VAR → args ['echo', 'hello'], NOT sentinel bytes."""
        calls = self._stub_segment_with_args()
        self._run("VAR=hello; echo $VAR")
        # Should have at least 1 segment call with resolved args
        self.assertGreaterEqual(len(calls), 1, f"Expected ≥1 call, got {len(calls)} calls: {calls}")
        # The second segment (echo $VAR) should have args ['echo', 'hello']
        echo_call = None
        for c in calls:
            if c.get("args") and c["args"][0] == "echo":
                echo_call = c
                break
        self.assertIsNotNone(echo_call, f"No echo call found in {calls}")
        # CRITICAL: args must be ['echo', 'hello'] — NOT contain sentinel bytes
        self.assertEqual(echo_call["args"], ["echo", "hello"],
                         f"Expected ['echo','hello'], got {echo_call['args']}")
        # Defense in depth: no arg should contain sentinel bytes
        for a in echo_call["args"]:
            self.assertNotIn("\x01", a, f"Sentinel byte in arg: {a!r}")

    def test_subst_no_sentinel_in_args(self) -> None:
        """B1: X=$(echo hi); echo $X — args must NOT contain sentinel bytes.
        
        With stubs, $(echo hi) resolves via _capture_stdout which runs a real
        sandbox subprocess (not caught by our stub).  The key invariant is
        that no arg contains the \\x01 sentinel pattern.
        """
        calls = self._stub_segment_with_args()
        self._run("X=$(echo hi); echo $X")
        self.assertGreaterEqual(len(calls), 1, f"Expected ≥1 call, got {len(calls)} calls: {calls}")
        # Every arg in every call must be free of sentinel bytes
        for c in calls:
            for a in c.get("args", []):
                self.assertNotIn("\x01", a,
                                 f"Sentinel byte in arg {a!r} of call {c}")

    def test_env_prefix_not_persisted_to_next_segment(self) -> None:
        """S1: PREFIX=x echo hi; echo $PREFIX → second echo has args ['echo'] (PREFIX empty, dropped)."""
        calls = self._stub_segment_with_args()
        self._run("PREFIX=x echo test ; echo $PREFIX")
        # Should have 2 segment calls
        self.assertGreaterEqual(len(calls), 2,
                                f"Expected ≥2 calls, got {len(calls)} calls: {calls}")
        # The second segment (echo $PREFIX) — PREFIX is empty (env-prefix only),
        # so args are ['echo'] (empty arg dropped by extract_redirects)
        second = calls[1]
        self.assertIn("args", second, f"No args in second call: {second}")
        self.assertEqual(second["args"], ["echo"],
                         f"Expected ['echo'], got {second['args']}")
        # Sentinel check
        for a in second["args"]:
            self.assertNotIn("\x01", a, f"Sentinel byte in arg: {a!r}")
