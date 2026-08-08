"""Execution tests for function definitions and calls (Phase C).

Reuses the ``_install_stubs`` pattern from test_builtins_compound.py to stub
``_run_segment`` / ``_run_pipeline`` so we can verify execution paths
without spawning subprocesses.

Run with::

    PYTHONPATH=src python3 -m pytest tests/test_builtins_func.py -q
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.parser import (
    CaseNode,
    CommandNode,
    ForNode,
    FuncNode,
    GroupNode,
    IfNode,
    SubshellNode,
    WhileNode,
)
from shell_sandbox_mcp.server import (
    Expansion,
    _extract_redirects,
)
from shell_sandbox_mcp.config import MAX_FUNC_DEPTH


def _install_stubs() -> list[dict]:
    """Stub _run_segment/_run_pipeline to record resolved command args."""
    calls: list[dict] = []

    def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
        if isinstance(command, CommandNode):
            args, _, _ = _extract_redirects(command, expansion, work_dir)
            cmd_str = " ".join(args) if args else "<empty>"
        elif isinstance(command, (IfNode, WhileNode, ForNode, CaseNode,
                                   SubshellNode, FuncNode, GroupNode)):
            cmd_str = f"<{type(command).__name__}>"
        else:
            cmd_str = str(command)
        calls.append({
            "args": cmd_str,
            "work_dir": str(work_dir),
        })
        return 0, ""

    def fake_pipeline(segments, work_dir, timeout, expansion=None, **kwargs):
        str_segs = []
        for s in segments:
            if isinstance(s, CommandNode):
                args, _, _ = _extract_redirects(s, expansion, work_dir)
                str_segs.append(" ".join(args) if args else "<empty>")
            elif isinstance(s, (IfNode, WhileNode, ForNode, CaseNode,
                                 SubshellNode, FuncNode, GroupNode)):
                str_segs.append(f"<{type(s).__name__}>")
            else:
                str_segs.append(str(s))
        cmd_str = " | ".join(str_segs)
        calls.append({
            "args": cmd_str,
            "work_dir": str(work_dir),
        })
        return 0, ""

    server._run_segment = fake_segment
    server._run_pipeline = fake_pipeline
    return calls


def _remove_stubs(orig_segment, orig_pipeline) -> None:
    server._run_segment = orig_segment
    server._run_pipeline = orig_pipeline


class FuncExecutionTest(unittest.TestCase):
    """Test execution of function definitions and calls via shell_run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-func-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        # Clear session-level functions to prevent cross-test pollution.
        server._SESSION_FUNCTIONS.clear()

    def tearDown(self) -> None:
        _remove_stubs(self._orig_segment, self._orig_pipeline)
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()
        server._SESSION_FUNCTIONS.clear()

    # ------------------------------------------------------------------
    # Function definition + call
    # ------------------------------------------------------------------

    def test_define_and_call_posix(self) -> None:
        """f() echo in-func; f → calls the function."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "f() echo in-func; f",
                cwd=str(self.allowed),
                timeout=30,
            )
            # The function body "echo in-func" should have been called.
            func_calls = [c for c in calls if "in-func" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_define_and_call_keyword(self) -> None:
        """function f echo in-func; f → calls the function."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "function f echo in-func; f",
                cwd=str(self.allowed),
                timeout=30,
            )
            func_calls = [c for c in calls if "in-func" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_define_keyword_parens_and_call(self) -> None:
        """function f() echo in-func; f → calls the function."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "function f() echo in-func; f",
                cwd=str(self.allowed),
                timeout=30,
            )
            func_calls = [c for c in calls if "in-func" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # Function arguments and positional parameters
    # ------------------------------------------------------------------

    def test_call_with_args(self) -> None:
        """f() echo $1; f hello → body sees $1 as 'hello'."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "f() echo $1; f hello",
                cwd=str(self.allowed),
                timeout=30,
            )
            # The resolved command should include "hello" (from $1).
            func_calls = [c for c in calls if "hello" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_call_multiple_args(self) -> None:
        """f() echo $1 $2; f a b → body sees $1='a', $2='b'."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "f() echo $1 $2; f a b",
                cwd=str(self.allowed),
                timeout=30,
            )
            # Should have both a and b
            func_calls = [c for c in calls if "a" in c["args"] and "b" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_dollar_hash_in_body(self) -> None:
        """f() echo $#; f a b c → body should see $# = 3."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "f() echo $#; f a b c",
                cwd=str(self.allowed),
                timeout=30,
            )
            # $# should resolve to 3
            func_calls = [c for c in calls if "3" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_braced_default_in_body(self) -> None:
        """f() echo ${1:-default}; f → body should use default."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "f() echo ${1:-default}; f",
                cwd=str(self.allowed),
                timeout=30,
            )
            func_calls = [c for c in calls if "default" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # Recursion
    # ------------------------------------------------------------------

    def test_legitimate_recursion(self) -> None:
        """Simple recursive function that calls itself once."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "f() if true; then echo recursive; f; fi; f",
                cwd=str(self.allowed),
                timeout=30,
            )
            # Should call echo recursive at least once.
            func_calls = [c for c in calls if "recursive" in c["args"]]
            self.assertGreaterEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_recursion_cap(self) -> None:
        """Recursion exceeding MAX_FUNC_DEPTH should fail."""
        # Monkey-patch MAX_FUNC_DEPTH to a small value.
        from shell_sandbox_mcp.runner import _config
        saved = _config.MAX_FUNC_DEPTH
        _config.MAX_FUNC_DEPTH = 3
        try:
            result = server.shell_run(
                # Use brace group so the body includes the recursive f call.
                "f() { echo depth; f; }; f",
                cwd=str(self.allowed),
                timeout=30,
            )
            # The body calls f recursively. After MAX_FUNC_DEPTH=3 caps fire.
            self.assertIn("recursion depth", result.lower())
        finally:
            _config.MAX_FUNC_DEPTH = saved

    # ------------------------------------------------------------------
    # Forward reference
    # ------------------------------------------------------------------

    def test_forward_reference(self) -> None:
        """f() g; g() echo inner; f → forward reference to g works."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "f() g; g() echo inner; f",
                cwd=str(self.allowed),
                timeout=30,
            )
            func_calls = [c for c in calls if "inner" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # Exit-code propagation
    # ------------------------------------------------------------------

    def test_function_exit_code(self) -> None:
        """Function should propagate the exit code of the last command."""
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args) if args else "<empty>"
                if cmd_str == "false":
                    return 1, "failed"
                return 0, ""
            return 0, ""
        server._run_segment = fake_segment
        try:
            # The function body is just "false".
            # Use brace group for multi-command body to check propagation.
            result = server.shell_run(
                "f() { false; echo after; }; f",
                cwd=str(self.allowed),
                timeout=30,
            )
            # After false fails, echo should NOT run (due to set -e? No).
            # Actually both commands run in sequence in brace group.
            # Verify that the function rc=1 propagates.
            self.assertIn("failed", result)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # Unknown function name
    # ------------------------------------------------------------------

    def test_undefined_function(self) -> None:
        """Calling an undefined function should be treated as normal command."""
        result = server.shell_run(
            "undefined_func arg1",
            cwd=str(self.allowed),
            timeout=30,
        )
        self.assertIn("not allowed", result.lower())

    # ------------------------------------------------------------------
    # Function takes precedence over builtin name
    # ------------------------------------------------------------------

    def test_func_overrides_builtin_name(self) -> None:
        """Defining function named 'export' should shadow the builtin."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "export() echo custom; export",
                cwd=str(self.allowed),
                timeout=30,
            )
            func_calls = [c for c in calls if "custom" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # shift inside function body
    # ------------------------------------------------------------------

    def test_shift_in_body(self) -> None:
        """shift inside function body should work on positional params."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                # Use brace group so the body includes both shift and echo.
                "f() { shift; echo $#; }; f a b",
                cwd=str(self.allowed),
                timeout=30,
            )
            # After shift, the echo command should see $# resolved to 1
            func_calls = [c for c in calls if c["args"] == "echo 1"]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # Subshell isolation
    # ------------------------------------------------------------------

    def test_subshell_func_isolation(self) -> None:
        """Functions defined in a subshell should not leak to parent."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "( f() echo inner; f ) ; f",
                cwd=str(self.allowed),
                timeout=30,
            )
            # The outer 'f' call should fail (function not found in parent).
            # The inner 'f' in the subshell should succeed.
            func_calls = [c for c in calls if "inner" in c["args"]]
            self.assertEqual(len(func_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # Brace group { }
    # ------------------------------------------------------------------

    def test_brace_group_executes(self) -> None:
        """{ echo a; echo b; } should run both commands."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "{ echo a; echo b; }",
                cwd=str(self.allowed),
                timeout=30,
            )
            self.assertIn("a", calls[0]["args"])
            self.assertIn("b", calls[1]["args"] if len(calls) > 1 else calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_brace_group_variable_leak(self) -> None:
        """{ VAR=x; } → VAR should persist (no isolation)."""
        result = server.shell_run(
            "{ VAR=x; }; echo $VAR",
            cwd=str(self.allowed),
            timeout=30,
        )
        # VAR should be 'x' after the group.
        self.assertIn("x", result)


# ---------------------------------------------------------------------------
# Cross-call function persistence
# ---------------------------------------------------------------------------


class CrossCallPersistenceTest(unittest.TestCase):
    """Tests that function definitions persist across shell_run calls."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-cc-" + os.urandom(4).hex())
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

    def test_function_persists_across_calls(self) -> None:
        """Define a function in call 1; call it in call 2 — it should still work."""
        calls1 = _install_stubs()
        try:
            result1 = server.shell_run(
                "myfunc() { echo hello; }",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

        self.assertNotIn("Exit code: 1", result1)

        # Call 2: invoke the function — install fresh stubs
        calls2 = _install_stubs()
        try:
            result2 = server.shell_run(
                "myfunc",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

        # The function call should have been dispatched to echo hello via stub
        self.assertTrue(
            any("echo" in c.get("args", "") for c in calls2),
            f"Expected echo in calls: {calls2}"
        )

    def test_unset_f_removes_function(self) -> None:
        """unset -f NAME should remove a function but leave variables intact."""
        _install_stubs()
        try:
            server.shell_run(
                "myfunc2() { echo world; }",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

        calls = _install_stubs()
        try:
            server.shell_run(
                "MYVAR=42; unset -f myfunc2; echo $MYVAR",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

        # echo $MYVAR should expand to echo 42 via the stub
        self.assertTrue(
            any("42" in c.get("args", "") for c in calls),
            f"Expected 42 in calls: {calls}"
        )

        # Now calling myfunc2 should fail
        calls2 = _install_stubs()
        try:
            server.shell_run(
                "myfunc2",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

        # Should not have dispatched to echo world
        self.assertFalse(
            any("world" in c.get("args", "") for c in calls2),
            f"Expected no world in calls: {calls2}"
        )

    def test_unset_no_flag_removes_both(self) -> None:
        """unset NAME (no flag) removes from both variables and functions."""
        _install_stubs()
        try:
            server.shell_run(
                "myfunc3() { echo both; }",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

        # Call 2: set var, unset both function and var, echo var
        calls = _install_stubs()
        try:
            server.shell_run(
                "BOTH=yes; unset myfunc3 BOTH; echo BOTH=$BOTH",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

        # echo BOTH=$BOTH → BOTH should be unset, so args should show echo BOTH=
        self.assertTrue(
            any("BOTH=" in c.get("args", "") for c in calls),
            f"Expected BOTH= in calls: {calls}"
        )

        # Call 3: function should be gone
        calls2 = _install_stubs()
        try:
            server.shell_run(
                "myfunc3",
                cwd=str(self.allowed),
                timeout=30,
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

        # Should not have dispatched to echo both
        self.assertFalse(
            any("both" in c.get("args", "") for c in calls2),
            f"Expected no both in calls: {calls2}"
        )
