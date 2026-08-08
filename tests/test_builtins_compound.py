"""Execution tests for AST-native compound commands (if/while/until/for).

Reuses the ``_install_stubs`` pattern from test_builtins_for.py to stub
``_run_segment`` / ``_run_pipeline`` so we can verify execution paths
without spawning subprocesses.

Run with::

    PYTHONPATH=src python3 -m pytest tests/test_builtins_compound.py -q
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.server import (
    CommandNode,
    Expansion,
    _extract_redirects,
)


# Reuse the stub helpers from test_builtins_for.py — ported inline.
def _install_stubs() -> list[dict]:
    """Stub _run_segment/_run_pipeline to record resolved command args."""
    calls: list[dict] = []

    def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
        if isinstance(command, CommandNode):
            args, _, _ = _extract_redirects(command, expansion, work_dir)
            cmd_str = " ".join(args) if args else "<empty>"
        else:
            cmd_str = command
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


class CompoundExecutionTest(unittest.TestCase):
    """Test execution of if/while/until/for via shell_run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-cmp-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline

    def tearDown(self) -> None:
        _remove_stubs(self._orig_segment, self._orig_pipeline)
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # if / else
    # ------------------------------------------------------------------

    def test_if_true_runs_body(self) -> None:
        calls = _install_stubs()
        # Use custom stubs where anything starting with 'true' returns rc=0,
        # and 'false' returns rc=1.
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                if cmd_str == "false":
                    return 1, ""
                calls.append({
                    "args": cmd_str,
                    "work_dir": str(work_dir),
                })
                return 0, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "if true; then echo yes; fi",
                cwd=str(self.allowed),
                timeout=30,
            )
            self.assertIn("yes", calls[-1]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_if_false_skips_body(self) -> None:
        calls = _install_stubs()
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                if cmd_str == "false":
                    calls.append({"args": cmd_str, "work_dir": str(work_dir)})
                    return 1, ""
                calls.append({"args": cmd_str, "work_dir": str(work_dir)})
                return 0, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "if false; then echo yes; fi",
                cwd=str(self.allowed),
                timeout=30,
            )
            # Only false runs (the condition), no body.
            self.assertEqual(len(calls), 1)
            self.assertIn("false", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_if_false_else_runs(self) -> None:
        calls = _install_stubs()
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                if cmd_str == "false":
                    calls.append({"args": cmd_str, "work_dir": str(work_dir)})
                    return 1, ""
                calls.append({"args": cmd_str, "work_dir": str(work_dir)})
                return 0, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "if false; then echo yes; else echo no; fi",
                cwd=str(self.allowed),
                timeout=30,
            )
            self.assertEqual(len(calls), 2)  # false + echo no
            self.assertIn("no", calls[1]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_if_elif(self) -> None:
        calls = _install_stubs()
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                if cmd_str == "false":
                    calls.append({"args": cmd_str, "work_dir": str(work_dir)})
                    return 1, ""
                calls.append({"args": cmd_str, "work_dir": str(work_dir)})
                return 0, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "if false; then echo a; elif true; then echo b; fi",
                cwd=str(self.allowed),
                timeout=30,
            )
            # false (rc=1) → skip a, true (rc=0) → run b
            self.assertIn("b", calls[-1]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_if_condition_with_and_and(self) -> None:
        """Condition with && — both commands run."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "if true && echo cond; then echo body; fi",
                cwd=str(self.allowed),
                timeout=30,
            )
            # true && echo cond → then echo body
            self.assertIn("body", calls[-1]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # while / until
    # ------------------------------------------------------------------

    def test_while_false_never_enters(self) -> None:
        calls = _install_stubs()
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                if cmd_str == "false":
                    calls.append({"args": cmd_str, "work_dir": str(work_dir)})
                    return 1, ""
                calls.append({"args": cmd_str, "work_dir": str(work_dir)})
                return 0, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "while false; do echo loop; done",
                cwd=str(self.allowed),
                timeout=30,
            )
            # Only the condition false runs.
            self.assertEqual(len(calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_while_once(self) -> None:
        """while true runs once because the stub returns rc=0, then we
        need the condition to fail to exit.  We'll run true once and then
        condition must change — but with stubs all return 0.  So we
        test the MAX_LOOP_ITER cap instead."""
        calls = _install_stubs()
        # Make segment fail after first iteration
        iteration = [0]

        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            iteration[0] += 1
            # First call is the condition (true → rc=0),
            # second call is body, third call is condition again.
            # Make third fail.
            if iteration[0] == 3:
                return 1, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "while true; do echo loop; done",
                cwd=str(self.allowed),
                timeout=30,
            )
            # condition passes (rc=0), body runs, condition fails (rc=1)
            self.assertEqual(iteration[0], 3)  # cond1, body1, cond2
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_until_false_enters(self) -> None:
        """until false → enters body because false rc=1."""
        calls = _install_stubs()
        iteration = [0]

        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            iteration[0] += 1
            # First: condition (false → rc=1 → enter body)
            # Second: body (echo loop)
            # Third: condition again (make it true → rc=0 → exit)
            if iteration[0] == 3:
                return 0, ""
            if iteration[0] == 1:
                return 1, ""  # false
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "until false; do echo loop; done",
                cwd=str(self.allowed),
                timeout=30,
            )
            # cond (rc=1) → body → cond (rc=0) → exit
            self.assertEqual(iteration[0], 3)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # for (AST-native)
    # ------------------------------------------------------------------

    def test_for_basic(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in a b c; do echo $i; done",
                cwd=str(self.allowed),
                timeout=30,
            )
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0]["args"], "echo a")
            self.assertEqual(calls[1]["args"], "echo b")
            self.assertEqual(calls[2]["args"], "echo c")
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_for_no_in_clause(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i; do echo $i; done",
                cwd=str(self.allowed),
                timeout=30,
            )
            self.assertEqual(len(calls), 0)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_for_cd_inside_body_does_not_persist(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in a b; do cd sub && echo $i; done",
                cwd=str(self.allowed),
                timeout=30,
            )
            self.assertEqual(len(calls), 2)
            # Both calls should have work_dir ending in /sub
            for call in calls:
                self.assertIn("sub", call["work_dir"])
                self.assertIn("echo", call["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_for_invalid_var_name(self) -> None:
        result = server.shell_run(
            "for 1x in a; do echo hi; done",
            cwd=str(self.allowed),
            timeout=30,
        )
        self.assertIn("invalid variable name", str(result))

    # ------------------------------------------------------------------
    # variable state persistence
    # ------------------------------------------------------------------

    def test_variable_set_in_if_body_persists(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "x=1; if true; then x=2; fi; echo $x",
                cwd=str(self.allowed),
                timeout=30,
            )
            # After if body, x should be 2.
            self.assertIn("echo 2", calls[-1]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_cd_inside_if_body_does_not_leak(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "cd sub; if true; then cd ..; fi; echo hi",
                cwd=str(self.allowed),
                timeout=30,
            )
            # After if body cd .., we should be back in sub's parent.
            # echo hi should run in .../sub (cd sub persisted).
            echo_call = calls[-1]
            self.assertIn("sub", echo_call["work_dir"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_for_containing_if(self) -> None:
        """For-loop body containing an if statement."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "for i in 1 2; do if true; then echo $i; fi; done",
                cwd=str(self.allowed),
                timeout=30,
            )
            # Two iterations, each with one echo.
            self.assertGreaterEqual(len(calls), 2)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_exit_code_propagation_if(self) -> None:
        """Exit code reflects last command in executed branch."""
        calls = _install_stubs()
        # Make the body command fail
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                if "echo body" in cmd_str:
                    return 3, "body output"
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "if true; then echo body; fi",
                cwd=str(self.allowed),
                timeout=30,
                structured=True,
            )
            self.assertEqual(result["rc"], 3)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_empty_condition_returns_zero(self) -> None:
        """Empty condition or body returns rc=0."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "if true; then :; fi",
                cwd=str(self.allowed),
                timeout=30,
                structured=True,
            )
            self.assertEqual(result["rc"], 0)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # Short-circuit across chains after compound
    # ------------------------------------------------------------------

    def test_and_and_skip_after_compound_failure(self) -> None:
        """if true; then false; fi && echo NEXT — && chain skips when compound rc != 0."""
        calls = _install_stubs()
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                if cmd_str == "false":
                    calls.append({"args": cmd_str})
                    return 1, ""
                calls.append({"args": cmd_str})
                return 0, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "if true; then false; fi && echo NEXT",
                cwd=str(self.allowed),
                timeout=30,
                structured=True,
            )
            # false runs (rc=1), echo NEXT should be skipped
            cmd_args = [c["args"] for c in calls]
            self.assertIn("false", cmd_args)
            self.assertNotIn("echo NEXT", cmd_args)
            self.assertTrue(result["skipped"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_or_or_skip_after_compound_success(self) -> None:
        """if true; then true; fi || echo NO — || chain skips when compound rc == 0."""
        calls = _install_stubs()
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                calls.append({"args": cmd_str})
                return 0, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "if true; then true; fi || echo NO",
                cwd=str(self.allowed),
                timeout=30,
                structured=True,
            )
            # true runs (rc=0), echo NO should be skipped
            cmd_args = [c["args"] for c in calls]
            self.assertIn("true", cmd_args)
            self.assertNotIn("echo NO", cmd_args)
            self.assertTrue(result["skipped"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_and_and_runs_after_compound_success(self) -> None:
        """if true; then true; fi && echo NEXT — && chain runs when compound rc == 0."""
        calls = _install_stubs()
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                calls.append({"args": cmd_str})
                return 0, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "if true; then true; fi && echo NEXT",
                cwd=str(self.allowed),
                timeout=30,
                structured=True,
            )
            cmd_args = [c["args"] for c in calls]
            self.assertIn("true", cmd_args)
            self.assertIn("echo NEXT", cmd_args)
            self.assertFalse(result["skipped"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_heredoc_inside_if_body_runs(self) -> None:
        """if true; then cat <<EOF\nhello\nEOF\nfi — heredoc inside compound body."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "if true; then cat <<EOF\nhello\nEOF\nfi",
                cwd=str(self.allowed),
                timeout=30,
            )
            # Condition (true) runs first, then body (cat with heredoc).
            self.assertGreaterEqual(len(calls), 2)
            body_calls = [c for c in calls if "cat" in c["args"]]
            self.assertEqual(len(body_calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)


# ------------------------------------------------------------------
# case / esac execution tests
# ------------------------------------------------------------------


class CaseExecutionTest(unittest.TestCase):
    """Test execution of case/esac via shell_run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-case-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline

    def tearDown(self) -> None:
        _remove_stubs(self._orig_segment, self._orig_pipeline)
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def test_case_exact_match(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "case x in x) echo matched;; esac",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("matched", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_glob_star(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "case hello in *) echo default;; esac",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("default", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_no_match(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "case x in a) echo a;; b) echo b;; esac",
                cwd=str(self.allowed), timeout=30,
                structured=True,
            )
            self.assertEqual(len(calls), 0)
            self.assertEqual(result["rc"], 0)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_pipe_alternation_match(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "case b in a|b) echo ab;; esac",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("ab", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_default_clause(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "case x in a) echo a;; *) echo default;; esac",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("default", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_quoted_star_literal(self) -> None:
        """Quoted * in pattern is literal, not glob."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                'case "*" in "*") echo literal;; *) echo glob;; esac',
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("literal", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_var_subject(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "x=hello; case $x in hello) echo matched;; esac",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertIn("matched", calls[-1]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_var_pattern(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "p=hello; case hello in $p) echo matched;; esac",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("matched", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_exit_code_propagation(self) -> None:
        """Exit code from matched clause propagates."""
        calls = _install_stubs()
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                if "echo fail" in cmd_str:
                    return 3, "fail"
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "case x in x) echo fail;; esac",
                cwd=str(self.allowed), timeout=30, structured=True,
            )
            self.assertEqual(result["rc"], 3)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_nested_if_in_body(self) -> None:
        """Case clause body can contain an if statement."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "case x in x) if true; then echo inner; fi;; esac",
                cwd=str(self.allowed), timeout=30,
            )
            # One call for 'true' (condition), one for 'echo inner' (body)
            self.assertGreaterEqual(len(calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_case_chained_with_and_and(self) -> None:
        """&& chain after case works."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "case x in x) echo ok;; esac && echo NEXT",
                cwd=str(self.allowed), timeout=30,
            )
            # echo ok runs (rc=0), then echo NEXT runs
            self.assertGreaterEqual(len(calls), 2)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)


# ------------------------------------------------------------------
# subshell execution tests
# ------------------------------------------------------------------


class SubshellExecutionTest(unittest.TestCase):
    """Test execution of subshells via shell_run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-sub-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline

    def tearDown(self) -> None:
        _remove_stubs(self._orig_segment, self._orig_pipeline)
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def test_subshell_basic(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "(echo hi)",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("hi", calls[0]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_subshell_var_isolation(self) -> None:
        """Variable set inside subshell does NOT leak to parent."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "(x=2); echo $x",
                cwd=str(self.allowed), timeout=30,
            )
            # echo $x should expand to empty string (x not set in parent)
            self.assertIn("echo", calls[-1]["args"])
            self.assertNotIn("2", calls[-1]["args"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_subshell_exit_code(self) -> None:
        """Exit code from subshell propagates."""
        calls = _install_stubs()
        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args)
                if "false" in cmd_str:
                    return 1, ""
            return 0, ""

        server._run_segment = fake_segment
        try:
            result = server.shell_run(
                "(false)",
                cwd=str(self.allowed), timeout=30, structured=True,
            )
            self.assertEqual(result["rc"], 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_subshell_cd_isolation(self) -> None:
        """cd inside subshell does NOT leak."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "cd sub; (cd ..); echo hi",
                cwd=str(self.allowed), timeout=30,
            )
            # echo hi should run in .../sub (cd sub persisted)
            echo_call = calls[-1]
            self.assertIn("sub", echo_call["work_dir"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_subshell_with_if(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "( if true; then echo inner; fi )",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertGreaterEqual(len(calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_nested_subshell(self) -> None:
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "( ( echo inner ) )",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertGreaterEqual(len(calls), 1)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_subshell_chained_and_and(self) -> None:
        """&& after subshell works."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "(echo hi) && echo NEXT",
                cwd=str(self.allowed), timeout=30,
            )
            self.assertGreaterEqual(len(calls), 2)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_subshell_chained_or_or_skip(self) -> None:
        """|| after successful subshell skips."""
        calls = _install_stubs()
        try:
            result = server.shell_run(
                "(echo hi) || echo NO",
                cwd=str(self.allowed), timeout=30, structured=True,
            )
            self.assertTrue(result["skipped"])
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
