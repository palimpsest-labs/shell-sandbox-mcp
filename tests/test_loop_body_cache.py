"""Tests for the parse-level command cache (``parse_command`` LRU cache).

Verifies that caching the value-agnostic parse artifacts (program AST,
serialized command, and sentinel registry) does NOT freeze per-iteration
values: ``$( ... )`` output is re-captured and ``$VAR`` is re-resolved
from the current env on every call.

The test suite here (like test_builtins_for.py / test_builtins_compound.py)
stubs ``_run_segment`` / ``_run_pipeline`` so execution paths can be verified
without spawning real subprocesses.  ``_capture_stdout`` is stubbed to emit a
fresh, distinct value per call, so the ``$( ... )`` re-capture behaviour is
observable without a real clock or subprocess.

Run with::

    python3 -m pytest tests/test_loop_body_cache.py -q
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.parser import _clear_parse_cache, _populate_expansion
from shell_sandbox_mcp.server import CommandNode, _extract_redirects


def _install_stubs() -> list[dict]:
    """Stub _run_segment/_run_pipeline to record resolved command args."""
    calls: list[dict] = []

    def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
        if isinstance(command, CommandNode):
            args, _, _ = _extract_redirects(command, expansion, work_dir)
            cmd_str = " ".join(args) if args else "<empty>"
        else:
            cmd_str = command
        calls.append({"args": cmd_str})
        return 0, ""

    def fake_pipeline(segments, work_dir, timeout, expansion=None, **kwargs):
        str_segs = []
        for s in segments:
            if isinstance(s, CommandNode):
                args, _, _ = _extract_redirects(s, expansion, work_dir)
                str_segs.append(" ".join(args) if args else "<empty>")
            else:
                str_segs.append(str(s))
        calls.append({"args": " | ".join(str_segs)})
        return 0, ""

    server._run_segment = fake_segment
    server._run_pipeline = fake_pipeline
    return calls


def _remove_stubs(orig_segment, orig_pipeline) -> None:
    server._run_segment = orig_segment
    server._run_pipeline = orig_pipeline


class LoopBodyCacheTest(unittest.TestCase):
    """Exercise the parse-level cache via stubbed execution."""

    def setUp(self) -> None:
        _clear_parse_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-lbc-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_capture = server._capture_stdout
        self._captured_substs: list[str] = []

    def tearDown(self) -> None:
        _remove_stubs(self._orig_segment, self._orig_pipeline)
        server._capture_stdout = self._orig_capture
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def _install_subst_counter(self) -> None:
        """Stub _capture_stdout to emit a fresh distinct value per $(...) call."""
        def fake_capture(command, work_dir, timeout, depth,
                         deadline=None, subst_count=None, env=None):
            self._captured_substs.append(command)
            return 0, f"ts-{len(self._captured_substs)}".encode("utf-8")
        server._capture_stdout = fake_capture

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.allowed), timeout=30)

    # ------------------------------------------------------------------
    # value re-resolution per iteration (parse-level cache)
    # ------------------------------------------------------------------

    def test_command_substitution_recaptured_each_iteration(self) -> None:
        """$(...) must be re-captured on EVERY iteration, not frozen."""
        calls = _install_stubs()
        self._install_subst_counter()
        try:
            self._run("for x in 1 2 3; do echo $(date +%s); done")
            args = [c["args"] for c in calls]
            self.assertEqual(len(args), 3)
            # Fresh output per iteration — no frozen value.
            self.assertEqual(len(set(args)), 3)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)
            server._capture_stdout = self._orig_capture

    def test_loop_var_changes_per_iteration(self) -> None:
        """$x is re-read from the current env each iteration."""
        calls = _install_stubs()
        try:
            self._run("for x in a b c; do echo $x; done")
            self.assertEqual(
                [c["args"] for c in calls],
                ["echo a", "echo b", "echo c"],
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_ifs_change_affects_next_iteration(self) -> None:
        """An IFS change inside the body is seen by the next (cached) iteration."""
        calls = _install_stubs()
        try:
            # Iteration 1 sets IFS=: so $x ("a:b") splits to `echo a b`.
            # Iteration 2 (cache hit) must still see IFS=: and split "c:d".
            self._run('for x in "a:b" "c:d"; do IFS=:; echo $x; done')
            self.assertEqual(
                [c["args"] for c in calls],
                ["echo a b", "echo c d"],
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_nested_loops(self) -> None:
        """Nested loops share the parse cache and still resolve fresh values."""
        calls = _install_stubs()
        try:
            self._run("for i in 1 2; do for j in a b; do echo $i$j; done; done")
            self.assertEqual(
                [c["args"] for c in calls],
                ["echo 1a", "echo 1b", "echo 2a", "echo 2b"],
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # control flow through the cached body
    # ------------------------------------------------------------------

    @staticmethod
    def _install_test_stubs() -> list[dict]:
        """Like _install_stubs but makes `test A = B` return rc by equality."""
        calls: list[dict] = []

        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                cmd_str = " ".join(args) if args else "<empty>"
            else:
                cmd_str = command
            calls.append({"args": cmd_str})
            if args and args[0] == "test":
                # test L = R → rc 0 iff L == R
                return (0, "") if args[1] == args[3] else (1, "")
            return 0, ""

        def fake_pipeline(segments, work_dir, timeout, expansion=None, **kwargs):
            calls.append({"args": " | ".join(str(s) for s in segments)})
            return 0, ""

        server._run_segment = fake_segment
        server._run_pipeline = fake_pipeline
        return calls

    def test_break_still_works(self) -> None:
        """break inside a cached body terminates the loop."""
        calls = self._install_test_stubs()
        try:
            self._run('for x in 1 2 3; do if test "$x" = 2; then break; fi; echo $x; done')
            self.assertEqual(
                [c["args"] for c in calls],
                ["test 1 = 2", "echo 1", "test 2 = 2"],
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_continue_still_works(self) -> None:
        """continue inside a cached body skips the current iteration."""
        calls = self._install_test_stubs()
        try:
            self._run('for x in 1 2 3; do if test "$x" = 2; then continue; fi; echo $x; done')
            self.assertEqual(
                [c["args"] for c in calls],
                ["test 1 = 2", "echo 1", "test 2 = 2", "test 3 = 2", "echo 3"],
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    # ------------------------------------------------------------------
    # edge cases
    # ------------------------------------------------------------------

    def test_empty_body(self) -> None:
        """An empty loop body is a no-op across iterations."""
        calls = _install_stubs()
        try:
            self._run("for x in 1 2 3; do done")
            self.assertEqual(len(calls), 0)
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_parse_error_in_body_returns_rc1(self) -> None:
        """A parse error in the body surfaces as rc=1 with a message (first iter)."""
        res = server.shell_run(
            'for x in 1 2; do echo "unclosed; done',
            cwd=str(self.allowed), timeout=30, structured=True,
        )
        self.assertEqual(res["rc"], 1)
        self.assertTrue(res["output"], "expected a non-empty error message")

    # ------------------------------------------------------------------
    # new parse-level cache tests
    # ------------------------------------------------------------------

    def test_clear_parse_cache_exists_and_works(self) -> None:
        """_clear_parse_cache is importable and clears the cache."""
        from shell_sandbox_mcp.parser import _PARSE_CACHE
        _PARSE_CACHE["test-key"] = ("fake",)  # type: ignore[arg-type]
        self.assertIn("test-key", _PARSE_CACHE)
        _clear_parse_cache()
        self.assertEqual(len(_PARSE_CACHE), 0)

    def test_ast_immutability_guard(self) -> None:
        """Running the same parsed command twice through run_command works,
        proving cached AST nodes are not mutated downstream."""
        calls = _install_stubs()
        try:
            # First run should produce the expected output.
            self._run("for x in 1 2; do echo hello; done")
            self.assertEqual(
                [c["args"] for c in calls],
                ["echo hello", "echo hello"],
            )
            calls.clear()
            # Second run with the same command — cache hit.
            self._run("for x in 1 2; do echo hello; done")
            self.assertEqual(
                [c["args"] for c in calls],
                ["echo hello", "echo hello"],
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)

    def test_identical_commands_different_env_produce_different_output(self) -> None:
        """Two calls to run_command with the same body text but different
        env values must produce different outputs — proves env is per-call,
        not frozen by the cache."""
        calls = _install_stubs()
        try:
            # First: set X=first before the loop
            self._run("X=first; for x in 1 1; do echo $X$x; done")
            self.assertEqual(
                [c["args"] for c in calls],
                ["echo first1", "echo first1"],
            )
            calls.clear()
            # Second: set X=second before the loop (same body text as before)
            self._run("X=second; for x in 1 1; do echo $X$x; done")
            self.assertEqual(
                [c["args"] for c in calls],
                ["echo second1", "echo second1"],
            )
        finally:
            _remove_stubs(self._orig_segment, self._orig_pipeline)


class PopulateExpansionOperatorCacheTest(unittest.TestCase):
    """Direct regression tests for the cache-hit re-resolution of braced
    parameter-expansion operator forms.

    ``_populate_expansion`` is the cache-hit counterpart to ``_build_ast``'s
    populate mode: it re-resolves each sentinel from the current env on
    calls after the first.  A past regression made its VARREF branch resolve
    EVERY sentinel via a plain ``env.get(source, "")``, so braced operator
    forms (``${Y:-default}``, ``${#Y}``, etc.) resolved to ``""`` on cache
    hits instead of their correct value.  These tests pin the mirrored
    branching that fixes it.
    """

    def _populate(self, source: str, env: dict) -> str:
        registry = [("arg", "\x01A0\x01", source, False)]  # VARREF, is_subst=False
        exp = _populate_expansion(registry, capture_fn=lambda s: (0, b""), env=env)
        return exp._arg_values["\x01A0\x01"]

    def test_default_operator_unset_var(self) -> None:
        """${Y:-fallback} with Y unset → fallback (was: '' on cache hits)."""
        self.assertEqual(self._populate("Y:-fallback", {}), "fallback")

    def test_default_operator_set_var(self) -> None:
        """${Y:-d} with Y set → Y's value (not the default)."""
        self.assertEqual(self._populate("Y:-d", {"Y": "real"}), "real")

    def test_length_operator(self) -> None:
        """${#Y} → string length, not ''."""
        self.assertEqual(self._populate("#Y", {"Y": "abcd"}), "4")

    def test_positional_plain(self) -> None:
        """$1 (plain positional) still resolves straight from env."""
        self.assertEqual(self._populate("1", {"1": "posarg"}), "posarg")

    def test_operator_form_with_dollar_in_body_loop(self) -> None:
        """End-to-end: ${Y:-fallback} is identical on both iterations of a
        loop whose body is parsed fresh on iteration 1 and re-resolved via
        the parse cache on iteration 2."""
        tmp = tempfile.TemporaryDirectory()
        allowed = Path(tempfile.gettempdir()) / ("sandbox-lbc-op-" + os.urandom(4).hex())
        allowed.mkdir()
        orig_segment, orig_pipeline = server._run_segment, server._run_pipeline
        orig_capture = server._capture_stdout
        _clear_parse_cache()
        calls = _install_stubs()
        try:
            server.shell_run("for x in 1 2; do echo ${Y:-fallback}; done",
                             cwd=str(allowed), timeout=30)
            self.assertEqual(
                [c["args"] for c in calls],
                ["echo fallback", "echo fallback"],
            )
        finally:
            _remove_stubs(orig_segment, orig_pipeline)
            server._capture_stdout = orig_capture
            shutil.rmtree(allowed, ignore_errors=True)
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
