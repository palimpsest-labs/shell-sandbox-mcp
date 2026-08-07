"""Tests for the opt-in ``structured=True`` result shape of ``shell_run``.

Run with the venv python that has `mcp` installed:

    PYTHONPATH=src <venv>/bin/python -m unittest discover -s tests -v

Covers the default (unchanged) string return plus the structured dict
(rc / skipped / stages / output) for: a simple command, a failing command,
an ``&&`` short-circuit skip, a pipeline, and the ``cd`` builtin.  Most
tests stub ``_run_segment`` / ``_run_pipeline`` for deterministic rc/output;
a few exercise the real sandbox subprocess path.
"""

import os
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.server import CommandNode, _serialize_command


class ShellRunStructuredTest(unittest.TestCase):
    """Exercise the ``structured=True`` dict return of ``shell_run``."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-structured-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = server._run_background

    def tearDown(self) -> None:
        import shutil

        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        server._run_background = self._orig_background
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _stub_segments(self, rc_map: dict[str, int]) -> None:
        """Stub _run_segment: rc from rc_map, output ``out:``/``err:`` per cmd."""
        def fake(command, work_dir: Path, timeout: int,
                 expansion=None) -> tuple[int, str]:
            c = (
                _serialize_command(command)
                if isinstance(command, CommandNode)
                else str(command)
            )
            rc = rc_map.get(c, 0)
            return rc, f"out:{c}" if rc == 0 else f"err:{c}"

        server._run_segment = fake

    def _stub_pipeline(self, rc: int, out: str) -> None:
        def fake(segments, work_dir: Path, timeout: int,
                 expansion=None) -> tuple[int, str]:
            return rc, out

        server._run_pipeline = fake

    # ------------------------------------------------------------------
    # default return is unchanged
    # ------------------------------------------------------------------

    def test_default_return_is_string_identical(self) -> None:
        self._stub_segments({})
        res = server.shell_run("echo hi", cwd=str(self.allowed))
        self.assertIsInstance(res, str)
        self.assertEqual(res, "out:echo hi")

    def test_default_return_skipped_chain_identical(self) -> None:
        self._stub_segments({"false": 1})
        res = server.shell_run("false && echo hi", cwd=str(self.allowed))
        self.assertEqual(res, "err:false\n(skipped: previous command exited 1) — echo hi")

    # ------------------------------------------------------------------
    # structured: simple command
    # ------------------------------------------------------------------

    def test_structured_simple_echo(self) -> None:
        self._stub_segments({})
        res = server.shell_run("echo hi", cwd=str(self.allowed), structured=True)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["rc"], 0)
        self.assertIs(res["skipped"], False)
        self.assertEqual(res["output"], "out:echo hi")
        self.assertEqual(len(res["stages"]), 1)
        st = res["stages"][0]
        self.assertEqual(st["command"], "echo hi")
        self.assertEqual(st["output"], "out:echo hi")
        self.assertEqual(st["rc"], 0)

    # ------------------------------------------------------------------
    # structured: failing command
    # ------------------------------------------------------------------

    def test_structured_failing_command(self) -> None:
        self._stub_segments({"false": 1})
        res = server.shell_run("false", cwd=str(self.allowed), structured=True)
        self.assertEqual(res["rc"], 1)
        self.assertIs(res["skipped"], False)
        self.assertEqual(res["output"], "err:false")
        self.assertEqual(res["stages"][0]["rc"], 1)
        self.assertEqual(res["stages"][0]["output"], "err:false")

    # ------------------------------------------------------------------
    # structured: && short-circuit skip
    # ------------------------------------------------------------------

    def test_structured_and_short_circuit(self) -> None:
        self._stub_segments({"false": 1})
        res = server.shell_run("false && echo hi", cwd=str(self.allowed), structured=True)
        self.assertEqual(res["rc"], 1)
        self.assertIs(res["skipped"], True)
        self.assertEqual(len(res["stages"]), 2)
        # first stage actually ran
        self.assertEqual(res["stages"][0]["command"], "false")
        self.assertEqual(res["stages"][0]["rc"], 1)
        # second stage was short-circuited -> rc None
        self.assertEqual(res["stages"][1]["command"], "echo hi")
        self.assertEqual(res["stages"][1]["rc"], None)
        self.assertIn("skipped: previous command exited 1", res["output"])

    # ------------------------------------------------------------------
    # structured: pipeline
    # ------------------------------------------------------------------

    def test_structured_pipeline(self) -> None:
        self._stub_pipeline(3, "pipeline-out")
        res = server.shell_run("echo a | wc -c", cwd=str(self.allowed), structured=True)
        self.assertEqual(res["rc"], 3)
        self.assertEqual(res["output"], "pipeline-out")
        self.assertEqual(len(res["stages"]), 1)
        st = res["stages"][0]
        self.assertEqual(st["command"], "echo a | wc -c")
        self.assertEqual(st["output"], "pipeline-out")
        self.assertEqual(st["rc"], 3)

    # ------------------------------------------------------------------
    # structured: cd builtin
    # ------------------------------------------------------------------

    def test_structured_cd_builtin(self) -> None:
        self._stub_segments({})
        res = server.shell_run("cd sub", cwd=str(self.allowed), structured=True)
        self.assertEqual(res["rc"], 0)
        self.assertIs(res["skipped"], False)
        self.assertEqual(len(res["stages"]), 1)
        st = res["stages"][0]
        self.assertEqual(st["command"], "cd sub")
        self.assertEqual(st["rc"], 0)
        self.assertEqual(st["output"], "")
        # no output from a lone cd -> "(no output)" preserved
        self.assertEqual(res["output"], "(no output)")

    # ------------------------------------------------------------------
    # structured: error / early-return paths return a dict
    # ------------------------------------------------------------------

    def test_structured_invalid_cwd_returns_dict(self) -> None:
        res = server.shell_run("echo hi", cwd="/no/such/dir", structured=True)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["rc"], 1)
        self.assertIs(res["skipped"], False)
        self.assertEqual(res["stages"], [])
        self.assertIn("Directory not found", res["output"])

    def test_structured_expansion_error_returns_dict(self) -> None:
        # ${VAR:?msg} on an unset var raises during expansion -> structured dict
        res = server.shell_run(
            "echo ${UNSET_VAR:?boom}", cwd=str(self.allowed), structured=True
        )
        self.assertIsInstance(res, dict)
        self.assertEqual(res["rc"], 1)
        self.assertIs(res["skipped"], False)
        self.assertEqual(res["stages"], [])
        self.assertIn("boom", res["output"])

    def test_structured_empty_command_returns_dict(self) -> None:
        res = server.shell_run("", cwd=str(self.allowed), structured=True)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["rc"], 1)
        self.assertIs(res["skipped"], False)
        self.assertEqual(res["stages"], [])
        self.assertIn("Empty command", res["output"])


if __name__ == "__main__":
    unittest.main()
