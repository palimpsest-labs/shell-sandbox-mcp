"""Tests for shell_sandbox_mcp.server security-critical helpers.

These cover path-containment, local-binary resolution, and cwd validation —
the code paths that decide what commands an agent may run. Run with the
venv python that has `mcp` installed:

    PYTHONPATH=src <venv>/bin/python -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exec(path: Path) -> None:
    path.write_text("#!/bin/sh\necho hi\n")
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# _contained_path / _resolve_local_binary
# ---------------------------------------------------------------------------


class LocalBinaryResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "build").mkdir()
        _make_exec(self.root / "hello")
        _make_exec(self.root / "build" / "tool")
        _make_exec(self.root / "plain_name")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_bare_name_resolves(self) -> None:
        self.assertEqual(
            server._resolve_local_binary("hello", self.root),
            str((self.root / "hello").resolve()),
        )

    def test_relative_path_resolves(self) -> None:
        self.assertEqual(
            server._resolve_local_binary("./hello", self.root),
            str((self.root / "hello").resolve()),
        )
        self.assertEqual(
            server._resolve_local_binary("build/tool", self.root),
            str((self.root / "build" / "tool").resolve()),
        )

    def test_absolute_path_inside_cwd(self) -> None:
        self.assertEqual(
            server._resolve_local_binary(str(self.root / "hello"), self.root),
            str((self.root / "hello").resolve()),
        )

    def test_dotdot_escape_rejected(self) -> None:
        self.assertIsNone(server._resolve_local_binary("../escape", self.root))
        # craft a path that resolves above root
        self.assertIsNone(server._resolve_local_binary("../../etc/passwd", self.root))

    def test_absolute_path_outside_cwd_rejected(self) -> None:
        self.assertIsNone(server._resolve_local_binary("/etc/hostname", self.root))

    def test_dot_and_dotdot_rejected(self) -> None:
        self.assertIsNone(server._resolve_local_binary(".", self.root))
        self.assertIsNone(server._resolve_local_binary("..", self.root))

    def test_nonexistent_rejected(self) -> None:
        self.assertIsNone(server._resolve_local_binary("./nope", self.root))

    def test_non_executable_rejected(self) -> None:
        (self.root / "notexec").write_text("#!/bin/sh\n")
        (self.root / "notexec").chmod(0o644)
        self.assertIsNone(server._resolve_local_binary("./notexec", self.root))

    def test_symlink_escaping_cwd_rejected(self) -> None:
        # symlink inside root -> /etc; resolve() follows it, containment fails
        (self.root / "evil").symlink_to("/etc")
        self.assertIsNone(server._resolve_local_binary("evil/hostname", self.root))

    def test_symlink_within_cwd_allowed(self) -> None:
        target = self.root / "build" / "tool"
        link = self.root / "mylink"
        link.symlink_to(target)
        self.assertEqual(
            server._resolve_local_binary("mylink", self.root),
            str(target.resolve()),
        )


# ---------------------------------------------------------------------------
# _validate_cwd
# ---------------------------------------------------------------------------


class ValidateCwdTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tmpdir = Path(tempfile.gettempdir())
        # /tmp is an allowed dir in the default config
        self.allowed = self.tmpdir / ("sandbox-test-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self.sub = self.allowed / "sub"
        self.sub.mkdir()

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def test_allowed_dir_and_subdir_ok(self) -> None:
        self.assertIsNone(server._validate_cwd(self.allowed.resolve(), str(self.allowed)))
        self.assertIsNone(server._validate_cwd(self.sub.resolve(), str(self.sub)))

    def test_missing_dir_error(self) -> None:
        missing = self.allowed / "does-not-exist"
        err = server._validate_cwd(missing.resolve(), "does-not-exist")
        self.assertIn("Directory not found", err)

    def test_outside_allowed_rejected(self) -> None:
        # home dir is NOT under an allowed dir (~/projects or /tmp)
        home = Path.home()
        err = server._validate_cwd(home, str(home))
        self.assertIn("not in allowed paths", err)

    def test_uses_raw_input_in_message(self) -> None:
        home = Path.home()
        err = server._validate_cwd(home, "~/user-typed-path")
        self.assertIn("~/user-typed-path", err)


# ---------------------------------------------------------------------------
# _binary_still_contained (TOCTOU narrowing)
# ---------------------------------------------------------------------------


class BinaryStillContainedTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _make_exec(self.root / "tool")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_valid_local_binary_true(self) -> None:
        self.assertTrue(
            server._binary_still_contained(str((self.root / "tool").resolve()), self.root)
        )

    def test_removed_binary_false(self) -> None:
        (self.root / "tool").unlink()
        self.assertFalse(
            server._binary_still_contained(str((self.root / "tool").resolve()), self.root)
        )

    def test_outside_path_false(self) -> None:
        self.assertFalse(
            server._binary_still_contained("/bin/sh", self.root)
        )

    def test_swapped_to_symlink_escape_false(self) -> None:
        # Replace the real tool path with a symlink escaping the tree.
        (self.root / "tool").unlink()
        link = self.root / "tool"
        link.symlink_to("/bin")
        self.assertFalse(
            server._binary_still_contained(str(link.resolve()), self.root)
        )


# ---------------------------------------------------------------------------
# _split_command
# ---------------------------------------------------------------------------


class SplitCommandTest(unittest.TestCase):
    def test_no_operator_single_segment(self) -> None:
        self.assertEqual(
            server._split_command("ls -la"),
            [(None, ["ls -la"])],
        )

    def test_semicolon_splits(self) -> None:
        self.assertEqual(
            server._split_command("echo hi; echo bye"),
            [(None, ["echo hi"]), (";", ["echo bye"])],
        )

    def test_and_and_splits(self) -> None:
        self.assertEqual(
            server._split_command("make && make test"),
            [(None, ["make"]), ("&&", ["make test"])],
        )

    def test_or_or_splits(self) -> None:
        self.assertEqual(
            server._split_command("false || echo fallback"),
            [(None, ["false"]), ("||", ["echo fallback"])],
        )

    def test_mixed_operators(self) -> None:
        self.assertEqual(
            server._split_command("a && b; c || d"),
            [(None, ["a"]), ("&&", ["b"]), (";", ["c"]), ("||", ["d"])],
        )

    def test_operator_inside_quotes_preserved(self) -> None:
        self.assertEqual(
            server._split_command('echo "a; b"'),
            [(None, ['echo "a; b"'])],
        )
        self.assertEqual(
            server._split_command("printf 'a && b'; ls"),
            [(None, ["printf 'a && b'"]), (";", ["ls"])],
        )

    def test_whitespace_and_empty_segments_dropped(self) -> None:
        self.assertEqual(
            server._split_command("  a   ;;  b  "),
            [(None, ["a"]), (";", ["b"])],
        )

    def test_empty_command(self) -> None:
        self.assertEqual(server._split_command(""), [])
        self.assertEqual(server._split_command("   "), [])

    def test_only_operator_is_empty(self) -> None:
        self.assertEqual(server._split_command(";"), [])

    def test_single_pipe_splits_into_stages(self) -> None:
        self.assertEqual(
            server._split_command("ls | wc"),
            [(None, ["ls", "wc"])],
        )

    def test_multi_stage_pipeline(self) -> None:
        self.assertEqual(
            server._split_command("a | b | c"),
            [(None, ["a", "b", "c"])],
        )

    def test_pipe_inside_quotes_preserved(self) -> None:
        self.assertEqual(
            server._split_command('echo "a|b" | wc'),
            [(None, ['echo "a|b"', "wc"])],
        )
        self.assertEqual(
            server._split_command("printf 'a | b'"),
            [(None, ["printf 'a | b'"])],
        )

    def test_pipe_distinguished_from_or_or(self) -> None:
        # '||' is the chaining OR operator, not a pipe
        self.assertEqual(
            server._split_command("false || echo fallback | wc"),
            [(None, ["false"]), ("||", ["echo fallback", "wc"])],
        )

    def test_pipe_and_chain_mix(self) -> None:
        self.assertEqual(
            server._split_command("a | b && c | d ; e"),
            [
                (None, ["a", "b"]),
                ("&&", ["c", "d"]),
                (";", ["e"]),
            ],
        )

    def test_pipe_at_start_drops_empty_lead(self) -> None:
        self.assertEqual(
            server._split_command("| ls"),
            [(None, ["ls"])],
        )

    def test_pipe_at_end_drops_empty_tail(self) -> None:
        self.assertEqual(
            server._split_command("ls |"),
            [(None, ["ls"])],
        )

    def test_triple_pipe_treated_as_or_or_plus_empty_stage(self) -> None:
        # 'a ||| b' parses as 'a || b' (the middle empty stage is dropped).
        self.assertEqual(
            server._split_command("a ||| b"),
            [(None, ["a"]), ("||", ["b"])],
        )

    def test_bare_ampersand_rejected(self) -> None:
        for cmd in ("echo hi & ls", "a && b & c", "a & b"):
            with self.assertRaises(ValueError):
                server._split_command(cmd)

    def test_ampersand_inside_quotes_preserved(self) -> None:
        # '&' inside quotes is literal text, not a rejected operator.
        self.assertEqual(
            server._split_command('echo "a & b"'),
            [(None, ['echo "a & b"'])],
        )
        self.assertEqual(
            server._split_command("printf 'x & y'"),
            [(None, ["printf 'x & y'"])],
        )


# ---------------------------------------------------------------------------
# _run_segment (chaining orchestration)
# ---------------------------------------------------------------------------


class ShellRunChainingTest(unittest.TestCase):
    """Exercise `shell_run` chaining semantics without invoking the real
    sandbox by stubbing `_run_segment`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cwd = Path(self._tmp.name)
        # /tmp is allowed by default config
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-chain-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_run_segment = server._run_segment
        self.calls: list[str] = []

    def tearDown(self) -> None:
        import shutil

        server._run_segment = self._orig_run_segment
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def _stub(self, rc_map: dict[str, int]) -> None:
        def fake(command: str, work_dir: Path, timeout: int) -> tuple[int, str]:
            self.calls.append(command)
            rc = rc_map.get(command, 0)
            return rc, f"out:{command}" if rc == 0 else f"err:{command}"

        server._run_segment = fake

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.allowed))

    def test_semicolon_runs_all_segments(self) -> None:
        self._stub({"a": 0, "b": 0})
        out = self._run("a ; b")
        self.assertEqual(self.calls, ["a", "b"])
        self.assertIn("out:a", out)
        self.assertIn("out:b", out)

    def test_andand_skips_after_failure(self) -> None:
        self._stub({"a": 1, "b": 0, "c": 0})
        out = self._run("a && b && c")
        self.assertEqual(self.calls, ["a"])
        self.assertIn("skipped", out)
        self.assertNotIn("out:b", out)
        self.assertNotIn("out:c", out)

    def test_andand_runs_after_success(self) -> None:
        self._stub({"a": 0, "b": 0})
        self._run("a && b")
        self.assertEqual(self.calls, ["a", "b"])

    def test_oror_runs_after_failure(self) -> None:
        self._stub({"a": 1, "b": 0})
        self._run("a || b")
        self.assertEqual(self.calls, ["a", "b"])

    def test_oror_skips_after_success(self) -> None:
        self._stub({"a": 0, "b": 0})
        out = self._run("a || b")
        self.assertEqual(self.calls, ["a"])
        self.assertIn("skipped", out)

    def test_resolution_failure_short_circuits_andand(self) -> None:
        # 'notallowed' fails resolution inside _run_segment (rc 1).
        self._stub({"notallowed": 1, "b": 0})
        out = self._run("notallowed && b")
        self.assertEqual(self.calls, ["notallowed"])


# ---------------------------------------------------------------------------
# shell_run pipeline orchestration
# ---------------------------------------------------------------------------


class ShellRunPipelineTest(unittest.TestCase):
    """Exercise how `shell_run` routes pipe pipelines to `_run_pipeline`,
    and applies `&&`/`||` short-circuit to a pipeline's exit code, by stubbing
    both `_run_segment` and `_run_pipeline`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-pipe-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self.segment_calls: list[str] = []
        self.pipeline_calls: list[list[str]] = []

    def tearDown(self) -> None:
        import shutil

        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def _stub(
        self,
        pipeline_rc: dict[tuple[str, ...], int] | None = None,
        segment_rc: dict[str, int] | None = None,
    ) -> None:
        pipeline_rc = pipeline_rc or {}
        segment_rc = segment_rc or {}

        def fake_pipeline(stages: list[str], work_dir, timeout):
            self.pipeline_calls.append(stages)
            rc = pipeline_rc.get(tuple(stages), 0)
            return rc, f"pipe:{'|'.join(stages)}" if rc == 0 else f"err-pipe:{'|'.join(stages)}"

        def fake_segment(command: str, work_dir, timeout):
            self.segment_calls.append(command)
            rc = segment_rc.get(command, 0)
            return rc, f"out:{command}" if rc == 0 else f"err:{command}"

        server._run_pipeline = fake_pipeline
        server._run_segment = fake_segment

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.allowed))

    def test_two_stage_pipe_routes_to_pipeline(self) -> None:
        self._stub()
        out = self._run("ls | wc")
        self.assertEqual(self.pipeline_calls, [["ls", "wc"]])
        self.assertEqual(self.segment_calls, [])
        self.assertIn("pipe:ls|wc", out)

    def test_three_stage_pipe(self) -> None:
        self._stub()
        self._run("a | b | c")
        self.assertEqual(self.pipeline_calls, [["a", "b", "c"]])

    def test_single_stage_uses_segment(self) -> None:
        self._stub()
        self._run("ls")
        self.assertEqual(self.segment_calls, ["ls"])
        self.assertEqual(self.pipeline_calls, [])

    def test_andand_skips_pipeline_after_failure(self) -> None:
        self._stub(pipeline_rc={("a", "b"): 1}, segment_rc={"c": 0})
        out = self._run("a | b && c")
        self.assertEqual(self.pipeline_calls, [["a", "b"]])
        self.assertEqual(self.segment_calls, [])
        self.assertIn("skipped", out)
        self.assertNotIn("out:c", out)

    def test_andand_runs_after_pipeline_success(self) -> None:
        self._stub(pipeline_rc={("a", "b"): 0}, segment_rc={"c": 0})
        out = self._run("a | b && c")
        self.assertEqual(self.pipeline_calls, [["a", "b"]])
        self.assertEqual(self.segment_calls, ["c"])
        self.assertIn("out:c", out)

    def test_oror_runs_after_pipeline_failure(self) -> None:
        self._stub(pipeline_rc={("a", "b"): 1}, segment_rc={"c": 0})
        self._run("a | b || c")
        self.assertEqual(self.pipeline_calls, [["a", "b"]])
        self.assertEqual(self.segment_calls, ["c"])

    def test_oror_skips_after_pipeline_success(self) -> None:
        self._stub(pipeline_rc={("a", "b"): 0}, segment_rc={"c": 0})
        out = self._run("a | b || c")
        self.assertEqual(self.pipeline_calls, [["a", "b"]])
        self.assertEqual(self.segment_calls, [])
        self.assertIn("skipped", out)

    def test_pipeline_resolution_failure_short_circuits_andand(self) -> None:
        # A denied command inside a pipeline surfaces as rc 1 from _run_pipeline.
        self._stub(pipeline_rc={("nope", "b"): 1}, segment_rc={"c": 0})
        out = self._run("nope | b && c")
        self.assertEqual(self.pipeline_calls, [["nope", "b"]])
        self.assertEqual(self.segment_calls, [])
        self.assertIn("skipped", out)

    def test_bare_ampersand_returns_error_message(self) -> None:
        out = self._run("echo hi & ls")
        self.assertIn("Unsupported '&' operator", out)
        # nothing should have been executed
        self.assertEqual(self.pipeline_calls, [])
        self.assertEqual(self.segment_calls, [])


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

    def _fake_build(self, mapping: dict[str, tuple]):
        def fake(command: str, work_dir):
            return mapping.get(command, (None, None, None, None))

        server._build_invocation = fake

    def test_real_two_stage_pipe(self) -> None:
        self._fake_build({
            "producer": ("/bin/echo", ["/bin/echo", "hello"], None, {}),
            "consumer": ("/usr/bin/wc", ["/usr/bin/wc", "-c"], None, {}),
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
            "producer": (
                "/bin/sh",
                ["/bin/sh", "-c", "echo out; while true; do echo err >&2; done"],
                None,
                {},
            ),
            "consumer": ("/usr/bin/head", ["/usr/bin/head", "-n1"], None, {}),
        })
        rc, out = server._run_pipeline(["producer", "consumer"], self.work_dir, 10)
        self.assertEqual(rc, 0)
        self.assertIn("out", out)
        self.assertIn("[stderr]", out)

    def test_pipeline_timeout_kills_stages(self) -> None:
        # A last stage that never exits must be killed and reported as a
        # timeout rather than hanging the tool.
        self._fake_build({
            "producer": ("/bin/echo", ["/bin/echo", "hi"], None, {}),
            "consumer": (
                "/bin/sh",
                ["/bin/sh", "-c", "while true; do :; done"],
                None,
                {},
            ),
        })
        rc, out = server._run_pipeline(["producer", "consumer"], self.work_dir, 1)
        self.assertEqual(rc, 1)
        self.assertIn("timed out", out)

    def test_three_stage_real_pipe(self) -> None:
        import os as _os

        grep = "/usr/bin/grep" if _os.path.exists("/usr/bin/grep") else "/bin/grep"
        self._fake_build({
            "a": ("/bin/echo", ["/bin/echo", "one\ntwo\nthree"], None, {}),
            "b": (grep, [grep, "two"], None, {}),
            "c": ("/usr/bin/wc", ["/usr/bin/wc", "-l"], None, {}),
        })
        rc, out = server._run_pipeline(["a", "b", "c"], self.work_dir, 10)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "1")


# ---------------------------------------------------------------------------
# _git_config_paths
# ---------------------------------------------------------------------------


class GitConfigPathsTest(unittest.TestCase):
    def test_paths_resolved(self) -> None:
        paths = server._git_config_paths()
        self.assertEqual(len(paths), 2)
        for p in paths:
            # must be absolute, canonical (no '..' or symlinked HOME remainder)
            self.assertTrue(Path(p).is_absolute())
            self.assertEqual(str(Path(p).resolve()), p)


# ---------------------------------------------------------------------------
# _cosmo_toolchain_paths
# ---------------------------------------------------------------------------


class CosmoToolchainPathsTest(unittest.TestCase):
    def test_paths_resolved(self) -> None:
        paths = server._cosmo_toolchain_paths()
        # toolchain tree + busybox binary + APE loader
        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(Path(p).is_absolute())
            self.assertEqual(str(Path(p).resolve()), p)
        # first path must be the vendored toolchain root; busybox must be present
        self.assertEqual(Path(paths[0]), server.COSMO_TOOLCHAIN.resolve())
        self.assertEqual(Path(paths[1]), server.BUSYBOX_BIN.resolve())

    def test_cosmocc_configured_with_local_toolchain(self) -> None:
        cfg = server.COMMANDS["cosmocc"]
        # binary must point inside the vendored toolchain, not the host install
        self.assertTrue(
            cfg["binary"].startswith(str(server.COSMO_TOOLCHAIN.resolve()))
        )
        self.assertEqual(cfg["extra_unveil_rx"], server._cosmo_toolchain_paths)


if __name__ == "__main__":
    unittest.main()
