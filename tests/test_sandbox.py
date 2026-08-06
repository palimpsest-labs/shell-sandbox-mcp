"""Tests for shell_sandbox_mcp.server security-critical helpers.

These cover path-containment, local-binary resolution, and cwd validation —
the code paths that decide what commands an agent may run. Run with the
venv python that has `mcp` installed:

    PYTHONPATH=src <venv>/bin/python -m unittest discover -s tests -v
"""

import os
import subprocess
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
            [(None, ["ls -la"], False)],
        )

    def test_semicolon_splits(self) -> None:
        self.assertEqual(
            server._split_command("echo hi; echo bye"),
            [(None, ["echo hi"], False), (";", ["echo bye"], False)],
        )

    def test_and_and_splits(self) -> None:
        self.assertEqual(
            server._split_command("make && make test"),
            [(None, ["make"], False), ("&&", ["make test"], False)],
        )

    def test_or_or_splits(self) -> None:
        self.assertEqual(
            server._split_command("false || echo fallback"),
            [(None, ["false"], False), ("||", ["echo fallback"], False)],
        )

    def test_mixed_operators(self) -> None:
        self.assertEqual(
            server._split_command("a && b; c || d"),
            [(None, ["a"], False), ("&&", ["b"], False), (";", ["c"], False), ("||", ["d"], False)],
        )

    def test_operator_inside_quotes_preserved(self) -> None:
        self.assertEqual(
            server._split_command('echo "a; b"'),
            [(None, ['echo "a; b"'], False)],
        )
        self.assertEqual(
            server._split_command("printf 'a && b'; ls"),
            [(None, ["printf 'a && b'"], False), (";", ["ls"], False)],
        )

    def test_whitespace_and_empty_segments_dropped(self) -> None:
        self.assertEqual(
            server._split_command("  a   ;;  b  "),
            [(None, ["a"], False), (";", ["b"], False)],
        )

    def test_empty_command(self) -> None:
        self.assertEqual(server._split_command(""), [])
        self.assertEqual(server._split_command("   "), [])

    def test_only_operator_is_empty(self) -> None:
        self.assertEqual(server._split_command(";"), [])

    def test_single_pipe_splits_into_stages(self) -> None:
        self.assertEqual(
            server._split_command("ls | wc"),
            [(None, ["ls", "wc"], False)],
        )

    def test_multi_stage_pipeline(self) -> None:
        self.assertEqual(
            server._split_command("a | b | c"),
            [(None, ["a", "b", "c"], False)],
        )

    def test_pipe_inside_quotes_preserved(self) -> None:
        self.assertEqual(
            server._split_command('echo "a|b" | wc'),
            [(None, ['echo "a|b"', "wc"], False)],
        )
        self.assertEqual(
            server._split_command("printf 'a | b'"),
            [(None, ["printf 'a | b'"], False)],
        )

    def test_pipe_distinguished_from_or_or(self) -> None:
        # '||' is the chaining OR operator, not a pipe
        self.assertEqual(
            server._split_command("false || echo fallback | wc"),
            [(None, ["false"], False), ("||", ["echo fallback", "wc"], False)],
        )

    def test_pipe_and_chain_mix(self) -> None:
        self.assertEqual(
            server._split_command("a | b && c | d ; e"),
            [
                (None, ["a", "b"], False),
                ("&&", ["c", "d"], False),
                (";", ["e"], False),
            ],
        )

    def test_pipe_at_start_drops_empty_lead(self) -> None:
        self.assertEqual(
            server._split_command("| ls"),
            [(None, ["ls"], False)],
        )

    def test_pipe_at_end_drops_empty_tail(self) -> None:
        self.assertEqual(
            server._split_command("ls |"),
            [(None, ["ls"], False)],
        )

    def test_triple_pipe_treated_as_or_or_plus_empty_stage(self) -> None:
        # 'a ||| b' parses as 'a || b' (the middle empty stage is dropped).
        self.assertEqual(
            server._split_command("a ||| b"),
            [(None, ["a"], False), ("||", ["b"], False)],
        )

    def test_bare_ampersand_backgrounds_pipeline(self) -> None:
        # Bare '&' marks the current pipeline as backgrounded and resets the
        # operator for the next pipeline (acting like ';' semantically).
        self.assertEqual(
            server._split_command("a & b"),
            [(None, ["a"], True), (None, ["b"], False)],
        )
        self.assertEqual(
            server._split_command("echo hi & ls"),
            [(None, ["echo hi"], True), (None, ["ls"], False)],
        )

    def test_ampersand_with_and_operator(self) -> None:
        # '&&' before a backgrounded pipeline: the '&&' join is preserved,
        # the pipeline is marked backgrounded, and the next pipeline runs
        # unconditionally (prev_op reset by '&').
        self.assertEqual(
            server._split_command("a && b & c"),
            [(None, ["a"], False), ("&&", ["b"], True), (None, ["c"], False)],
        )

    def test_double_ampersand_stays_and_operator(self) -> None:
        # '&&' is still the AND chaining operator, not backgrounding.
        self.assertEqual(
            server._split_command("a && b"),
            [(None, ["a"], False), ("&&", ["b"], False)],
        )

    def test_fd_dup_ampersand_is_not_backgrounding(self) -> None:
        # The '&' inside `2>&1` / `1>&2` is part of a redirect operator, not a
        # backgrounding operator, so the whole segment must stay intact.
        self.assertEqual(
            server._split_command("echo hi 2>&1"),
            [(None, ["echo hi 2>&1"], False)],
        )
        self.assertEqual(
            server._split_command("cmd 1>&2"),
            [(None, ["cmd 1>&2"], False)],
        )
        # Backgrounding still works when '&' is NOT preceded by '>' + digit.
        self.assertEqual(
            server._split_command("grep x 2>err &"),
            [(None, ["grep x 2>err"], True)],
        )

    def test_ampersand_inside_quotes_preserved(self) -> None:
        # '&' inside quotes is literal text, not a backgrounding operator.
        self.assertEqual(
            server._split_command('echo "a & b"'),
            [(None, ['echo "a & b"'], False)],
        )
        self.assertEqual(
            server._split_command("printf 'x & y'"),
            [(None, ["printf 'x & y'"], False)],
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
        def fake(command, work_dir: Path, timeout: int, expansion=None) -> tuple[int, str]:
            # command may be a str (legacy) or CommandNode (AST-native)
            cmd_str = (
                _serialize_command(command)
                if isinstance(command, CommandNode)
                else command
            )
            self.calls.append(cmd_str)
            rc = rc_map.get(cmd_str, 0)
            return rc, f"out:{cmd_str}" if rc == 0 else f"err:{cmd_str}"

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
    `_run_segment`, `_run_pipeline`, and `_run_background`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-pipe-" + os.urandom(4).hex())
        self.allowed.mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = getattr(server, "_run_background", None)
        self.segment_calls: list[str] = []
        self.pipeline_calls: list[list[str]] = []
        self.background_calls: list[list[str]] = []

    def tearDown(self) -> None:
        import shutil

        server._run_segment = self._orig_segment
        server._run_pipeline = self._orig_pipeline
        if self._orig_background is not None:
            server._run_background = self._orig_background
        shutil.rmtree(self.allowed, ignore_errors=True)
        self._tmp.cleanup()

    def _stub(
        self,
        pipeline_rc: dict[tuple[str, ...], int] | None = None,
        segment_rc: dict[str, int] | None = None,
        background_rc: dict[tuple[str, ...], int] | None = None,
    ) -> None:
        pipeline_rc = pipeline_rc or {}
        segment_rc = segment_rc or {}
        background_rc = background_rc or {}

        def fake_pipeline(stages, work_dir, timeout, expansion=None):
            # stages may be list[str] (legacy) or list[CommandNode] (AST-native)
            str_stages = [
                _serialize_command(s) if isinstance(s, CommandNode) else s
                for s in stages
            ]
            self.pipeline_calls.append(str_stages)
            rc = pipeline_rc.get(tuple(str_stages), 0)
            return rc, f"pipe:{'|'.join(str_stages)}" if rc == 0 else f"err-pipe:{'|'.join(str_stages)}"

        def fake_segment(command, work_dir, timeout, expansion=None):
            # command may be a str (legacy) or CommandNode (AST-native)
            cmd_str = (
                _serialize_command(command)
                if isinstance(command, CommandNode)
                else command
            )
            self.segment_calls.append(cmd_str)
            rc = segment_rc.get(cmd_str, 0)
            return rc, f"out:{cmd_str}" if rc == 0 else f"err:{cmd_str}"

        def fake_background(stages, work_dir, expansion=None):
            # stages may be list[str] (legacy) or list[CommandNode] (AST-native)
            str_stages = [
                _serialize_command(s) if isinstance(s, CommandNode) else s
                for s in stages
            ]
            self.background_calls.append(str_stages)
            rc = background_rc.get(tuple(str_stages), 0)
            return rc, f"bg:{'|'.join(str_stages)}"

        server._run_pipeline = fake_pipeline
        server._run_segment = fake_segment
        server._run_background = fake_background

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

    def test_bare_ampersand_returns_immediately_via_background(self) -> None:
        self._stub()
        out = self._run("echo hi & ls")
        # 'echo hi' is backgrounded, 'ls' is a normal segment
        self.assertEqual(self.background_calls, [["echo hi"]])
        self.assertEqual(self.segment_calls, ["ls"])
        self.assertEqual(self.pipeline_calls, [])
        self.assertIn("bg:echo hi", out)
        self.assertIn("out:ls", out)


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

    def _fake_build(self, mapping: dict[str, server.Invocation]):
        def fake(command: str, work_dir, expansion=None):
            return mapping.get(command, server.EmptyInvocation())

        server._build_invocation = fake

    def test_real_two_stage_pipe(self) -> None:
        self._fake_build({
            "producer": server.Invocation("/bin/echo", ["/bin/echo", "hello"], None, {}, []),
            "consumer": server.Invocation("/usr/bin/wc", ["/usr/bin/wc", "-c"], None, {}, []),
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
            "producer": server.Invocation(
                "/bin/sh",
                ["/bin/sh", "-c", "echo out; while true; do echo err >&2; done"],
                None,
                {},
                [],
            ),
            "consumer": server.Invocation("/usr/bin/head", ["/usr/bin/head", "-n1"], None, {}, []),
        })
        rc, out = server._run_pipeline(["producer", "consumer"], self.work_dir, 10)
        self.assertEqual(rc, 0)
        self.assertIn("out", out)
        self.assertIn("[stderr]", out)

    def test_pipeline_timeout_kills_stages(self) -> None:
        # A last stage that never exits must be killed and reported as a
        # timeout rather than hanging the tool.
        self._fake_build({
            "producer": server.Invocation("/bin/echo", ["/bin/echo", "hi"], None, {}, []),
            "consumer": server.Invocation(
                "/bin/sh",
                ["/bin/sh", "-c", "while true; do :; done"],
                None,
                {},
                [],
            ),
        })
        rc, out = server._run_pipeline(["producer", "consumer"], self.work_dir, 1)
        self.assertEqual(rc, 1)
        self.assertIn("timed out", out)

    def test_three_stage_real_pipe(self) -> None:
        import os as _os

        grep = "/usr/bin/grep" if _os.path.exists("/usr/bin/grep") else "/bin/grep"
        self._fake_build({
            "a": server.Invocation("/bin/echo", ["/bin/echo", "one\ntwo\nthree"], None, {}, []),
            "b": server.Invocation(grep, [grep, "two"], None, {}, []),
            "c": server.Invocation("/usr/bin/wc", ["/usr/bin/wc", "-l"], None, {}, []),
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


# ---------------------------------------------------------------------------
# _git_extra_rx_paths
# ---------------------------------------------------------------------------


class GitExtraRxPathsTest(unittest.TestCase):
    def test_paths_resolved(self) -> None:
        work_dir = Path("/tmp/test-work")
        paths = server._git_extra_rx_paths(work_dir)
        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(Path(p).is_absolute())
            self.assertEqual(str(Path(p).resolve()), p)
        # first path must be <work_dir>/.git/hooks
        self.assertEqual(Path(paths[0]), (work_dir / ".git" / "hooks").resolve())
        # second path must be the cred shim
        self.assertEqual(Path(paths[1]), (server.REPO_ROOT / "bin" / "git-cred-readonly").resolve())

    def test_configured_in_git_command(self) -> None:
        cfg = server.COMMANDS["git"]
        self.assertIsNotNone(cfg.get("extra_unveil_rx"))
        self.assertTrue(callable(cfg["extra_unveil_rx"]))


# ---------------------------------------------------------------------------
# _git_readonly_paths
# ---------------------------------------------------------------------------


class GitReadonlyPathsTest(unittest.TestCase):
    def test_includes_config_and_credential(self) -> None:
        paths = server._git_readonly_paths()
        # config paths (2) + credential path (1) = 3
        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(Path(p).is_absolute())
            self.assertEqual(str(Path(p).resolve()), p)
        # last path must be the credentials file
        self.assertEqual(
            Path(paths[2]),
            (Path.home().resolve() / ".git-credentials").resolve(),
        )

    def test_configured_as_extra_unveil_in_git_command(self) -> None:
        cfg = server.COMMANDS["git"]
        self.assertEqual(cfg["extra_unveil"], server._git_readonly_paths)
        # git must NOT have extra_unveil_rw anymore
        self.assertNotIn("extra_unveil_rw", cfg)


# ---------------------------------------------------------------------------
# _stage_git_global_config
# ---------------------------------------------------------------------------


class StageGitGlobalConfigTest(unittest.TestCase):
    def test_returns_path_and_sets_credential_helper(self) -> None:
        try:
            staged = server._stage_git_global_config()
        except PermissionError:
            self.skipTest("Cannot read ~/.gitconfig in sandbox")
        self.assertTrue(Path(staged).exists())
        self.assertIn("sbx-git-global-", staged)
        # Read back with git config: the staged file must point
        # credential.helper at the read-only shim.
        result = subprocess.run(
            ["/usr/bin/git", "config", "--file", staged,
             "--get", "credential.helper"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            str((server.REPO_ROOT / "bin" / "git-cred-readonly").resolve()),
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
# _run_segment with redirects (stubbed subprocess — cosmo python can't fork)
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


class ExpandCommandTest(unittest.TestCase):
    """Test _expand_command with stubbed _capture_stdout."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)
        self._orig_capture = server._capture_stdout
        self.captures: list[str] = []

    def tearDown(self) -> None:
        server._capture_stdout = self._orig_capture
        self._tmp.cleanup()

    def _stub_capture(self, outputs: dict[str, str]) -> None:
        def fake(command, work_dir, timeout, depth, deadline=None, subst_count=None, env=None):
            self.captures.append(command)
            val = outputs.get(command, "")
            return 0, val.encode("utf-8")

        server._capture_stdout = fake

    def test_unquoted_heredoc(self) -> None:
        cmd = "cat <<EOF\nhello\nworld\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        # Should contain << + sentinel
        self.assertIn("<<", expanded)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn(sentinel, exp.heredoc_bodies)
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\nworld\n")

    def test_single_quoted_delimiter_no_expansion(self) -> None:
        self._stub_capture({"echo hi": "hi"})
        cmd = "cat <<'EOF'\n$(echo hi)\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        # Body should be literal $(echo hi), not expanded
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")
        self.assertEqual(len(self.captures), 0)  # no $() expansion triggered

    def test_unquoted_heredoc_expands_dollar_paren(self) -> None:
        self._stub_capture({"echo hello": "hello"})
        cmd = "cat <<EOF\n$(echo hello)\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")

    def test_escaped_dollar_paren_in_heredoc_not_expanded(self) -> None:
        """A backslash-escaped $() in an unquoted heredoc body stays literal."""
        self._stub_capture({"echo hi": "hi"})
        cmd = "cat <<EOF\n\\$(echo hi)\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "\\$(echo hi)\n")
        self.assertEqual(len(self.captures), 0)  # no $() expansion triggered

    def test_heredoc_tab_strip(self) -> None:
        cmd = "cat <<-EOF\n\t\thello\n\tEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn("<<-", expanded)
        # Body should have ALL leading tabs stripped
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")

    def test_herestring_unquoted(self) -> None:
        cmd = "cat <<<hello"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")
        self.assertIn("<<<", expanded)

    def test_herestring_quoted(self) -> None:
        cmd = "cat <<<'hello world'"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello world\n")

    def test_herestring_expands_dollar_paren_unless_single_quoted(self) -> None:
        self._stub_capture({"echo hi": "hi"})
        # Unquoted here-string with $()
        cmd = "cat <<<$(echo hi)"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hi\n")

        # Single-quoted here-string with $() — no expansion
        self.captures.clear()
        cmd2 = "cat <<<'$(echo hi)'"
        expanded2, exp2, _program2 = _expand_command(cmd2, self.work_dir, 30, 0)
        m2 = SENTINEL_HD.search(expanded2)
        self.assertIsNotNone(m2)
        sentinel2 = f"\x01H{m2.group(1)}\x01"
        self.assertEqual(exp2.heredoc_bodies[sentinel2], "$(echo hi)\n")
        self.assertEqual(len(self.captures), 0)

    def test_command_substitution_sentinel(self) -> None:
        self._stub_capture({"echo hello": "hello"})
        cmd = "echo $(echo hello)"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        # Should contain arg sentinel
        m = SENTINEL_ARG.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01A{m.group(1)}\x01"
        self.assertIn(sentinel, exp.arg_values)
        self.assertEqual(exp.arg_values[sentinel], "hello")
        # The expanded command should have "echo <sentinel>"
        self.assertTrue(expanded.startswith("echo "))

    def test_nested_command_substitution(self) -> None:
        outputs = {"echo inner": "inner", "echo $(echo inner)": "outer"}
        self._stub_capture(outputs)
        cmd = "echo $(echo $(echo inner))"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        # The outer $() captures "echo $(echo inner)" since the inner $()
        # is inside the outer $(), and the outer _capture_stdout call is what
        # triggers the recursive expansion (via its own _expand_command call).
        # The stub short-circuits that.
        self.assertIn("echo $(echo inner)", self.captures)

    def test_unbalanced_dollar_paren_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _expand_command("echo $(unclosed", self.work_dir, 30, 0)
        self.assertIn("Unbalanced", str(ctx.exception))

    def test_missing_heredoc_terminator_error(self) -> None:
        cmd = "cat <<EOF\nhello\nworld\n"
        with self.assertRaises(ValueError) as ctx:
            _expand_command(cmd, self.work_dir, 30, 0)
        self.assertIn("not found", str(ctx.exception))

    def test_depth_limit(self) -> None:
        # Do NOT stub _capture_stdout — the depth check must fire.
        # Pass depth=MAX_SUBST_DEPTH so that when _expand_command encounters $()
        # it calls _capture_stdout with depth+1 = MAX_SUBST_DEPTH+1 which
        # triggers the limit.
        with self.assertRaises(ValueError) as ctx:
            _expand_command("echo $(echo A)", self.work_dir, 30, MAX_SUBST_DEPTH)
        self.assertIn("depth", str(ctx.exception).lower())

    def test_count_limit(self) -> None:
        # Do NOT stub _capture_stdout — the count check must fire.
        # Create many $() substitutions to exceed MAX_SUBST_COUNT.
        parts = ["$(echo {})".format(i) for i in range(MAX_SUBST_COUNT + 5)]
        cmd = "echo " + " ".join(parts)
        with self.assertRaises(ValueError) as ctx:
            _expand_command(cmd, self.work_dir, 30, 0)
        self.assertIn("count", str(ctx.exception).lower())

    def test_quotes_inside_heredoc_body_preserved(self) -> None:
        cmd = "cat <<EOF\nline with \"quotes\" and 'apostrophes'\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "line with \"quotes\" and 'apostrophes'\n")

    def test_double_quoted_delimiter_no_expansion(self) -> None:
        self._stub_capture({"echo hi": "hi"})
        cmd = 'cat <<"EOF"\n$(echo hi)\nEOF'
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        sentinel = f"\x01H{m.group(1)}\x01"
        # Body should be literal $(echo hi), not expanded
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")


class CaptureStdoutTest(unittest.TestCase):
    """Test _capture_stdout with stubbed segment/pipeline cores."""

    def setUp(self) -> None:
        self._orig_segment_core = server._run_segment_core
        self._orig_pipeline_core = server._run_pipeline_core
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        server._run_segment_core = self._orig_segment_core
        server._run_pipeline_core = self._orig_pipeline_core
        self._tmp.cleanup()

    def _stub_segment(self, mapping: dict[str, tuple[int, bytes, bytes, list]]) -> None:
        def fake(command, work_dir, timeout, expansion=None):
            if command in mapping:
                return mapping[command]
            return (0, b"", b"", [])
        server._run_segment_core = fake

    def _stub_pipeline(self, mapping: dict[tuple, tuple[int, bytes, bytes, list]]) -> None:
        def fake(segments, work_dir, timeout, expansion=None):
            key = tuple(segments)
            if key in mapping:
                return mapping[key]
            return (0, b"", b"", [])
        server._run_pipeline_core = fake

    def test_single_segment_stdout(self) -> None:
        self._stub_segment({"echo hi": (0, b"hi\n", b"", [])})
        rc, stdout = _capture_stdout("echo hi", self.work_dir, 30, 1)
        self.assertEqual(rc, 0)
        self.assertEqual(stdout, b"hi\n")

    def test_pipeline_last_stage_stdout(self) -> None:
        self._stub_pipeline({("a", "b"): (0, b"result\n", b"", [])})
        rc, stdout = _capture_stdout("a | b", self.work_dir, 30, 1)
        self.assertEqual(rc, 0)
        self.assertEqual(stdout, b"result\n")

    def test_andand_short_circuit(self) -> None:
        self._stub_segment({
            "fail": (1, b"", b"fail", []),
            "succeed": (0, b"good", b"", []),
        })
        rc, stdout = _capture_stdout("fail && succeed", self.work_dir, 30, 1)
        self.assertEqual(rc, 1)
        self.assertEqual(stdout, b"")  # succeed skipped

    def test_background_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _capture_stdout("echo hi &", self.work_dir, 30, 1)
        self.assertIn("background", str(ctx.exception).lower())

    def test_output_truncated(self) -> None:
        long_output = b"x" * (MAX_SUBST_OUTPUT + 100)
        self._stub_segment({"big": (0, long_output, b"", [])})
        rc, stdout = _capture_stdout("big", self.work_dir, 30, 1)
        self.assertLessEqual(len(stdout), MAX_SUBST_OUTPUT)

    def test_empty_command(self) -> None:
        rc, stdout = _capture_stdout("", self.work_dir, 30, 1)
        self.assertEqual(rc, 0)
        self.assertEqual(stdout, b"")


class ResolveFdTargetsStdinTest(unittest.TestCase):
    """Test that _resolve_fd_targets returns stdin_bytes for heredoc/here-string."""

    def test_herestring_returns_stdin_bytes(self) -> None:
        redirs = [Redirect(fd=0, op="<<<", body="hello\n")]
        result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertEqual(len(result), 7)
        stdout_t, stderr_t, to_close, report, srf, stdin_b, stdin_f = result
        self.assertEqual(stdin_b, b"hello\n")
        self.assertIsNone(stdin_f)
        self.assertIn("[stdin <<<]", report)

    def test_heredoc_returns_stdin_bytes(self) -> None:
        redirs = [Redirect(fd=0, op="<<", body="line1\nline2\n")]
        result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        _, _, _, report, _, stdin_b, stdin_f = result
        self.assertEqual(stdin_b, b"line1\nline2\n")
        self.assertIsNone(stdin_f)
        self.assertIn("[stdin <<]", report)

    def test_heredoc_tab_returns_stdin_bytes(self) -> None:
        redirs = [Redirect(fd=0, op="<<-", body="tabbed\n", strip_tabs=True)]
        result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        _, _, _, report, _, stdin_b, _ = result
        self.assertEqual(stdin_b, b"tabbed\n")
        self.assertIn("[stdin <<-]", report)

    def test_no_stdin_redirect_returns_none(self) -> None:
        redirs = [Redirect(fd=1, op=">", raw_target="out.txt", target_path="/tmp/out.txt")]
        result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertEqual(len(result), 7)
        self.assertIsNone(result[5])  # stdin_bytes is None
        self.assertIsNone(result[6])  # stdin_file is None

    def test_multiple_stdin_rejected_by_resolve(self) -> None:
        redirs = [
            Redirect(fd=0, op="<<", body="a\n"),
            Redirect(fd=0, op="<<", body="b\n"),
        ]
        with self.assertRaises(ValueError) as ctx:
            _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertIn("Multiple stdin redirects", str(ctx.exception))

    def test_input_redirect_returns_stdin_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            infile = Path(tmp) / "in.txt"
            infile.write_text("data\n")
            redirs = [Redirect(fd=0, op="<", raw_target=str(infile), target_path=str(infile))]
            result = _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
            self.assertEqual(len(result), 7)
            stdout_t, stderr_t, to_close, report, srf, stdin_b, stdin_f = result
            self.assertIsNone(stdin_b)
            self.assertIsNotNone(stdin_f)
            self.assertIn(f"[stdin <- {infile}]", report)
            self.assertIn(stdin_f, to_close)
            # Sanity: the file object actually reads the file content.
            self.assertEqual(stdin_f.read(), b"data\n")
            stdin_f.close()

    def test_input_redirect_missing_file_raises(self) -> None:
        redirs = [Redirect(fd=0, op="<", raw_target="nope.txt", target_path="/nonexistent/nope.txt")]
        with self.assertRaises(ValueError) as ctx:
            _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
        self.assertIn("Input redirect file not found", str(ctx.exception))

    def test_input_redirect_conflicts_with_heredoc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            infile = Path(tmp) / "in.txt"
            infile.write_text("x\n")
            redirs = [
                Redirect(fd=0, op="<", raw_target=str(infile), target_path=str(infile)),
                Redirect(fd=0, op="<<", body="x\n"),
            ]
            with self.assertRaises(ValueError) as ctx:
                _resolve_fd_targets(redirs, subprocess.PIPE, subprocess.PIPE)
            self.assertIn("Multiple stdin redirects", str(ctx.exception))


class BuildInvocationHeredocTest(unittest.TestCase):
    """Test that _build_invocation threads expansion correctly."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_heredoc_redirect_passed_through(self) -> None:
        expansion = Expansion(
            arg_values={},
            heredoc_bodies={"\x01H0\x01": "body\n"},
        )
        inv = _build_invocation(
            "cat << \x01H0\x01", self.root, expansion=expansion,
        )
        self.assertIsInstance(inv, Invocation)
        self.assertEqual(len(inv.redirects), 1)
        self.assertEqual(inv.redirects[0].fd, 0)
        self.assertEqual(inv.redirects[0].op, "<<")
        self.assertEqual(inv.redirects[0].body, "body\n")


class RunSegmentCoreStdinTest(unittest.TestCase):
    """Test that _run_segment_core passes stdin_bytes via input= to subprocess.run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_build = server._build_invocation
        self._orig_run = server.subprocess.run

    def tearDown(self) -> None:
        server._build_invocation = self._orig_build
        server.subprocess.run = self._orig_run
        self._tmp.cleanup()

    def test_stdin_bytes_passed_to_subprocess_run(self) -> None:
        import subprocess as _sp

        captured_input = []

        def fake_run(args, **kwargs):
            captured_input.append(kwargs.get("input"))
            return _sp.CompletedProcess(args, 0, stdout=b"ok\n", stderr=b"")

        server.subprocess.run = fake_run

        def fake_build(command, work_dir, expansion=None):
            return Invocation(
                "/usr/bin/cat",
                ["/usr/bin/cat"],
                None,
                {},
                [Redirect(fd=0, op="<<", body="hello\n")],
            )

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_segment_core(
            "cat << DUMMY", self.root, 30,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(stdout_b, b"ok\n")
        self.assertEqual(captured_input[0], b"hello\n")

    def test_no_stdin_when_no_heredoc(self) -> None:
        import subprocess as _sp

        captured_input = []

        def fake_run(args, **kwargs):
            captured_input.append(kwargs.get("input"))
            return _sp.CompletedProcess(args, 0, stdout=b"", stderr=b"")

        server.subprocess.run = fake_run

        def fake_build(command, work_dir, expansion=None):
            return Invocation(
                "/usr/bin/echo",
                ["/usr/bin/echo", "hi"],
                None,
                {},
                [],
            )

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_segment_core(
            "echo hi", self.root, 30,
        )
        self.assertIsNone(captured_input[0])

    def test_input_redirect_passes_stdin_file_not_input(self) -> None:
        import subprocess as _sp

        infile = self.root / "in.txt"
        infile.write_text("data\n")
        captured = {}

        def fake_run(args, **kwargs):
            captured["input"] = kwargs.get("input")
            captured["stdin"] = kwargs.get("stdin")
            return _sp.CompletedProcess(args, 0, stdout=b"ok\n", stderr=b"")

        server.subprocess.run = fake_run

        def fake_build(command, work_dir, expansion=None):
            return Invocation(
                "/usr/bin/cat",
                ["/usr/bin/cat"],
                None,
                {},
                [Redirect(fd=0, op="<", raw_target=str(infile), target_path=str(infile))],
            )

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_segment_core(
            f"cat < {infile}", self.root, 30,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(stdout_b, b"ok\n")
        # Input redirect must go via stdin=<file>, NOT input=<bytes>.
        self.assertIsNone(captured.get("input"))
        self.assertIsNotNone(captured.get("stdin"))
        self.assertIn(f"[stdin <- {infile}]", report)


class RunPipelineCoreStdinTest(unittest.TestCase):
    """Test _run_pipeline_core stdin plumbing and non-first-stage rejection."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig_build = server._build_invocation
        self._orig_popen = server.subprocess.Popen

    def tearDown(self) -> None:
        server._build_invocation = self._orig_build
        server.subprocess.Popen = self._orig_popen
        self._tmp.cleanup()

    def test_heredoc_on_first_stage_uses_stdin_pipe(self) -> None:
        """First stage with heredoc gets stdin=PIPE with writer thread."""
        import io

        captured_stdins = []
        stdin_buffers = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                captured_stdins.append(kwargs.get("stdin"))
                # When stdin=subprocess.PIPE, provide a writable BytesIO buffer
                if kwargs.get("stdin") is subprocess.PIPE:
                    self.stdin = io.BytesIO()
                    stdin_buffers.append(self.stdin)
                else:
                    self.stdin = None
                self.stdout = _FakePipe()
                self.stderr = _FakePipe()
                self.pid = 9999
                self.returncode = 0

            def poll(self):
                return 0
            def wait(self):
                return 0
            def communicate(self, timeout=None):
                return (b"output\n", b"")
            def kill(self):
                pass

        class _FakePipe:
            def close(self):
                pass
            def read(self):
                return b""

        def fake_build(command, work_dir, expansion=None):
            if "cat" in command:
                return Invocation(
                    "/usr/bin/cat",
                    ["/usr/bin/cat"],
                    None,
                    {},
                    [Redirect(fd=0, op="<<", body="hello\n")],
                )
            if "grep" in command:
                return Invocation(
                    "/usr/bin/grep",
                    ["/usr/bin/grep", "x"],
                    None,
                    {},
                    [],
                )
            return EmptyInvocation()

        server._build_invocation = fake_build
        server.subprocess.Popen = FakePopen

        rc, stdout_b, stderr_b, report = _run_pipeline_core(
            ["cat << H0", "grep x"], self.root, 30,
        )
        self.assertEqual(rc, 0)
        # First stage should have gotten subprocess.PIPE for stdin
        self.assertIs(captured_stdins[0], subprocess.PIPE)
        # Second stage should have gotten the prev.stdout (which is the FakePipe)
        self.assertIsNotNone(captured_stdins[1])

    def test_heredoc_on_non_first_stage_rejected(self) -> None:
        """Non-first stage with heredoc should be rejected."""
        def fake_build(command, work_dir, expansion=None):
            if "echo" in command:
                return Invocation(
                    "/usr/bin/echo",
                    ["/usr/bin/echo", "hi"],
                    None,
                    {},
                    [],
                )
            if "cat" in command:
                return Invocation(
                    "/usr/bin/cat",
                    ["/usr/bin/cat"],
                    None,
                    {},
                    [Redirect(fd=0, op="<<", body="hello\n")],
                )
            return EmptyInvocation()

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_pipeline_core(
            ["echo hi", "cat << H0"], self.root, 30,
        )
        self.assertEqual(rc, 1)
        self.assertIn(b"not allowed on non-first", stdout_b)

    def test_input_redirect_on_first_stage_passes_file_stdin(self) -> None:
        """First-stage < file passes the file object as stdin= (not PIPE)."""
        infile = self.root / "in.txt"
        infile.write_text("data\n")

        captured_stdins = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                captured_stdins.append(kwargs.get("stdin"))
                self.stdin = None
                self.stdout = _FakePipe()
                self.stderr = _FakePipe()
                self.pid = 9999
                self.returncode = 0

            def poll(self):
                return 0
            def wait(self):
                return 0
            def communicate(self, timeout=None):
                return (b"output\n", b"")
            def kill(self):
                pass

        class _FakePipe:
            def close(self):
                pass
            def read(self):
                return b""

        def fake_build(command, work_dir, expansion=None):
            if "cat" in command:
                return Invocation(
                    "/usr/bin/cat",
                    ["/usr/bin/cat"],
                    None,
                    {},
                    [Redirect(fd=0, op="<", raw_target=str(infile), target_path=str(infile))],
                )
            if "grep" in command:
                return Invocation(
                    "/usr/bin/grep",
                    ["/usr/bin/grep", "x"],
                    None,
                    {},
                    [],
                )
            return EmptyInvocation()

        server._build_invocation = fake_build
        server.subprocess.Popen = FakePopen

        rc, stdout_b, stderr_b, report = _run_pipeline_core(
            [f"cat < {infile}", "grep x"], self.root, 30,
        )
        self.assertEqual(rc, 0)
        # First stage stdin must be the open file object, not a pipe.
        self.assertIsNot(captured_stdins[0], subprocess.PIPE)
        self.assertIsNotNone(captured_stdins[0])
        # The returned report is the LAST stage's; the first stage's stdin
        # report line must still have been produced internally (checked by
        # the file-object stdin above).

    def test_input_redirect_on_non_first_stage_rejected(self) -> None:
        """< file on a non-first stage is rejected like a heredoc."""
        def fake_build(command, work_dir, expansion=None):
            if "echo" in command:
                return Invocation(
                    "/usr/bin/echo",
                    ["/usr/bin/echo", "hi"],
                    None,
                    {},
                    [],
                )
            if "cat" in command:
                return Invocation(
                    "/usr/bin/cat",
                    ["/usr/bin/cat"],
                    None,
                    {},
                    [Redirect(fd=0, op="<", raw_target="in.txt", target_path="/tmp/in.txt")],
                )
            return EmptyInvocation()

        server._build_invocation = fake_build

        rc, stdout_b, stderr_b, report = _run_pipeline_core(
            ["echo hi", "cat < in.txt"], self.root, 30,
        )
        self.assertEqual(rc, 1)
        self.assertIn(b"not allowed on non-first", stdout_b)


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
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        # Verify the expanded command has a heredoc sentinel
        self.assertIn("<<", expanded)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn(sentinel, exp.heredoc_bodies)
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")
        args, redirs, err = _extract_redirects(
            "cat << " + sentinel, expansion=exp,
        )
        self.assertIsNone(err)
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].body, "hello\n")

    def test_heredoc_single_quoted_literal(self) -> None:
        """Verify single-quoted delimiters produce literal bodies."""
        cmd = "cat <<'EOF'\n$(echo hi)\nEOF"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")

    def test_herestring_expansion(self) -> None:
        """Verify here-string produces correct body."""
        cmd = "cat <<<hello world"
        expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello world\n")

    def test_command_substitution_single_arg(self) -> None:
        """Verify $(...) produces a sentinel with single-word value."""
        original = server._capture_stdout
        try:
            def fake(command, work_dir, timeout, depth, deadline=None, subst_count=None, env=None):
                return 0, b"a b"
            server._capture_stdout = fake
            cmd = "echo $(printf 'a b')"
            expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
            m = SENTINEL_ARG.search(expanded)
            self.assertIsNotNone(m)
            sentinel = f"\x01A{m.group(1)}\x01"
            self.assertEqual(exp.arg_values[sentinel], "a b")
            args, redirs, err = _extract_redirects(
                "echo " + sentinel, expansion=exp,
            )
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
            expanded, exp, _program = _expand_command(cmd, self.work_dir, 30, 0)
            m = SENTINEL_ARG.search(expanded)
            self.assertIsNotNone(m)
            sentinel = f"\x01A{m.group(1)}\x01"
            self.assertEqual(exp.arg_values.get(sentinel), "x")
        finally:
            server._capture_stdout = original


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

        def spy_extract(segment, expansion=None):
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


# ---------------------------------------------------------------------------
# cd builtin tests — stub _run_segment / _run_pipeline and record work_dir
# ---------------------------------------------------------------------------


class ShellRunCdTest(unittest.TestCase):
    """Exercise the per-call ``cd`` builtin (AST primary path).

    Stubs ``_run_segment`` to record the ``work_dir`` argument so we can
    assert ``cd`` changes the working directory for subsequent segments
    within the same ``shell_run`` invocation.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # Create an allowed cwd under /tmp with a real subdirectory inside it.
        self.allowed = Path(tempfile.gettempdir()) / ("sandbox-cd-" + os.urandom(4).hex())
        self.allowed.mkdir()
        (self.allowed / "sub").mkdir()
        self._orig_segment = server._run_segment
        self._orig_pipeline = server._run_pipeline
        self._orig_background = server._run_background
        self.segment_calls: list[tuple[str, str]] = []  # (cmd_str, work_dir)
        self.segment_rc_map: dict[str, int] = {}

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

    def _cmd_str(self, command) -> str:
        """Convert a CommandNode or str to a display string for stub maps."""
        if isinstance(command, CommandNode):
            return _serialize_command(command)
        return str(command)

    def _stub_segments(self) -> None:
        """Stub _run_segment to record (cmd_str, work_dir) for every call."""
        rc_map = self.segment_rc_map

        def fake_segment(command, work_dir: Path, timeout: int,
                         expansion=None) -> tuple[int, str]:
            c = self._cmd_str(command)
            self.segment_calls.append((c, str(work_dir)))
            rc = rc_map.get(c, 0)
            return rc, f"out:{c}" if rc == 0 else f"err:{c}"

        server._run_segment = fake_segment

    def _stub_pipeline(self) -> None:
        def fake_pipeline(segments, work_dir: Path, timeout: int,
                          expansion=None) -> tuple[int, str]:
            return 0, "pipeline-ok"

        server._run_pipeline = fake_pipeline

    def _stub_background(self) -> None:
        def fake_background(segments, work_dir: Path,
                            expansion=None) -> tuple[int, str]:
            return 0, "bg"

        server._run_background = fake_background

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.allowed))

    # ------------------------------------------------------------------
    # cd sub && cmd  —  cmd sees the updated work_dir
    # ------------------------------------------------------------------

    def test_cd_sub_and_cmd_uses_updated_work_dir(self) -> None:
        """cd sub && echo hi  →  echo runs inside <allowed>/sub."""
        self._stub_segments()
        out = self._run("cd sub && echo hi")
        # echo should have been called with work_dir = <allowed>/sub
        self.assertEqual(len(self.segment_calls), 1,
                         f"Expected 1 segment call (echo), got {self.segment_calls}")
        cmd_str, wd = self.segment_calls[0]
        self.assertEqual(cmd_str, "echo hi")
        expected = str((self.allowed / "sub").resolve())
        self.assertEqual(wd, expected,
                         f"echo work_dir: expected {expected}, got {wd}")
        self.assertIn("out:echo hi", out)

    def test_cd_sub_semicolon_cmd_uses_updated_work_dir(self) -> None:
        """cd sub ; echo hi  →  echo runs inside <allowed>/sub."""
        self._stub_segments()
        out = self._run("cd sub ; echo hi")
        self.assertEqual(len(self.segment_calls), 1)
        cmd_str, wd = self.segment_calls[0]
        self.assertEqual(cmd_str, "echo hi")
        expected = str((self.allowed / "sub").resolve())
        self.assertEqual(wd, expected)
        self.assertIn("out:echo hi", out)

    # ------------------------------------------------------------------
    # cd nonexistent && cmd  →  cmd skipped (cd fails, rc=1)
    # ------------------------------------------------------------------

    def test_cd_nonexistent_skips_and(self) -> None:
        """cd nonexistent && echo hi  →  echo skipped, error reported."""
        self._stub_segments()
        out = self._run("cd nonexistent && echo hi")
        self.assertEqual(len(self.segment_calls), 0,
                         "echo should NOT have been called")
        self.assertIn("Directory not found", out)
        self.assertIn("skipped", out)

    def test_cd_nonexistent_skips_and_in_output(self) -> None:
        """Verify the skip message appears after a failed cd."""
        self._stub_segments()
        out = self._run("cd no_such_dir && echo x")
        self.assertIn("Directory not found", out)
        self.assertIn("skipped", out)

    # ------------------------------------------------------------------
    # cd /etc && cmd  →  rejected (not in allowed dirs), cmd skipped
    # ------------------------------------------------------------------

    def test_cd_etc_rejected(self) -> None:
        """cd /etc && echo hi  →  rejected, echo skipped."""
        self._stub_segments()
        out = self._run("cd /etc && echo hi")
        self.assertEqual(len(self.segment_calls), 0)
        self.assertIn("not in allowed paths", out)
        self.assertIn("skipped", out)

    # ------------------------------------------------------------------
    # cd .. from allowed root  →  escape rejected
    # ------------------------------------------------------------------

    def test_cd_dotdot_escapes_allowed_root(self) -> None:
        """cd .. from an allowed root (like /tmp) escapes → rejected."""
        self._stub_segments()
        # Run with cwd=/tmp (an allowed root). cd .. → /  (not allowed).
        out = server.shell_run("cd .. && echo hi", cwd="/tmp")
        self.assertIn("not in allowed paths", out)
        self.assertIn("skipped", out)

    # ------------------------------------------------------------------
    # bare cd  →  error, && chain skips
    # ------------------------------------------------------------------

    def test_bare_cd_error_and_skip(self) -> None:
        """Bare cd → 'cd: no directory', && chain skipped."""
        self._stub_segments()
        out = self._run("cd && echo hi")
        self.assertEqual(len(self.segment_calls), 0)
        self.assertIn("cd: no directory", out)
        self.assertIn("skipped", out)

    # ------------------------------------------------------------------
    # cd too many args
    # ------------------------------------------------------------------

    def test_cd_too_many_args(self) -> None:
        """cd a b → 'cd: too many arguments'."""
        self._stub_segments()
        out = self._run("cd a b && echo hi")
        self.assertIn("cd: too many arguments", out)
        self.assertEqual(len(self.segment_calls), 0)

    # ------------------------------------------------------------------
    # standalone cd sub  →  silent success
    # ------------------------------------------------------------------

    def test_standalone_cd_sub_silent(self) -> None:
        """cd sub alone returns '(no output)'."""
        self._stub_segments()
        out = self._run("cd sub")
        self.assertEqual(out, "(no output)")
        self.assertEqual(len(self.segment_calls), 0,
                         "No segment should have been dispatched for cd")

    def test_standalone_cd_sub_ast_fast_path(self) -> None:
        """cd sub alone via the AST fast path returns '(no output)'."""
        self._stub_segments()
        out = self._run("cd sub")
        self.assertEqual(out, "(no output)")

    # ------------------------------------------------------------------
    # cd with dot (self)
    # ------------------------------------------------------------------

    def test_cd_dot_is_noop(self) -> None:
        """cd . is a no-op — subsequent cmd runs in the same dir."""
        self._stub_segments()
        out = self._run("cd . && echo hi")
        self.assertEqual(len(self.segment_calls), 1)
        _cmd_str, wd = self.segment_calls[0]
        self.assertEqual(wd, str(self.allowed.resolve()))
        self.assertIn("out:echo hi", out)

    # ------------------------------------------------------------------
    # cd with ~ expansion
    # ------------------------------------------------------------------

    def test_cd_tilde_expands_then_rejected_not_in_allowed(self) -> None:
        """cd ~ expands HOME (not in allowed dirs) → 'not in allowed paths'.

        HOME is not in the test's allowed set, so ``cd ~`` must fail the
        containment check — proving ``expanduser`` ran on the target before
        joining with work_dir (previously it returned "Directory not found: ~"
        because ``~`` was never expanded).
        """
        self._stub_segments()
        out = server.shell_run("cd ~ && echo hi", cwd=str(self.allowed))
        self.assertEqual(len(self.segment_calls), 0)
        self.assertIn("not in allowed paths", out)
        self.assertNotIn("Directory not found: ~", out)

    # ------------------------------------------------------------------
    # cd -- <dir> (end-of-options)
    # ------------------------------------------------------------------

    def test_cd_dashdash_sub_and_cmd_uses_updated_work_dir(self) -> None:
        """cd -- sub && echo hi  →  echo runs inside <allowed>/sub."""
        self._stub_segments()
        out = self._run("cd -- sub && echo hi")
        self.assertEqual(len(self.segment_calls), 1,
                         f"Expected 1 segment call (echo), got {self.segment_calls}")
        cmd_str, wd = self.segment_calls[0]
        self.assertEqual(cmd_str, "echo hi")
        expected = str((self.allowed / "sub").resolve())
        self.assertEqual(wd, expected,
                         f"echo work_dir: expected {expected}, got {wd}")
        self.assertIn("out:echo hi", out)

    def test_cd_dashdash_alone_no_directory(self) -> None:
        """cd -- alone → 'cd: no directory' (like bare cd)."""
        self._stub_segments()
        out = self._run("cd -- && echo hi")
        self.assertEqual(len(self.segment_calls), 0)
        self.assertIn("cd: no directory", out)
        self.assertIn("skipped", out)


if __name__ == "__main__":
    unittest.main()
