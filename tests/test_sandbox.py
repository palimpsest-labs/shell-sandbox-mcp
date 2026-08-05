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

        def fake_pipeline(stages: list[str], work_dir, timeout):
            self.pipeline_calls.append(stages)
            rc = pipeline_rc.get(tuple(stages), 0)
            return rc, f"pipe:{'|'.join(stages)}" if rc == 0 else f"err-pipe:{'|'.join(stages)}"

        def fake_segment(command: str, work_dir, timeout):
            self.segment_calls.append(command)
            rc = segment_rc.get(command, 0)
            return rc, f"out:{command}" if rc == 0 else f"err:{command}"

        def fake_background(stages: list[str], work_dir):
            self.background_calls.append(stages)
            rc = background_rc.get(tuple(stages), 0)
            return rc, f"bg:{'|'.join(stages)}"

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

    def _fake_build(self, mapping: dict[str, tuple]):
        def fake(command: str, work_dir):
            return mapping.get(command, (None, None, None, None, []))

        server._build_invocation = fake

    def test_real_two_stage_pipe(self) -> None:
        self._fake_build({
            "producer": ("/bin/echo", ["/bin/echo", "hello"], None, {}, []),
            "consumer": ("/usr/bin/wc", ["/usr/bin/wc", "-c"], None, {}, []),
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
                [],
            ),
            "consumer": ("/usr/bin/head", ["/usr/bin/head", "-n1"], None, {}, []),
        })
        rc, out = server._run_pipeline(["producer", "consumer"], self.work_dir, 10)
        self.assertEqual(rc, 0)
        self.assertIn("out", out)
        self.assertIn("[stderr]", out)

    def test_pipeline_timeout_kills_stages(self) -> None:
        # A last stage that never exits must be killed and reported as a
        # timeout rather than hanging the tool.
        self._fake_build({
            "producer": ("/bin/echo", ["/bin/echo", "hi"], None, {}, []),
            "consumer": (
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
            "a": ("/bin/echo", ["/bin/echo", "one\ntwo\nthree"], None, {}, []),
            "b": (grep, [grep, "two"], None, {}, []),
            "c": ("/usr/bin/wc", ["/usr/bin/wc", "-l"], None, {}, []),
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
        self.assertEqual(err, "Input redirects are not supported")

    def test_input_heredoc_error(self) -> None:
        args, redirs, err = self._extract("cmd << EOF")
        self.assertEqual(err, "Input redirects are not supported")

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
        self.assertIn("escapes working directory", err)

    def test_absolute_escape(self) -> None:
        redirs = [server.Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target='/etc/passwd')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNotNone(err)
        self.assertIn("escapes working directory", err)

    def test_symlink_escape(self) -> None:
        (self.root / "evil").symlink_to("/etc")
        redirs = [server.Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target='evil/hostname')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNotNone(err)
        self.assertIn("escapes working directory", err)

    def test_2gt1_passes_through(self) -> None:
        redirs = [server.Redirect(fd=2, op='>&', target_path=None, target_fd=1, raw_target='1')]
        validated, err = server._validate_redirect_paths(redirs, self.root)
        self.assertIsNone(err)
        self.assertEqual(validated, redirs)


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
        binary, sandbox_args, env, cfg, redirects = server._build_invocation(
            "echo hi > out.txt", self.root,
        )
        self.assertIsNotNone(binary)
        self.assertIsNotNone(sandbox_args)
        self.assertEqual(len(redirects), 1)
        self.assertEqual(redirects[0].target_path, str((self.root / "out.txt").resolve()))
        # Ensure > and out.txt are NOT in sandbox_args
        self.assertNotIn(">", sandbox_args)
        self.assertNotIn("out.txt", sandbox_args)

    def test_escape_path_error(self) -> None:
        err, binary, sandbox_args, cfg, redirects = server._build_invocation(
            "echo > ../escape", self.root,
        )
        self.assertIsNotNone(err)
        self.assertIn("escapes working directory", err)
        self.assertIsNone(binary)

    def test_invalid_fd_error(self) -> None:
        err, binary, sandbox_args, cfg, redirects = server._build_invocation(
            "echo 3> f", self.root,
        )
        self.assertIsNotNone(err)
        self.assertIn("only support fds 1 and 2", err)


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
        def fake_build(command, work_dir):
            return (
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
        def fake_build(command, work_dir):
            if command == "producer > f":
                return (
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "echo", "hi"],
                    None,
                    {},
                    [server.Redirect(fd=1, op='>', target_path=str(work_dir / "f"), target_fd=None, raw_target="f")],
                )
            if command == "consumer":
                return (
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "cat"],
                    None,
                    {},
                    [],
                )
            return (None, None, None, None, [])

        server._build_invocation = fake_build
        rc, out = server._run_pipeline(["producer > f", "consumer"], self.root, 10)
        self.assertEqual(rc, 1)
        self.assertIn("Cannot redirect stdout of intermediate pipe stage", out)

    def test_last_stage_stdout_redirect(self) -> None:
        outfile = self.root / "out.txt"

        def fake_build(command, work_dir):
            if command == "producer":
                return (
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "echo", "hello"],
                    None,
                    {},
                    [],
                )
            if command == f"consumer > {outfile}":
                return (
                    str(server.BUSYBOX_BIN.resolve()),
                    [str(server.BUSYBOX_BIN.resolve()), "cat"],
                    None,
                    {},
                    [server.Redirect(fd=1, op='>', target_path=str(outfile), target_fd=None, raw_target="out.txt")],
                )
            return (None, None, None, None, [])

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

        def fake_build(command, work_dir):
            return (
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

        def fake_build(command, work_dir):
            return (
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


if __name__ == "__main__":
    unittest.main()
