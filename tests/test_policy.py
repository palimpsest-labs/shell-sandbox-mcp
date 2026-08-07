"""Tests for git/cosmo policy path helpers. Run with the venv python that has `mcp` installed:

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
    COMMANDS,
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
# _musl_toolchain_paths
# ---------------------------------------------------------------------------


class MuslToolchainPathsTest(unittest.TestCase):
    def test_paths_resolved(self) -> None:
        paths = server._musl_toolchain_paths()
        # toolchain tree + busybox binary
        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(Path(p).is_absolute())
            self.assertEqual(str(Path(p).resolve()), p)
        # first path must be the vendored musl toolchain root; busybox must be present
        self.assertEqual(Path(paths[0]), server.MUSL_TOOLCHAIN.resolve())
        self.assertEqual(Path(paths[1]), server.BUSYBOX_BIN.resolve())

    def test_musl_gcc_configured_with_local_toolchain(self) -> None:
        cfg = server.COMMANDS["musl-gcc"]
        # binary must point inside the vendored musl toolchain
        self.assertTrue(
            cfg["binary"].startswith(str(server.MUSL_TOOLCHAIN.resolve()))
        )
        self.assertEqual(cfg["extra_unveil_rx"], server._musl_toolchain_paths)

    def test_musl_toolchain_binaries_are_real_bootlin_names(self) -> None:
        # Bootlin's buildroot toolchain uses the x86_64-buildroot-linux-musl
        # triple. The gcc/cc drivers are the `.br_real` binaries: the public
        # `<triple>-gcc`/`-cc` are symlinks to a `toolchain-wrapper` that
        # breaks under the sandbox (argv[0] gets resolved to the symlink
        # target, so it looks for the non-existent `toolchain-wrapper.br_real`).
        expected = {
            "musl-gcc": "x86_64-buildroot-linux-musl-gcc.br_real",
            "musl-cc": "x86_64-buildroot-linux-musl-cc.br_real",
            "musl-ld": "x86_64-buildroot-linux-musl-ld",
            "musl-ar": "x86_64-buildroot-linux-musl-ar",
            # generic compiler names now resolve to the musl toolchain too
            "gcc": "x86_64-buildroot-linux-musl-gcc.br_real",
            "cc": "x86_64-buildroot-linux-musl-cc.br_real",
            "clang": "x86_64-buildroot-linux-musl-gcc.br_real",
        }
        for cmd, name in expected.items():
            cfg = server.COMMANDS[cmd]
            self.assertTrue(
                cfg["binary"].startswith(str(server.MUSL_TOOLCHAIN.resolve()))
            )
            # binary name must be the expected real Bootlin binary
            self.assertEqual(Path(cfg["binary"]).name, name)
            self.assertEqual(cfg["extra_unveil_rx"], server._musl_toolchain_paths)
            # the resolved binary must actually exist
            self.assertTrue(
                Path(cfg["binary"]).is_file(),
                f"{cmd} binary should exist: {cfg['binary']}",
            )

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
# no_pledge per-command flag tests
# ---------------------------------------------------------------------------


class NoPledgeFlagTest(unittest.TestCase):
    def test_git_has_no_pledge_flag(self) -> None:
        cfg = COMMANDS["git"]
        self.assertTrue(cfg.get("no_pledge"), "git must have no_pledge=True")

    def test_only_git_uses_no_pledge(self) -> None:
        no_pledge_cmds = [k for k, v in COMMANDS.items() if v.get("no_pledge")]
        self.assertEqual(no_pledge_cmds, ["git"],
                         f"Expected only 'git' to use no_pledge, got {no_pledge_cmds}")

    def test_no_pledge_not_on_busybox_or_other_commands(self) -> None:
        """Ensure no_pledge is NOT set on commands that don't need it."""
        for name, cfg in COMMANDS.items():
            if name == "git":
                continue
            self.assertFalse(
                cfg.get("no_pledge", False),
                f"{name} must NOT have no_pledge=True (security: "
                f"no_pledge strips seccomp from a potentially dangerous command)",
            )



if __name__ == "__main__":
    unittest.main()
