"""Tests for APE loader bootstrap in bin/run-sandbox.

Covers:
- Unit: running the wrapper with an empty HOME bootstraps the loader.
- E2E: cosmocc-compiled APE binary runs through the sandbox.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = REPO_ROOT / "bin"
VENDORED_APE_X86_64 = BIN_DIR / "cosmo-toolchain" / "bin" / "ape-x86_64.elf"


class ApeLoaderBootstrapTest(unittest.TestCase):
    """Unit: run-sandbox bootstraps $HOME/.ape-1.10 when missing/stale."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ape-test-")
        self.fake_home = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_bootstrap_creates_loader_in_empty_home(self) -> None:
        if not VENDORED_APE_X86_64.is_file():
            self.skipTest("vendored APE loader not found")
        wrapper = BIN_DIR / "run-sandbox"
        self.assertTrue(wrapper.is_file(), f"wrapper missing: {wrapper}")

        # Run the wrapper with HOME pointed at the empty temp dir.
        # The wrapper bootstrap should create $HOME/.ape-1.10 before
        # exec-ing sandbox. Use true (busybox) so sandbox exits quickly.
        env = os.environ.copy()
        env["HOME"] = str(self.fake_home)
        # unset SANDBOX_UNVEIL_RX so the wrapper sets it fresh
        env.pop("SANDBOX_UNVEIL_RX", None)
        result = subprocess.run(
            [str(wrapper), "stdio rpath wpath cpath", "/tmp", "--", "true"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True, text=True, timeout=15,
        )
        # binfmt_misc handles the sandbox binary itself on this host;
        # the important assertion is that the loader was bootstrapped.
        ape_loader = self.fake_home / ".ape-1.10"
        self.assertTrue(
            ape_loader.is_file(),
            f"APE loader not created at {ape_loader}; "
            f"wrapper stdout: {result.stdout}, stderr: {result.stderr}",
        )
        # Verify byte-identical to vendored copy.
        subprocess.run(
            ["cmp", "-s", str(VENDORED_APE_X86_64), str(ape_loader)],
            check=True,
        )
        # Permissions: must be executable.
        self.assertTrue(os.access(ape_loader, os.X_OK),
                        f"APE loader not executable: {ape_loader}")

    def test_bootstrap_noop_when_loader_exists_and_matches(self) -> None:
        """When ~/.ape-1.10 already matches, no temp file dance occurs."""
        if not VENDORED_APE_X86_64.is_file():
            self.skipTest("vendored APE loader not found")
        wrapper = BIN_DIR / "run-sandbox"

        # Pre-populate with a byte-identical copy.
        ape_loader = self.fake_home / ".ape-1.10"
        shutil.copy(VENDORED_APE_X86_64, ape_loader)
        os.chmod(ape_loader, 0o755)
        orig_mtime = ape_loader.stat().st_mtime

        env = os.environ.copy()
        env["HOME"] = str(self.fake_home)
        env.pop("SANDBOX_UNVEIL_RX", None)
        subprocess.run(
            [str(wrapper), "stdio rpath wpath cpath", "/tmp", "--", "true"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True, text=True, timeout=15,
        )
        # File must still exist and mtime unchanged (no re-copy).
        self.assertTrue(ape_loader.is_file())
        self.assertEqual(ape_loader.stat().st_mtime, orig_mtime)


class ApeLoaderE2ETest(unittest.TestCase):
    """E2E: cosmocc-compiled APE binary runs through the sandbox."""

    def setUp(self) -> None:
        # Create temp dir under repo root (not /tmp which is noexec).
        self._tmp_base = tempfile.mkdtemp(
            dir=str(REPO_ROOT),
            prefix=".ape-e2e-",
        )
        self.work_dir = Path(self._tmp_base)

    def tearDown(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.work_dir))

    def test_cosmocc_compile_and_run(self) -> None:
        """Compile a trivial C program with cosmocc and run the APE binary."""
        hi_c = self.work_dir / "hi.c"
        hi_c.write_text(
            '#include <stdio.h>\n'
            'int main(void) { puts("ok"); return 0; }\n'
        )
        compile_out = self._run("cosmocc -o hi hi.c")
        self.assertNotIn("Exit code:", compile_out,
                         f"cosmocc compilation failed:\n{compile_out}")
        self.assertTrue((self.work_dir / "hi").is_file(),
                        f"Binary not produced: {list(self.work_dir.iterdir())}")

        run_out = self._run("./hi")
        self.assertIn("ok", run_out)
        self.assertNotIn("Exit code:", run_out)


if __name__ == "__main__":
    unittest.main()
