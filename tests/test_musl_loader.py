"""Tests for musl loader fallback — dynamically-linked local binaries."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from shell_sandbox_mcp import server
from shell_sandbox_mcp.config import MUSL_LOADER, MUSL_RTLIB


class MuslLoaderFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temp dir under the project root (not /tmp, which is noexec).
        # Use a fixed prefix so the dir is easily identifiable and cleaned up.
        self._tmp_base = tempfile.mkdtemp(
            dir=str(Path(__file__).resolve().parent.parent),
            prefix=".musl-test-",
        )
        self.work_dir = Path(self._tmp_base)

    def tearDown(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def _run(self, command: str) -> str:
        return server.shell_run(command, cwd=str(self.work_dir))

    def test_local_binary_cfg_sets_musl_loader_env_and_rtlib_unveil(self) -> None:
        """A plain local binary must get the loader env + rtlib rx unveil."""
        prog = self.work_dir / "myprog"
        prog.write_text("#!/bin/sh\necho hi\n")
        os.chmod(prog, 0o755)

        _bin, _args, cfg = server._resolve_command(["myprog"], self.work_dir)
        self.assertTrue(cfg.get("is_local_binary"))
        self.assertEqual(
            cfg.get("env", {}).get("SANDBOX_MUSL_LOADER"), str(MUSL_LOADER.resolve())
        )
        self.assertEqual(
            cfg.get("env", {}).get("SANDBOX_MUSL_RTLIB"), str(MUSL_RTLIB.resolve())
        )
        rx = cfg["extra_unveil_rx"](self.work_dir) \
            if callable(cfg.get("extra_unveil_rx")) else cfg.get("extra_unveil_rx")
        self.assertIn(str(MUSL_RTLIB.resolve()), rx or [])

    def test_allowlisted_commands_do_not_set_musl_loader_env(self) -> None:
        """Non-local commands (git/make/python3) must NOT get the loader env,
        so the fallback branch is dead for them (regression guard against
        accidentally broadening the loader fallback to allowlisted commands)."""
        for cmd in ("git", "make", "python3"):
            _bin, _args, cfg = server._resolve_command([cmd], self.work_dir)
            env = cfg.get("env", {})
            self.assertNotIn("SANDBOX_MUSL_LOADER", env, f"{cmd} leaks loader env")
            self.assertNotIn("SANDBOX_MUSL_RTLIB", env, f"{cmd} leaks rtlib env")

    def test_dynamic_musl_binary_runs_via_loader_fallback(self) -> None:
        hello_c = self.work_dir / "hello.c"
        hello_c.write_text(
            '#include <stdio.h>\n'
            'int main(void) { printf("hello dynamic musl\\n"); return 0; }\n'
        )
        compile_out = self._run(
            "gcc -o hello hello.c"
            " -Wl,-dynamic-linker=/lib/ld-musl-x86_64.so.1"
        )
        # gcc succeeds silently: shell_run returns "(no output)" on success
        # and "Exit code: N" only on failure. Assert the binary was produced.
        self.assertNotIn("Exit code:", compile_out,
                         f"Compilation failed:\n{compile_out}")
        self.assertTrue((self.work_dir / "hello").is_file())
        run_out = self._run("./hello")
        self.assertIn("hello dynamic musl", run_out)
        self.assertNotIn("Exit code:", run_out)
