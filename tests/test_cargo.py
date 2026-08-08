"""Tests for cargo — promise tokens and end-to-end build.

Contract tests verify the pledge tokens cargo is configured with. The e2e
test compiles and runs a tiny Rust binary; like the musl-loader e2e, it only
works when the sandbox can spawn its own sandbox binary (i.e. the real MCP
env, not a sandbox-in-sandbox), so failures there should be re-checked in the
real environment rather than treated as a policy regression.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.server import _build_invocation, Invocation


class CargoPromisesContractTest(unittest.TestCase):
    """cargo's pledge tokens must cover what a build needs: flock (Cargo.lock
    + target-dir locking), fattr (rustc sets build-artifact mtimes/perms),
    inet/dns (fetch crates from a registry), and proc/prot_exec (spawn rustc).
    cargo must NOT opt out of pledge (unlike git's no_pledge=True)."""

    def _cfg(self):
        _bin, _args, cfg = server._resolve_command(["cargo"])
        self.assertIsNotNone(cfg)
        return cfg

    def test_flock_present(self) -> None:
        self.assertIn("flock", self._cfg()["promises"],
                      "cargo needs flock to lock Cargo.lock / target dir")

    def test_fattr_present(self) -> None:
        self.assertIn("fattr", self._cfg()["promises"],
                      "cargo/rustc set file mtimes/perms on build artifacts")

    def test_proc_and_prot_exec_present(self) -> None:
        for tok in ("proc", "prot_exec"):
            self.assertIn(tok, self._cfg()["promises"])

    def test_network_tokens_present(self) -> None:
        self.assertIn("inet", self._cfg()["promises"])
        self.assertIn("dns", self._cfg()["promises"])

    def test_no_no_pledge(self) -> None:
        self.assertFalse(server.COMMANDS["cargo"].get("no_pledge", False),
                         "cargo must keep pledge() active (unveil+pledge boundary)")


class CargoHomeEnvTest(unittest.TestCase):
    """_build_invocation must redirect CARGO_HOME into the workspace so cargo's
    registry cache / config stay inside the unveiled tree instead of $HOME/.cargo."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_cargo_invocation_sets_cargo_home_in_workdir(self) -> None:
        inv = _build_invocation("cargo build", self.root)
        self.assertIsInstance(inv, Invocation)
        self.assertIsNotNone(inv.env)
        ch = inv.env.get("CARGO_HOME")
        self.assertIsNotNone(ch, "cargo invocation must set CARGO_HOME")
        # must be inside the work_dir (the unveiled tree)
        self.assertEqual(str(Path(ch).resolve()), str((self.root / ".cargo-home").resolve()))
        self.assertTrue(Path(ch).is_dir(), "CARGO_HOME dir should be created")

    def test_non_cargo_commands_do_not_set_cargo_home(self) -> None:
        inv = _build_invocation("make build", self.root)
        self.assertIsInstance(inv, Invocation)
        self.assertIsNotNone(inv.env)
        self.assertNotIn("CARGO_HOME", inv.env)


class CargoEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_base = tempfile.mkdtemp(
            dir=str(Path(__file__).resolve().parent.parent),
            prefix=".cargo-test-",
        )
        self.work_dir = Path(self._tmp_base)
        (self.work_dir / "src").mkdir()
        (self.work_dir / "Cargo.toml").write_text(
            '[package]\nname="hello"\nversion="0.1.0"\nedition="2021"\n'
        )
        (self.work_dir / "src" / "main.rs").write_text(
            'fn main() { println!("hello from rust"); }\n'
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_cargo_build_and_run(self) -> None:
        build = server.shell_run("cargo build", cwd=str(self.work_dir), timeout=60)
        self.assertNotIn("Exit code:", build, f"cargo build failed:\n{build}")
        hello = self.work_dir / "target" / "debug" / "hello"
        self.assertTrue(hello.is_file(), f"binary not produced:\n{build}")
        run = server.shell_run("./target/debug/hello", cwd=str(self.work_dir))
        self.assertIn("hello from rust", run)


if __name__ == "__main__":
    unittest.main()
