"""Tests for python3 environment (PYTHONPATH extras, dynamic version, venv
detection).  Run with the venv python that has `mcp` installed:

    PYTHONPATH=src <venv>/bin/python -m unittest discover -s tests -v
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from shell_sandbox_mcp import executor, server
from shell_sandbox_mcp.server import (
    COMMANDS,
    Invocation,
    _build_invocation,
    _maybe_venv_cfg,
    _python_version,
    _resolve_command,
    _resolve_venv_fallback,
)


# ---------------------------------------------------------------------------
# _python_version — dynamic python version (B2 fix)
# ---------------------------------------------------------------------------


class PythonVersionTest(unittest.TestCase):
    """Test that _python_version queries the binary once and caches."""

    def test_returns_version_and_is_cached(self) -> None:
        """Mock subprocess.run; assert cached result and single call."""
        # Clear any cached value from earlier imports.
        _python_version.cache_clear()

        with patch("shell_sandbox_mcp.config.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="3.12\n")
            v1 = _python_version()
            v2 = _python_version()

            self.assertEqual(v1, "3.12")
            self.assertEqual(v2, "3.12")
            mock_run.assert_called_once()

    def test_raises_runtime_error_on_failure(self) -> None:
        """If the subprocess fails, a RuntimeError propagates."""
        _python_version.cache_clear()

        with patch("shell_sandbox_mcp.config.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
            with self.assertRaises(RuntimeError) as ctx:
                _python_version()
            self.assertIn("Failed to determine python version",
                          str(ctx.exception))


# ---------------------------------------------------------------------------
# _maybe_venv_cfg — venv detection when _resolve_local_binary succeeds
# ---------------------------------------------------------------------------


class MaybeVenvCfgTest(unittest.TestCase):
    """Test _maybe_venv_cfg detection of venv pythons."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._python3_binary = COMMANDS["python3"]["binary"]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_returns_none_for_non_musl_binary(self) -> None:
        """A resolved binary that is NOT the vendored musl python returns None."""
        cfg = _maybe_venv_cfg("some/tool", self.root, "/usr/bin/python3")
        self.assertIsNone(cfg)

    def test_returns_none_when_no_pyvenv_cfg(self) -> None:
        """The vendored musl python invoked without a venv layout returns None."""
        # Create a plain directory with a fake python binary (musl path).
        # No pyvenv.cfg present, so it should not be treated as a venv.
        cfg = _maybe_venv_cfg("bin/python", self.root, self._python3_binary)
        self.assertIsNone(cfg)

    def test_detects_venv_with_pyvenv_cfg(self) -> None:
        """The vendored musl python inside a venv (with pyvenv.cfg) gets a venv cfg."""
        # Create a minimal venv layout.
        venv_root = self.root / ".venv"
        venv_bin = venv_root / "bin"
        venv_bin.mkdir(parents=True)
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n")

        resolved_binary = self._python3_binary
        cfg = _maybe_venv_cfg(".venv/bin/python", self.root, resolved_binary)
        self.assertIsNotNone(cfg)
        self.assertNotIn("prepend_args", cfg)
        self.assertTrue(cfg.get("is_venv"))
        self.assertEqual(
            Path(cfg["site_dir_name"]).resolve(),
            venv_root.resolve(),
        )

    def test_cmd_name_absolute_path(self) -> None:
        """Absolute venv path is handled correctly."""
        venv_root = self.root / ".venv"
        venv_bin = venv_root / "bin"
        venv_bin.mkdir(parents=True)
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n")

        cfg = _maybe_venv_cfg(
            str(venv_bin / "python"), self.root, self._python3_binary,
        )
        self.assertIsNotNone(cfg)
        self.assertTrue(cfg.get("is_venv"))


# ---------------------------------------------------------------------------
# _resolve_venv_fallback — venv detection when containment fails
# ---------------------------------------------------------------------------


class ResolveVenvFallbackTest(unittest.TestCase):
    """Test _resolve_venv_fallback for venvs outside the work tree."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._python3_binary = COMMANDS["python3"]["binary"]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_returns_none_when_no_pyvenv_cfg(self) -> None:
        """Without a pyvenv.cfg, fallback returns None."""
        cfg = _resolve_venv_fallback("some/dir/python", self.root)
        self.assertIsNone(cfg)

    def test_returns_cfg_when_venv_points_to_musl(self) -> None:
        """When the symlink resolves to the vendored musl python, returns a venv cfg."""
        python3_path = Path(self._python3_binary).resolve()

        # Create venv layout with bin/python as a symlink to the musl python.
        venv_root = self.root / ".venv"
        venv_bin = venv_root / "bin"
        venv_bin.mkdir(parents=True)
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n")
        # Create a symlink; resolve it to check the target.
        venv_python = venv_bin / "python"
        venv_python.symlink_to(python3_path)

        cfg = _resolve_venv_fallback(".venv/bin/python", self.root)
        self.assertIsNotNone(cfg)
        self.assertNotIn("prepend_args", cfg)
        self.assertTrue(cfg.get("is_venv"))
        self.assertEqual(
            Path(cfg["site_dir_name"]).resolve(),
            venv_root.resolve(),
        )
        # binary must be the allowlisted vendored musl python (not the
        # resolved path that may live outside the work_dir).
        self.assertEqual(
            Path(cfg["binary"]).resolve(),
            Path(self._python3_binary).resolve(),
        )

    def test_returns_none_when_target_not_musl(self) -> None:
        """If the symlink points somewhere else, fallback returns None."""
        venv_root = self.root / ".venv"
        venv_bin = venv_root / "bin"
        venv_bin.mkdir(parents=True)
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n")
        venv_python = venv_bin / "python"
        venv_python.symlink_to("/usr/bin/python3")  # not the musl python

        cfg = _resolve_venv_fallback(".venv/bin/python", self.root)
        self.assertIsNone(cfg)


# ---------------------------------------------------------------------------
# _build_invocation — PYTHONPATH with pythonpath_extra src/
# ---------------------------------------------------------------------------


class BuildInvocationPythonPathExtraTest(unittest.TestCase):
    """Test that _build_invocation includes src/ in PYTHONPATH for python3."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # _build_invocation uses executor._python_version (a separate
        # module-level name binding from server._python_version), so we
        # must patch the executor binding.  Also clear the shared lru_cache
        # so we don't depend on test ordering / cached values from
        # PythonVersionTest.
        _python_version.cache_clear()
        self._orig_pyver = executor._python_version
        executor._python_version = lambda: "3.12"

    def tearDown(self) -> None:
        executor._python_version = self._orig_pyver
        self._tmp.cleanup()

    def test_pythonpath_includes_src_when_present(self) -> None:
        """When work_dir/src/ exists, PYTHONPATH appends it after site-packages."""
        (self.root / "src").mkdir()
        inv = _build_invocation("python3 -c 'import sys'", self.root)
        self.assertIsInstance(inv, Invocation)
        self.assertIsNotNone(inv.env)
        pythonpath = inv.env.get("PYTHONPATH", "")
        parts = pythonpath.split(os.pathsep)
        self.assertGreaterEqual(len(parts), 2,
                                f"Expected >=2 PYTHONPATH parts, got {parts}")
        # First entry must be the .py-site site-packages.
        self.assertIn(".py-site", parts[0])
        # Last entry must be src/.
        self.assertEqual(
            Path(parts[-1]).resolve(),
            (self.root / "src").resolve(),
        )

    def test_pythonpath_no_src_when_absent(self) -> None:
        """When work_dir/src/ does NOT exist, it is not added to PYTHONPATH."""
        # Ensure src/ is NOT present.
        src_dir = self.root / "src"
        if src_dir.exists():
            src_dir.rmdir()
        inv = _build_invocation("python3 -c 'import sys'", self.root)
        self.assertIsInstance(inv, Invocation)
        self.assertIsNotNone(inv.env)
        pythonpath = inv.env.get("PYTHONPATH", "")
        parts = pythonpath.split(os.pathsep)
        # Only site-packages — no src.
        self.assertEqual(len(parts), 1,
                         f"Expected 1 PYTHONPATH part, got {parts}")
        self.assertIn(".py-site", parts[0])

    def test_site_packages_uses_dynamic_version(self) -> None:
        """site-packages path uses the version from _python_version."""
        inv = _build_invocation("python3 -c 'import sys'", self.root)
        self.assertIsInstance(inv, Invocation)
        self.assertIsNotNone(inv.env)
        pythonpath = inv.env.get("PYTHONPATH", "")
        # The path should contain "python3.12" (the mocked version).
        self.assertIn("python3.12", pythonpath)

    def test_src_escaping_work_dir_is_skipped(self) -> None:
        """A pythonpath_extra dir that resolves outside work_dir is dropped."""
        # Create src/ as a symlink pointing outside work_dir.
        outside = Path(tempfile.mkdtemp(dir="/tmp"))
        try:
            src_link = self.root / "src"
            src_link.symlink_to(outside)

            inv = _build_invocation("python3 -c 'import sys'", self.root)
            self.assertIsInstance(inv, Invocation)
            self.assertIsNotNone(inv.env)
            pythonpath = inv.env.get("PYTHONPATH", "")
            parts = pythonpath.split(os.pathsep)
            # Only site-packages — the escaping src/ must be skipped.
            self.assertEqual(len(parts), 1,
                             f"Expected 1 part, got {parts}")
            self.assertIn(".py-site", parts[0])
        finally:
            outside.rmdir()


# ---------------------------------------------------------------------------
# _resolve_command — venv python resolution integration
# ---------------------------------------------------------------------------


class ResolveCommandVenvTest(unittest.TestCase):
    """Test that _resolve_command returns a venv cfg for venv pythons."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_venv_python_gets_prepend_args(self) -> None:
        """A venv/bin/python command resolves without prepend_args."""
        python3_path = Path(COMMANDS["python3"]["binary"]).resolve()

        # Create a venv layout.
        venv_root = self.root / ".venv"
        venv_bin = venv_root / "bin"
        venv_bin.mkdir(parents=True)
        (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n")
        venv_python = venv_bin / "python"
        venv_python.symlink_to(python3_path)

        binary, final_args, cfg = _resolve_command(
            [".venv/bin/python", "-c", "print(1)"],
            work_dir=self.root,
        )
        self.assertIsNotNone(binary)
        self.assertIsNotNone(cfg)
        self.assertTrue(cfg.get("is_venv"),
                        f"Expected is_venv=True, got cfg={cfg}")
        self.assertNotIn("prepend_args", cfg)
        # The binary must be the allowlisted vendored musl python.
        self.assertEqual(
            Path(binary).resolve(),
            python3_path,
        )

    def test_plain_local_binary_not_venv(self) -> None:
        """A regular local binary (not in a venv) does NOT get is_venv."""
        # Create a plain executable script.
        script = self.root / "myscript"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o755)

        binary, final_args, cfg = _resolve_command(
            ["myscript"], work_dir=self.root,
        )
        self.assertIsNotNone(binary)
        self.assertIsNotNone(cfg)
        self.assertNotIn("is_venv", cfg,
                         "Plain local binary should not be flagged as venv")
        self.assertNotIn("prepend_args", cfg)


if __name__ == "__main__":
    unittest.main()
