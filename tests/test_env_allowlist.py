"""Tests for environment allowlist hardening (_base_env, _build_invocation).

Validates that:
- _base_env drops unknown/credential vars from the host environment
- _base_env keeps allowlisted vars
- _build_invocation always returns a dict (no env=None leak)
- PWD and TMPDIR are deliberately omitted from the allowlist
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shell_sandbox_mcp import server
from shell_sandbox_mcp.server import _base_env, _build_invocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class BaseEnvTest(unittest.TestCase):
    """Test that _base_env correctly filters the host environment."""

    def test_allowlisted_vars_present(self) -> None:
        """All vars in the allowlist that exist in os.environ are kept."""
        env = _base_env()
        for k in server._ENV_ALLOWLIST:
            if k in os.environ:
                self.assertIn(k, env, f"Allowlisted var {k} should be in _base_env")
                self.assertEqual(env[k], os.environ[k])

    def test_unknown_var_dropped(self) -> None:
        """_base_env must drop vars NOT in the allowlist."""
        with patch.dict(os.environ, {"SBX_FAKE_SECRET": "top-secret"}, clear=False):
            env = _base_env()
            self.assertNotIn("SBX_FAKE_SECRET", env,
                             "Non-allowlisted var must be dropped by _base_env")

    def test_pwd_omitted(self) -> None:
        """PWD is deliberately excluded from the allowlist (cwd= is set explicitly)."""
        env = _base_env()
        self.assertNotIn("PWD", env, "PWD should not be in allowlist")

    def test_tmpdir_omitted(self) -> None:
        """TMPDIR is deliberately excluded (sandbox uses /tmp directly via unveil)."""
        env = _base_env()
        self.assertNotIn("TMPDIR", env, "TMPDIR should not be in allowlist")


# ---------------------------------------------------------------------------
# _build_invocation — no host-env leak
# ---------------------------------------------------------------------------


class BuildInvocationEnvTest(unittest.TestCase):
    """Test that _build_invocation does not leak host env vars to subprocess."""

    def _dummy_work_dir(self) -> Path:
        """Create a minimal temporary dir and return it."""
        return Path(tempfile.mkdtemp())

    def test_env_never_none(self) -> None:
        """_build_invocation must always return a dict for env, never None."""
        inv = _build_invocation(
            "echo hello", self._dummy_work_dir(),
        )
        self.assertIsNotNone(inv.sandbox_args, "Should resolve echo as a command")
        self.assertIsInstance(inv.env, dict, "env must always be a dict")
        # allowlisted vars present
        self.assertIn("PATH", inv.env)
        self.assertIn("HOME", inv.env)

    def test_no_host_secret_leak(self) -> None:
        """Set a fake secret in os.environ; assert it is NOT in the subprocess env."""
        with patch.dict(os.environ, {"SBX_SECRET_TOKEN": "leaked-value"}, clear=False):
            inv = _build_invocation(
                "echo hello", self._dummy_work_dir(),
            )
            self.assertIsNotNone(inv.sandbox_args)
            self.assertIsInstance(inv.env, dict)
            self.assertNotIn("SBX_SECRET_TOKEN", inv.env,
                             "Non-allowlisted env var must not leak to subprocess")

    def test_allowlisted_vars_in_subprocess_env(self) -> None:
        """Allowlisted vars like HOME should be in the returned env."""
        inv = _build_invocation(
            "echo hello", self._dummy_work_dir(),
        )
        self.assertIsNotNone(inv.sandbox_args)
        self.assertIsInstance(inv.env, dict)
        if "HOME" in os.environ:
            self.assertEqual(inv.env["HOME"], os.environ["HOME"])
        if "USER" in os.environ:
            self.assertEqual(inv.env["USER"], os.environ["USER"])

    @patch("shell_sandbox_mcp.server._stage_git_global_config")
    def test_git_env_vars_present(self, mock_stage: unittest.mock.Mock) -> None:
        """Git commands get GIT_CONFIG_GLOBAL and SANDBOX_NO_PLEDGE in env."""
        mock_stage.return_value = "/tmp/fake-git-config"
        inv = _build_invocation(
            "git status", self._dummy_work_dir(),
        )
        self.assertIsNotNone(inv.sandbox_args)
        self.assertIsInstance(inv.env, dict)
        # GIT_CONFIG_GLOBAL and SANDBOX_NO_PLEDGE should be in git's env
        self.assertIn("GIT_CONFIG_GLOBAL", inv.env)
        self.assertIn("SANDBOX_NO_PLEDGE", inv.env)

    @patch("subprocess.run")
    def test_env_passed_to_subprocess(self, mock_run: unittest.mock.Mock) -> None:
        """Verify that the env dict from _build_invocation is passed to subprocess.run."""
        from shell_sandbox_mcp.server import _run_segment_core
        mock_run.return_value = unittest.mock.Mock(
            returncode=0, stdout=b"hello", stderr=b"",
        )
        with patch.dict(os.environ, {"SBX_SECRET": "value"}, clear=False):
            _run_segment_core("echo hello", self._dummy_work_dir(), 30)
        self.assertTrue(mock_run.called)
        # Check that env kwarg was passed and does NOT contain SBX_SECRET
        call_kwargs = mock_run.call_args.kwargs
        self.assertIn("env", call_kwargs)
        env_passed = call_kwargs["env"]
        self.assertIsInstance(env_passed, dict)
        self.assertNotIn("SBX_SECRET", env_passed)


if __name__ == "__main__":
    unittest.main()
