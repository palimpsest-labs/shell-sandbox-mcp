"""Configuration — paths, limits, environment allowlist.

Constants and helper functions for sandbox paths, timeout/limit defaults,
and host-environment filtering.  No dependencies on other sandbox modules.
"""

import functools
import os
import subprocess
import tempfile
from pathlib import Path

from .parser import MAX_HEREDOC_BODY, MAX_SUBST_COUNT, MAX_SUBST_DEPTH, MAX_SUBST_OUTPUT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SANDBOX_BIN = REPO_ROOT / "bin" / "sandbox"
SANDBOX_WRAPPER = REPO_ROOT / "bin" / "run-sandbox"
BUSYBOX_BIN = REPO_ROOT / "bin" / "busybox"
COSMO_TOOLCHAIN = REPO_ROOT / "bin" / "cosmo-toolchain"
MUSL_TOOLCHAIN = REPO_ROOT / "bin" / "musl-toolchain"
DEFAULT_ALLOWED_DIRS = [
    str(Path.home() / "projects"),
    "/tmp",
]
# Redirect targets may live inside the working directory OR under these extra
# roots (e.g. /tmp, and /dev/null for `> /dev/null` / `< /dev/null`). Used by
# _validate_redirect_paths. Only *absolute* targets are checked against these
# roots; relative targets always resolve against the working directory.
EXTRA_REDIRECT_ROOTS = [
    Path("/tmp").resolve(),
    Path("/dev/null").resolve(),
]
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300
MAX_OUTPUT = 1_000_000  # 1 MB


@functools.lru_cache(maxsize=1)
def _cosmo_py_version() -> str:
    """Return the Python major.minor version string for the vendored cosmo
    python binary, e.g. ``"3.12"``.  Queried once per process and cached so
    repeated calls are free.

    Raises :class:`RuntimeError` if the cosmo python binary is missing or
    the subprocess fails for any reason.
    """
    binary = REPO_ROOT / "bin" / "cosmo" / "python"
    try:
        out = subprocess.run(
            [str(binary), "-S", "-c",
             "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to determine cosmo python version from {binary}: {exc}"
        ) from exc
    return out.stdout.strip()

# ---------------------------------------------------------------------------
# Environment allowlist
# ---------------------------------------------------------------------------

_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM",
    "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "LC_TIME", "TZ",
    # TMPDIR: build tools (GCC/make/configure) consult it. The sandbox still
    # confines the filesystem via unveil to work_dir + /tmp, so a host TMPDIR
    # pointing under /tmp is fine.
    "TMPDIR",
)


def _base_env() -> dict[str, str]:
    """Allowlisted host env for a sandboxed subprocess.  Unlisted vars DROPPED.

    Per-command unveil_env (SANDBOX_*, PYTHONUSERBASE, PYTHONPATH,
    GIT_CONFIG_GLOBAL) are layered on by _build_invocation and are NOT part
    of this allowlist — they are always added regardless.
    """
    return {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
