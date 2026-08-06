"""Command allowlist and resolution — git helpers, toolchain paths,
COMMANDS table, BUSYBOX_APPLETS, and the command resolver.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .config import BUSYBOX_BIN, COSMO_TOOLCHAIN, REPO_ROOT
from .containment import _contained_path


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git_config_paths() -> list[str]:
    """Return the git config dotfiles under $HOME that must be unveiled.

    Covers the user-global config (~/.gitconfig) and the XDG config file
    (~/.config/git/config). Missing paths are ignored by the sandbox.
    Paths are resolved to their canonical form, matching the rest of the code.
    """
    home = Path.home().resolve()
    return [
        str((home / ".gitconfig").resolve()),
        str((home / ".config" / "git" / "config").resolve()),
    ]


def _git_credential_paths() -> list[str]:
    """Return the git credential store file under $HOME that must be unveiled.

    Now unveiled read-ONLY: the read-only credential helper shim
    (bin/git-cred-readonly) handles `get` and no-ops `store`, so the
    sandboxed git never writes ~/.git-credentials (which would need wc on
    $HOME — not granted).
    """
    return [
        str((Path.home().resolve() / ".git-credentials").resolve()),
    ]


def _git_extra_rx_paths(work_dir: Path) -> list[str]:
    """Read-execute paths git needs inside the sandbox.

    - <work_dir>/.git/hooks: so git can exec hook scripts (pre-push,
      post-commit, ...) installed by tools like git-lfs. rx only (no write):
      git runs existing hooks, it never writes them here. The dir may not
      exist yet; unveil tolerates ENOENT.
    - bin/git-cred-readonly: the read-only credential helper shim (Issue 2),
      which git must exec when credential.helper fires.
    """
    return [
        str((work_dir / ".git" / "hooks").resolve()),
        str((REPO_ROOT / "bin" / "git-cred-readonly").resolve()),
    ]


# Staged sandbox-global git config (idempotent, atomic write).
# Written once per python process under /tmp; reused across invocations.
_GIT_GLOBAL_CONFIG = Path(tempfile.gettempdir()) / f"sbx-git-global-{os.getuid()}-{os.getpid()}.config"
_GIT_CRED_SHIM = REPO_ROOT / "bin" / "git-cred-readonly"


def _git_readonly_paths() -> list[str]:
    """Read-only $HOME paths git needs: global/XDG config AND the credential
    store file. The cred file is read-ONLY here because the read-only helper
    shim (bin/git-cred-readonly) handles `get` and no-ops `store`, so the
    sandboxed git never writes ~/.git-credentials (which would need wc on
    $HOME — not granted)."""
    return _git_config_paths() + _git_credential_paths()


def _stage_git_global_config() -> str:
    """Write a sandbox-global git config = copy of ~/.gitconfig with
    `credential.helper` replaced by the read-only shim. Idempotent: content
    is identical across invocations, so concurrent writes are safe. Returns
    the path to set as GIT_CONFIG_GLOBAL."""
    cfg = _GIT_GLOBAL_CONFIG
    src = Path.home() / ".gitconfig"
    tmp = cfg.with_suffix(".tmp")
    if src.is_file():
        tmp.write_bytes(src.read_bytes())
    else:
        tmp.write_text("")
    subprocess.run(
        ["/usr/bin/git", "config", "--file", str(tmp),
         "--replace-all", "credential.helper", str(_GIT_CRED_SHIM.resolve())],
        check=True,
    )
    os.chmod(tmp, 0o600)
    os.replace(tmp, cfg)  # atomic
    return str(cfg)


def _cosmo_toolchain_paths(work_dir: Optional[Path] = None) -> list[str]:
    """Return the paths that must be unveiled read-execute for the vendored
    cosmocc toolchain to run: the toolchain tree itself (its compilers read
    headers/libs from it), the busybox binary (used as a non-preserving `mv`
    by build tools), and the Cosmopolitan APE loader under $HOME. The loaders
    at /usr/bin are already unveiled by the sandbox default.
    """
    return [
        str(COSMO_TOOLCHAIN.resolve()),
        str(BUSYBOX_BIN.resolve()),
        str((Path.home().resolve() / ".ape-1.10").resolve()),
    ]


def _cosmo_toolchain_bin() -> str:
    """Return the vendored toolchain's bin directory.

    Prepended to PATH for build commands so that `mv` (a busybox applet
    symlinked into the toolchain bin) is resolved instead of GNU /usr/bin/mv,
    which would otherwise try to preserve file attributes that the sandbox
    blocks (noisy but harmless warnings during builds).
    """
    return str((COSMO_TOOLCHAIN / "bin").resolve())


COMMANDS = {
    "git": {
        "binary": "/usr/bin/git",
        "promises": "stdio rpath wpath cpath prot_exec inet dns proc",
        "description": "Git version control",
        "extra_unveil": _git_readonly_paths,      # config + cred file, READ-ONLY
        "extra_unveil_rx": _git_extra_rx_paths,   # .git/hooks + cred shim
        "is_git": True,
        "no_pledge": True,
    },
    "cargo": {
        "binary": "cargo",
        "promises": "stdio rpath wpath cpath proc prot_exec",
        "description": "Rust package manager (build, test, check, fmt, clippy)",
    },
    "make": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "make").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": (
            "GNU make build tool (vendored). Spawns compiler subprocesses, "
            "so it needs exec access to the toolchain."
        ),
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    "cosmocc": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "cosmocc").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": (
            "Cosmopolitan C/C++ compiler (vendored toolchain). Compiles "
            "portable APE binaries for both x86_64 and aarch64."
        ),
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    # Compiler aliases — all map to the same vendored cosmocc driver.
    "gcc": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "cosmocc").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": "Alias for cosmocc (Cosmopolitan C/C++ compiler)",
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    "cc": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "cosmocc").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": "Alias for cosmocc (Cosmopolitan C/C++ compiler)",
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    "clang": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "cosmocc").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": "Alias for cosmocc (Cosmopolitan C/C++ compiler)",
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    "python3": {
        "binary": str(REPO_ROOT / "bin" / "cosmo" / "python"),
        # fattr/chown: pip must set file mtimes (utime) and ownership when
        # unpacking wheels/sdists during `pip install` — without them package
        # extraction fails with "Operation not permitted" at tarfile.utime.
        # proc/prot_exec: let python spawn subprocesses (needed by the unit
        # test suite's real-subprocess integration tests, and by build tools
        # that shell out to python).
        "promises": "stdio rpath wpath cpath inet dns recvfd fattr chown proc prot_exec",
        "description": (
            "Python 3 interpreter (Cosmopolitan static build). Runs with "
            "-S so PYTHONPATH (a sandbox-local site dir inside the cwd) "
            "is honored; supports pip/network installs. "
            "Use `pip install --user <pkg>` to install into .py-site "
            "(the supported path). The cosmo python has no ensurepip, so "
            "use `python3 -m venv --without-pip <dir>` to create a venv "
            "(pip bootstrap is unsupported)."
        ),
        # Prepend -S: the cosmo python's patched site module wipes PYTHONPATH
        # entries, so we disable it and rely on PYTHONPATH for the sandbox-
        # local site-packages dir.
        "prepend_args": ["-S"],
        # A per-invocation sandbox-local site base dir, created under the
        # cwd and exposed via PYTHONPATH/PYTHONUSERBASE. `pip install --user`
        # installs into <dir>/lib/python3.12/site-packages, and imports pick
        # it up via PYTHONPATH. Keeps packages inside the sandbox workspace
        # rather than unveiling the host's ~/.local.
        "site_dir_name": ".py-site",
        # Extra dirs (relative to cwd) appended to PYTHONPATH so the
        # project's own package is importable without an editable install.
        # .pth files are not processed under -S, so PYTHONPATH is the only
        # channel. Appended AFTER site-packages so user-installed packages
        # take precedence.
        "pythonpath_extra": ["src"],
    },
    "file": {
        "binary": str(REPO_ROOT / "bin" / "cosmo" / "file"),
        "promises": "stdio rpath",
        "description": "Determine file type (Cosmopolitan static build)",
    },
}

BUSYBOX_APPLETS = [
    # text/read
    "cat", "head", "tail", "wc", "sort", "uniq", "nl", "tac", "tee",
    "grep", "ls", "echo", "printf", "test", "expr",
    "cut", "tr", "diff", "cmp", "md5sum", "sha256sum",
    "which", "basename", "dirname", "realpath",
    # file manipulation
    "mkdir", "cp", "mv", "rm", "touch", "chmod",
    # inspection
    "find", "sed", "awk", "stat", "readlink", "df", "du", "uname",
    "id", "date", "env", "seq", "shuf", "xargs", "unzip",
    "true", "false", "sleep",
]


# ---------------------------------------------------------------------------
# Command resolution
# ---------------------------------------------------------------------------


def _detect_venv_root(cmd_name: str, work_dir: Path) -> Optional[Path]:
    """Return the venv root directory if *cmd_name* lives inside a venv
    that is contained within *work_dir*, or ``None``.

    A venv is identified by a ``pyvenv.cfg`` file in the grandparent of
    *cmd_name*'s path (standard venv layout:
    ``<venv_root>/{pyvenv.cfg, bin/python}``).  The venv root must reside
    inside the work tree — a venv outside ``work_dir`` is rejected (returns
    ``None``) to prevent ``site_dir_name``/``PYTHONUSERBASE``/``PYTHONPATH``
    from escaping the sandbox.
    """
    raw = Path(cmd_name)
    candidate = raw if raw.is_absolute() else (work_dir / raw)
    # candidate.parent is the bin/ directory; its parent is the venv root.
    try:
        venv_root = candidate.parent.parent
    except (ValueError, OSError):
        return None
    if not (venv_root / "pyvenv.cfg").is_file():
        return None
    # Containment: the venv root must stay within the work tree.
    try:
        venv_root.resolve().relative_to(work_dir.resolve())
    except ValueError:
        return None
    return venv_root


def _maybe_venv_cfg(
    cmd_name: str, work_dir: Path, resolved_binary: str,
) -> Optional[dict]:
    """If *resolved_binary* is the allowlisted cosmo python AND *cmd_name*
    (resolved against *work_dir*) lives inside a venv (detected by a
    ``pyvenv.cfg`` in the parent of the ``bin/`` directory), return a cfg
    that gives the venv python the ``-S`` + venv-site-packages treatment.

    Returns ``None`` when this is not a venv python.
    """
    python3_binary = COMMANDS["python3"]["binary"]
    if Path(resolved_binary).resolve() != Path(python3_binary).resolve():
        return None

    venv_root = _detect_venv_root(cmd_name, work_dir)
    if venv_root is None:
        return None

    return {
        "binary": resolved_binary,
        "promises": COMMANDS["python3"]["promises"],
        "description": f"Venv python: {cmd_name}",
        # is_local_binary is deliberately omitted so the TOCTOU containment
        # re-check in _build_pipeline_plan is skipped.  The binary is the
        # trusted cosmo python; the venv symlink may resolve outside the
        # work_dir, which would fail _binary_still_contained.
        "prepend_args": ["-S"],
        "site_dir_name": str(venv_root.resolve()),
        "is_venv": True,
        "pythonpath_extra": COMMANDS["python3"].get("pythonpath_extra", []),
    }


def _resolve_venv_fallback(
    cmd_name: str, work_dir: Path,
) -> Optional[dict]:
    """When ``_resolve_local_binary`` fails (containment rejects the resolved
    cosmo python because it lives outside the work tree), check whether
    *cmd_name* still looks like a venv python whose symlink target is the
    cosmo python.  Returns a cfg using the allowlisted cosmo python path
    directly, or ``None``.
    """
    venv_root = _detect_venv_root(cmd_name, work_dir)
    if venv_root is None:
        return None

    # Resolve the final target to verify it really is the cosmo python.
    raw = Path(cmd_name)
    candidate = raw if raw.is_absolute() else (work_dir / raw)
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError):
        return None

    python3_binary = COMMANDS["python3"]["binary"]
    if resolved != Path(python3_binary).resolve():
        return None

    return {
        "binary": python3_binary,
        "promises": COMMANDS["python3"]["promises"],
        "description": f"Venv python: {cmd_name}",
        "prepend_args": ["-S"],
        "site_dir_name": str(venv_root.resolve()),
        "is_venv": True,
        "pythonpath_extra": COMMANDS["python3"].get("pythonpath_extra", []),
    }


def _resolve_command(
    args: list[str],
    work_dir: Optional[Path] = None,
) -> tuple[str, list[str], dict] | tuple[None, str, None]:
    """Resolve and validate a command against the allowlist.

    If the command is not a known allowlisted command, and a working
    directory is provided, tries to resolve the first token as a local
    binary path relative to that directory (e.g. "./scripts/foo" or
    "target/release/bar"). Such a binary is only allowed if it resolves
    to an executable file inside (or below) the working directory.

    Returns (binary_path, final_args, command_config) or (None, error_msg, None).
    """
    if not args:
        return None, "No command provided.", None

    cmd_name = args[0]

    # Check if it's a busybox applet alias
    if cmd_name in BUSYBOX_APPLETS:
        cfg = {
            "binary": str(BUSYBOX_BIN.resolve()),
            "promises": "stdio rpath wpath cpath",
            "description": f"BusyBox {cmd_name}",
        }
        binary = cfg["binary"]
        full_args = [binary, cmd_name] + args[1:]
        return binary, full_args, cfg

    # Reject direct busybox invocation — must use applet aliases
    if cmd_name == "busybox":
        return None, (
            f"Direct 'busybox' invocation not allowed. "
            f"Use individual applets instead: {', '.join(BUSYBOX_APPLETS[:10])}..."
        ), None

    # Check direct commands (git, cargo, ...)
    if cmd_name in COMMANDS:
        cfg = COMMANDS[cmd_name]
        return cfg["binary"], args, cfg

    # Fall back to resolving a local binary under the working directory.
    if work_dir is not None:
        binary = _resolve_local_binary(cmd_name, work_dir)
        if binary is not None:
            # Check for a venv python before assuming a plain local binary.
            cfg = _maybe_venv_cfg(cmd_name, work_dir, binary)
            if cfg is None:
                cfg = {
                    "binary": binary,
                    "promises": "stdio rpath wpath cpath prot_exec",
                    "description": f"Local binary under cwd: {binary}",
                    "is_local_binary": True,
                }
            return binary, args, cfg

        # Venv fallback: _resolve_local_binary failed (containment check
        # rejected the resolved cosmo python outside the work_dir), but
        # the path may still be a valid venv python.
        cfg = _resolve_venv_fallback(cmd_name, work_dir)
        if cfg is not None:
            return cfg["binary"], args, cfg

    return None, f"Command not allowed: {cmd_name}. Use shell_list to see allowed commands.", None


def _resolve_local_binary(cmd_name: str, work_dir: Path) -> Optional[str]:
    """Try to resolve `cmd_name` to an executable file under `work_dir`.

    Accepts a bare name, a relative path (e.g. "scripts/foo" or
    "./target/bar"), or an absolute path, as long as it stays within the
    working tree. Returns the resolved binary path, or None if it is not
    an allowed local executable.
    """
    if cmd_name in (".", ".."):
        return None

    candidate = _contained_path(cmd_name, work_dir)
    if candidate is None:
        return None

    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        return None

    return str(candidate)
