"""Command allowlist and resolution — git helpers, toolchain paths,
COMMANDS table, BUSYBOX_APPLETS, and the command resolver.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .config import (
    BUSYBOX_BIN,
    COSMO_TOOLCHAIN,
    MUSL_LOADER,
    MUSL_RTLIB,
    MUSL_TOOLCHAIN,
    PYTHON_MUSL_INSTALL,
    REPO_ROOT,
)
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
    # 0o600 (NOT 0o666): this file holds the git credential shim, so it is a
    # secret. Force 0600 explicitly, deliberately ignoring the process umask.
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


def _musl_toolchain_paths(work_dir: Optional[Path] = None) -> list[str]:
    """Return the paths that must be unveiled read-execute for the vendored
    musl cross-toolchain to run: the toolchain tree itself (its cross
    compilers read headers/libs/specs from it) and the busybox binary (used
    as a non-preserving `mv` by build tools).
    """
    return [
        str(MUSL_TOOLCHAIN.resolve()),
        str(BUSYBOX_BIN.resolve()),
    ]


def _musl_toolchain_bin() -> str:
    """Return the vendored musl toolchain's bin directory.

    Prepended to PATH for cross-compile commands so the cross tools (and the
    busybox `mv` symlink) resolve ahead of any host equivalents.
    """
    return str((MUSL_TOOLCHAIN / "bin").resolve())


def _python_musl_paths(work_dir: Optional[Path] = None) -> list[str]:
    """Return the paths that must be unveiled read-execute for the vendored
    musl CPython interpreter to run: the entire install tree (binary +
    stdlib .py + lib-dynload .so) and its rtlib (ld-musl loader + libc.so
    that the dynamic linker maps). The interpreter's PT_INTERP and rpath
    point inside this tree, so all of it must be visible to the kernel
    loader.
    """
    return [str(PYTHON_MUSL_INSTALL.resolve())]


def _musl_rtlib_paths(work_dir: Optional[Path] = None) -> list[str]:
    """Return the vendored musl rtlib directory (loader + libc.so) so local
    dynamically-linked musl binaries can be run via loader fallback.

    The loader is at ``MUSL_LOADER`` and ``libc.so`` is alongside it; unveiling
    the entire rtlib directory rx makes both visible to the kernel.
    """
    return [str(MUSL_RTLIB.resolve())]


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
        # flock: cargo locks Cargo.lock + target dir for concurrent builds.
        # fattr: rustc sets file mtimes/perms on build artifacts in target/.
        # proc/prot_exec: cargo spawns rustc/linker subprocesses.
        # inet/dns: cargo may fetch crates from a registry (crates.io) during
        #   `cargo build`/`fetch` when the lockfile is regenerated.
        "promises": "stdio rpath wpath cpath proc prot_exec flock fattr inet dns",
        # Redirect CARGO_HOME into the workspace (<cwd>/.cargo-home) so cargo's
        # registry cache and global config stay inside the sandboxed tree
        # (work_dir is unveiled rwcx, so no extra unveil is needed). Without
        # this, cargo writes to $HOME/.cargo, which is not unveiled, and any
        # build with [dependencies] fails with Permission denied. Mirrors the
        # python3 site_dir_name pattern.
        "cargo_home": ".cargo-home",
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
    "gcc": {
        "binary": str((MUSL_TOOLCHAIN / "bin" / "x86_64-buildroot-linux-musl-gcc.br_real").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": (
            "x86_64 musl cross C compiler (vendored Bootlin toolchain). "
            "Produces statically-linked musl binaries for x86_64. Uses the "
            ".br_real driver directly (the buildroot toolchain-wrapper "
            "symlink breaks under the sandbox because argv[0] is resolved). "
            "Use `cosmocc` for Cosmopolitan APE builds."
        ),
        "extra_unveil_rx": _musl_toolchain_paths,
        "path_prefix": _musl_toolchain_bin,
    },
    "cc": {
        "binary": str((MUSL_TOOLCHAIN / "bin" / "x86_64-buildroot-linux-musl-cc.br_real").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": (
            "x86_64 musl cross C compiler (vendored Bootlin toolchain). "
            "Produces statically-linked musl binaries for x86_64. Uses the "
            ".br_real driver directly (the buildroot toolchain-wrapper "
            "symlink breaks under the sandbox because argv[0] is resolved). "
            "Use `cosmocc` for Cosmopolitan APE builds."
        ),
        "extra_unveil_rx": _musl_toolchain_paths,
        "path_prefix": _musl_toolchain_bin,
    },
    "clang": {
        "binary": str((MUSL_TOOLCHAIN / "bin" / "x86_64-buildroot-linux-musl-gcc.br_real").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": (
            "Maps to the x86_64 musl cross C compiler (vendored Bootlin "
            "toolchain); no clang ships with the musl toolchain, so `clang` "
            "invokes the musl gcc driver to produce statically-linked musl "
            "binaries. Use `cosmocc` for Cosmopolitan APE builds."
        ),
        "extra_unveil_rx": _musl_toolchain_paths,
        "path_prefix": _musl_toolchain_bin,
    },
    "musl-gcc": {
        "binary": str((MUSL_TOOLCHAIN / "bin" / "x86_64-buildroot-linux-musl-gcc.br_real").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": (
            "x86_64 musl cross C compiler (vendored Bootlin toolchain). "
            "Produces statically-linked musl binaries for x86_64. Uses the "
            ".br_real driver directly (the buildroot toolchain-wrapper "
            "symlink breaks under the sandbox because argv[0] is resolved)."
        ),
        "extra_unveil_rx": _musl_toolchain_paths,
        "path_prefix": _musl_toolchain_bin,
    },
    "musl-cc": {
        "binary": str((MUSL_TOOLCHAIN / "bin" / "x86_64-buildroot-linux-musl-cc.br_real").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": "Alias for musl-gcc (x86_64 musl cross C compiler)",
        "extra_unveil_rx": _musl_toolchain_paths,
        "path_prefix": _musl_toolchain_bin,
    },
    "musl-ld": {
        "binary": str((MUSL_TOOLCHAIN / "bin" / "x86_64-buildroot-linux-musl-ld").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": "x86_64 musl cross linker (vendored Bootlin toolchain)",
        "extra_unveil_rx": _musl_toolchain_paths,
        "path_prefix": _musl_toolchain_bin,
    },
    "musl-ar": {
        "binary": str((MUSL_TOOLCHAIN / "bin" / "x86_64-buildroot-linux-musl-ar").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": "x86_64 musl cross archiver (vendored Bootlin toolchain)",
        "extra_unveil_rx": _musl_toolchain_paths,
        "path_prefix": _musl_toolchain_bin,
    },
    "python3": {
        "binary": str(PYTHON_MUSL_INSTALL / "bin" / "python3.12"),
        # fattr/chown: pip must set file mtimes (utime) and ownership when
        # unpacking wheels/sdists during `pip install` — without them package
        # extraction fails with "Operation not permitted" at tarfile.utime.
        # proc/prot_exec: let python spawn subprocesses (needed by the unit
        # test suite's real-subprocess integration tests, and by build tools
        # that shell out to python).
        "promises": "stdio rpath wpath cpath inet dns recvfd fattr chown proc prot_exec",
        "description": (
            "Python 3.12.11 interpreter (vendored musl CPython, dynamically "
            "linked against the staged ld-musl loader under "
            "bin/python-musl/install/lib/rtlib/). Real CPython — supports "
            "ctypes dlopen, .pth files, and standard site.py processing. "
            "Use `pip install --user <pkg>` to install into .py-site (the "
            "supported path)."
        ),
        # No prepend_args: real CPython reads PYTHONPATH through site.py and
        # honors .pth files, so we don't need the cosmo -S bypass anymore.
        # A per-invocation sandbox-local site base dir, created under the
        # cwd and exposed via PYTHONPATH/PYTHONUSERBASE. `pip install --user`
        # installs into <dir>/lib/python3.12/site-packages, and imports pick
        # it up via PYTHONPATH. Keeps packages inside the sandbox workspace
        # rather than unveiling the host's ~/.local.
        "site_dir_name": ".py-site",
        # Extra dirs (relative to cwd) appended to PYTHONPATH so the
        # project's own package is importable without an editable install.
        # Appended AFTER site-packages so user-installed packages take
        # precedence.
        "pythonpath_extra": ["src"],
        # Reveal the entire vendored tree rx so the kernel loader can map
        # ld-musl + libc.so + python3.12 + lib-dynload/*.so + stdlib .py
        # (the interpreter is dynamically linked, unlike the self-contained
        # cosmo APE).
        "extra_unveil_rx": _python_musl_paths,
        # Let python-subprocess git read ~/.gitconfig (config only, never
        # ~/.git-credentials — security invariant).
        "extra_unveil": _git_config_paths(),
        # Set SSL_CERT_FILE so the vendored musl CPython can verify TLS
        # certificates from any cwd (its default CA bundle is under the
        # vendored deps/ tree, which is not unveiled; /etc/ssl is already
        # unveiled read-only by sandbox.c).
        "env": {"SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt"},
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

# Base pledges for busybox applets.  Override per applet below for applets
# that need extra syscalls (fattr for utimensat, proc/prot_exec for spawning).
_BUSYBOX_BASE_PROMISES = "stdio rpath wpath cpath"

# Per-applet promise suffix overrides (appended to _BUSYBOX_BASE_PROMISES).
_BUSYBOX_PROMISE_OVERRIDES: dict[str, str] = {
    "touch": " fattr",
    "xargs": " proc prot_exec",
    "find": " proc prot_exec",
}

# Applets that spawn subprocesses get the user's git config unveiled (read-only,
# never the credential file) so spawned git can read user.name/email. The cred
# store stays out of the unveil of ANY fork/exec-capable parent (security).
_BUSYBOX_UNVEIL_OVERRIDES: dict[str, list[str]] = {
    "xargs": _git_config_paths(),
    "find": _git_config_paths(),
}


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
    """If *resolved_binary* is the allowlisted vendored musl python AND
    *cmd_name* (resolved against *work_dir*) lives inside a venv (detected
    by a ``pyvenv.cfg`` in the parent of the ``bin/`` directory), return a
    cfg that gives the venv python the venv-site-packages treatment.

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
        # trusted vendored musl python; the venv symlink may resolve outside
        # the work_dir, which would fail _binary_still_contained.
        "site_dir_name": str(venv_root.resolve()),
        "is_venv": True,
        "pythonpath_extra": COMMANDS["python3"].get("pythonpath_extra", []),
        "extra_unveil_rx": COMMANDS["python3"].get("extra_unveil_rx"),
        "extra_unveil": COMMANDS["python3"].get("extra_unveil"),
        "env": dict(COMMANDS["python3"].get("env", {})),
    }


def _resolve_venv_fallback(
    cmd_name: str, work_dir: Path,
) -> Optional[dict]:
    """When ``_resolve_local_binary`` fails (containment rejects the resolved
    vendored musl python because it lives outside the work tree), check
    whether *cmd_name* still looks like a venv python whose symlink target
    is the vendored musl python.  Returns a cfg using the allowlisted
    python path directly, or ``None``.
    """
    venv_root = _detect_venv_root(cmd_name, work_dir)
    if venv_root is None:
        return None

    # Resolve the final target to verify it really is the vendored musl python.
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
        "site_dir_name": str(venv_root.resolve()),
        "is_venv": True,
        "pythonpath_extra": COMMANDS["python3"].get("pythonpath_extra", []),
        "extra_unveil_rx": COMMANDS["python3"].get("extra_unveil_rx"),
        "extra_unveil": COMMANDS["python3"].get("extra_unveil"),
        "env": dict(COMMANDS["python3"].get("env", {})),
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
        promises = _BUSYBOX_BASE_PROMISES + _BUSYBOX_PROMISE_OVERRIDES.get(cmd_name, "")
        cfg: dict = {
            "binary": str(BUSYBOX_BIN.resolve()),
            "promises": promises,
            "description": f"BusyBox {cmd_name}",
        }
        if cmd_name in _BUSYBOX_UNVEIL_OVERRIDES:
            cfg["extra_unveil"] = _BUSYBOX_UNVEIL_OVERRIDES[cmd_name]
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
                    "promises": "stdio rpath wpath cpath prot_exec proc",
                    "description": f"Local binary under cwd: {binary}",
                    "is_local_binary": True,
                    "extra_unveil": _git_config_paths(),
                    "extra_unveil_rx": _musl_rtlib_paths,
                    "env": {
                        "SANDBOX_MUSL_LOADER": str(MUSL_LOADER.resolve()),
                        "SANDBOX_MUSL_RTLIB": str(MUSL_RTLIB.resolve()),
                    },
                }
            return binary, args, cfg

        # Venv fallback: _resolve_local_binary failed (containment check
        # rejected the resolved vendored musl python outside the work_dir), but
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
