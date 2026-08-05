#!/usr/bin/env python3
"""Shell Sandbox MCP Server — safe shell command execution via pledge + busybox.

Tools:
  shell_run(command, cwd, timeout) — run a command in a pledge sandbox
  shell_list                          — list allowed commands
"""

import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Redirect dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Redirect:
    """A parsed shell redirect operator extracted from a command segment."""
    fd: int                          # 1 (stdout) or 2 (stderr)
    op: Literal[">", ">>", ">&"]     # truncate, append, dup
    target_path: Optional[str]       # resolved absolute path (None for ">&")
    target_fd: Optional[int]         # source fd for ">&" (1 or 2); else None
    raw_target: Optional[str]        # user-typed target (for messages)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SANDBOX_BIN = REPO_ROOT / "bin" / "sandbox"
SANDBOX_WRAPPER = REPO_ROOT / "bin" / "run-sandbox"
BUSYBOX_BIN = REPO_ROOT / "bin" / "busybox"
COSMO_TOOLCHAIN = REPO_ROOT / "bin" / "cosmo-toolchain"
DEFAULT_ALLOWED_DIRS = [
    str(Path.home() / "projects"),
    "/tmp",
]
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300
MAX_OUTPUT = 1_000_000  # 1 MB

# ---------------------------------------------------------------------------
# Command definitions
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
    """Return the git credential store file under $HOME that must be unveiled
    read-write, so the `store` credential helper can read and update it
    (e.g. for authenticated pushes against GitHub).
    """
    return [
        str((Path.home().resolve() / ".git-credentials").resolve()),
    ]


def _cosmo_toolchain_paths() -> list[str]:
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
        "extra_unveil": _git_config_paths,
        "extra_unveil_rw": _git_credential_paths,
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
        "promises": "stdio rpath wpath cpath inet dns recvfd",
        "description": (
            "Python 3 interpreter (Cosmopolitan static build). Runs with "
            "-S so PYTHONPATH (a sandbox-local site dir inside the cwd) "
            "is honored; supports pip/network installs."
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
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "shell-sandbox",
    instructions="Run shell commands safely via pledge + busybox sandbox",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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
            cfg = {
                "binary": binary,
                "promises": "stdio rpath wpath cpath prot_exec",
                "description": f"Local binary under cwd: {binary}",
                "is_local_binary": True,
            }
            return binary, args, cfg

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


def _contained_path(cmd_name: str, work_dir: Path) -> Optional[Path]:
    """Resolve `cmd_name` relative to `work_dir`, returning the resolved
    path only if it stays within the working tree; else None.

    Uses `resolve()` (which follows symlinks), so a symlink inside the cwd
    pointing outside it is caught by the containment check.
    """
    try:
        raw = Path(cmd_name)
        candidate = raw if raw.is_absolute() else (work_dir / raw)
        candidate = candidate.resolve()
        # Must stay inside the working directory (or below it).
        candidate.relative_to(work_dir)
    except (ValueError, OSError):
        return None
    return candidate


def _binary_still_contained(binary: str, work_dir: Path) -> bool:
    """Re-verify a previously-resolved local binary path is still an
    executable file contained within the work dir, right before exec.

    Narrows the TOCTOU window between initial resolution and exec: if a
    directory component was swapped for a symlink escaping the tree in the
    meantime, the re-resolve + containment check catches it. Called with the
    path actually passed to the sandbox.
    """
    candidate = _contained_path(binary, work_dir)
    if candidate is None:
        return False
    return candidate.is_file() and os.access(candidate, os.X_OK)



def _validate_cwd(resolved: Path, raw: str) -> Optional[str]:
    """Validate a working directory. `resolved` must be an already-resolved
    absolute path; `raw` is the user's original input for error messages.

    Returns an error string, or None if valid.
    """
    if not resolved.is_dir():
        return f"Directory not found: {raw}"

    # Must be within an allowed tree or a subdirectory thereof
    for allowed in DEFAULT_ALLOWED_DIRS:
        allowed_path = Path(allowed).expanduser().resolve()
        try:
            resolved.relative_to(allowed_path)
            return None
        except ValueError:
            continue

    return f"Directory not in allowed paths: {raw}"


# ---------------------------------------------------------------------------
# Command chaining
# ---------------------------------------------------------------------------

def _split_command(command: str) -> list[tuple[Optional[str], list[str], bool]]:
    """Split a command string into a chain of pipe-connected pipelines.

    A pipeline is a list of segments joined by a single top-level `|` (pipe),
    where each segment's stdout feeds the next segment's stdin. Pipelines are
    joined by top-level `;`, `&&`, `||`, or `&` operators. Operators nested
    inside single or double quotes are preserved as literal text of the segment
    (they are not shell operators).

    Returns a list of ``(operator, pipeline, backgrounded)`` triples where
    ``operator`` is None for the first pipeline and ``';'``, ``'&&'``, or
    ``'||'`` for each subsequent one (never ``'&'`` — bare ``&`` acts like
    ``;`` semantically but sets ``backgrounded=True`` on the preceding
    pipeline). Each ``pipeline`` is a list of pipe-separated segment strings.
    Leading/trailing whitespace is stripped from each segment; empty segments
    are dropped.
    """
    pipelines: list[tuple[Optional[str], list[str], bool]] = []
    current_pipeline: list[str] = []
    current_seg: list[str] = []
    i, n = 0, len(command)
    quote: Optional[str] = None
    prev_op: Optional[str] = None

    def flush_segment() -> None:
        text = "".join(current_seg).strip()
        if text:
            current_pipeline.append(text)
        current_seg.clear()

    def flush_pipeline(backgrounded: bool = False) -> None:
        nonlocal prev_op
        flush_segment()
        if current_pipeline:
            pipelines.append((prev_op, current_pipeline[:], backgrounded))
        del current_pipeline[:]
        prev_op = None

    while i < n:
        c = command[i]
        if quote is not None:
            current_seg.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            current_seg.append(c)
            i += 1
            continue
        if c == ";":
            flush_pipeline()
            prev_op = ";"
            i += 1
            continue
        if c == "&" and i + 1 < n and command[i + 1] == "&":
            flush_pipeline()
            prev_op = "&&"
            i += 2
            continue
        if c == "&" and i > 0 and command[i - 1] == ">" and i + 1 < n and command[i + 1].isdigit():
            # fd-dup redirect like `2>&1` / `1>&2`: the `&` here is part of a
            # redirect operator, not a backgrounding operator. Leave it in the
            # current segment so _extract_redirects can parse it.
            current_seg.append(c)
            i += 1
            continue
        if c == "&":
            # Bare '&' backgrounding: flush the current pipeline marked as
            # backgrounded and reset prev_op (like ';' — next pipeline always
            # runs).
            flush_pipeline(backgrounded=True)
            i += 1
            continue
        if c == "|" and i + 1 < n and command[i + 1] == "|":
            flush_pipeline()
            prev_op = "||"
            i += 2
            continue
        if c == "|":
            # single pipe — end the current segment within the pipeline
            flush_segment()
            i += 1
            continue
        current_seg.append(c)
        i += 1

    flush_pipeline()
    return pipelines


# ---------------------------------------------------------------------------
# Redirect parsing
# ---------------------------------------------------------------------------


def _extract_redirects(segment: str) -> tuple[list[str], list[Redirect], Optional[str]]:
    """Tokenize a command segment, extracting redirect operators.

    Single-pass char-by-char tokenizer that:
    - Splits on unquoted whitespace
    - Strips quote characters (``'`` and ``"``) from words (POSIX-style)
    - Recognizes unquoted redirect operators only at word boundaries:
      ``> file``, ``>> file``, ``2> file``, ``2>> file``, ``2>&1``, ``1>&2``
    - Glued forms like ``foo>bar`` are treated as literal words (a deliberate
      divergence from POSIX; this avoids ambiguous parsing with commands that
      embed ``>`` in their arguments).

    Returns ``(args, redirects, None)`` on success, or
    ``([], [], error_msg)`` on error.
    ``args`` is the tokenized command with quotes stripped and redirect
    operators (plus their targets) removed.
    """
    args: list[str] = []
    redirects: list[Redirect] = []
    i = 0
    n = len(segment)

    def _read_word() -> Optional[str]:
        """Read a shell word starting at *i*, stripping quotes.
        Stops at unquoted whitespace.  Returns ``None`` on unbalanced quotes.
        """
        nonlocal i
        chars: list[str] = []
        quote: Optional[str] = None
        while i < n:
            c = segment[i]
            if quote is not None:
                if c == quote:
                    quote = None
                else:
                    chars.append(c)
                i += 1
            elif c in ("'", '"'):
                quote = c
                i += 1
            elif c in (' ', '\t'):
                break
            else:
                chars.append(c)
                i += 1
        if quote is not None:
            return None
        return ''.join(chars)

    def _read_redirect_target() -> Optional[str]:
        """Read the target of a ``>`` / ``>>`` operator.
        *i* is positioned right after the operator characters.
        The target may be glued (``>out.txt``) or space-separated
        (``> out.txt``).  Returns the target word, or ``None`` if missing.
        """
        nonlocal i
        if i < n and segment[i] not in (' ', '\t'):
            return _read_word()
        while i < n and segment[i] in (' ', '\t'):
            i += 1
        if i >= n:
            return None
        return _read_word()

    while i < n:
        while i < n and segment[i] in (' ', '\t'):
            i += 1
        if i >= n:
            break

        rem = segment[i:]

        # -- longest-match-first: 4-char patterns --
        if rem.startswith('2>&1'):
            i += 4
            redirects.append(Redirect(fd=2, op='>&', target_path=None, target_fd=1, raw_target='1'))
            continue
        if rem.startswith('1>&2'):
            i += 4
            redirects.append(Redirect(fd=1, op='>&', target_path=None, target_fd=2, raw_target='2'))
            continue

        # -- 3-char patterns --
        if rem.startswith('2>>'):
            i += 3
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=2, op='>>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('1>>'):
            i += 3
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=1, op='>>', target_path=None, target_fd=None, raw_target=t))
            continue

        # -- 2-char patterns --
        if rem.startswith('>>'):
            i += 2
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=1, op='>>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('2>'):
            i += 2
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=2, op='>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('1>'):
            i += 2
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('<<'):
            return [], [], "Input redirects are not supported"

        # -- 1-char patterns --
        if rem.startswith('>'):
            i += 1
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('<'):
            return [], [], "Input redirects are not supported"

        # -- digit + >  patterns that aren't the allowed 1> / 2> → error --
        if len(rem) >= 2 and rem[0].isdigit() and rem[1] == '>':
            fd = int(rem[0])
            return [], [], f"Redirects only support fds 1 and 2 (got {fd})"

        # -- regular word --
        w = _read_word()
        if w is None:
            return [], [], "Unbalanced quotes in command"
        if w:
            args.append(w)

    return args, redirects, None


def _validate_redirect_paths(
    redirects: list[Redirect],
    work_dir: Path,
) -> tuple[list[Redirect], Optional[str]]:
    """Resolve and validate the paths in a list of redirects.

    For each ``>`` / ``>>`` redirect, resolves ``raw_target`` against
    ``work_dir`` via :func:`_contained_path`.  Returns the updated list
    (with ``target_path`` populated) or ``([], error_msg)`` if a target
    escapes the working directory.  ``>&`` redirects pass through unchanged.
    """
    from dataclasses import replace

    validated: list[Redirect] = []
    for r in redirects:
        if r.op in (">", ">>"):
            cand = _contained_path(r.raw_target, work_dir)
            if cand is None:
                return [], f"Redirect target escapes working directory: {r.raw_target}"
            validated.append(replace(r, target_path=str(cand)))
        else:
            validated.append(r)
    return validated, None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _build_invocation(
    command: str,
    work_dir: Path,
) -> tuple[Optional[str], Optional[list[str]], Optional[dict], Optional[dict], list[Redirect]]:
    """Parse, resolve, and build the sandbox invocation for one segment.

    Returns ``(binary, sandbox_args, env, cfg, redirects)`` on success.
    ``env`` is ``None`` when no env overrides are needed.
    On failure, returns a tuple whose first element is the error message
    (string) and whose remaining elements are ``None`` / empty:
    ``(error_msg, None, None, None, [])``.
    An empty command returns ``(None, None, None, None, [])``.
    """
    args, raw_redirects, parse_err = _extract_redirects(command)
    if parse_err is not None:
        return parse_err, None, None, None, []

    if not args:
        return None, None, None, None, []

    # Validate redirect paths against the working directory
    redirects, path_err = _validate_redirect_paths(raw_redirects, work_dir)
    if path_err is not None:
        return path_err, None, None, None, []

    # Resolve and validate against the allowlist (or a local binary under cwd)
    binary, final_args, cfg = _resolve_command(args, work_dir)
    if binary is None:
        return final_args, None, None, None, []  # error message

    # Build sandbox invocation via the exec wrapper (bypasses posix_spawn APE issue)
    # Prepend per-command args (e.g. python3's "-S") before the user's args.
    prepend = cfg.get("prepend_args", [])
    sandbox_args = [
        str(SANDBOX_WRAPPER.resolve()),
        cfg["promises"],
        str(work_dir),  # unveil directory
        "--",
        binary,
    ] + prepend + final_args[1:]

    # Extra unveil paths via env vars the sandbox honors:
    #   SANDBOX_UNVEIL_R  — read-only (e.g. git config dotfiles under $HOME)
    #   SANDBOX_UNVEIL_RW — read-write (optional)
    env: Optional[dict] = None
    unveil_env: dict[str, str] = {}
    extra_unveil = cfg.get("extra_unveil")
    if extra_unveil:
        paths = extra_unveil() if callable(extra_unveil) else extra_unveil
        if paths:
            unveil_env["SANDBOX_UNVEIL_R"] = ":".join(paths)

    extra_unveil_rw = cfg.get("extra_unveil_rw")
    if extra_unveil_rw:
        paths = extra_unveil_rw() if callable(extra_unveil_rw) else extra_unveil_rw
        if paths:
            unveil_env["SANDBOX_UNVEIL_RW"] = ":".join(paths)

    # Extra read-execute unveil paths (e.g. a vendored compiler toolchain and
    # the Cosmopolitan APE loader) so build tools can exec their subprocesses.
    extra_unveil_rx = cfg.get("extra_unveil_rx")
    if extra_unveil_rx:
        paths = extra_unveil_rx() if callable(extra_unveil_rx) else extra_unveil_rx
        if paths:
            unveil_env["SANDBOX_UNVEIL_RX"] = ":".join(paths)

    # Sandbox-local python site dir: create <cwd>/.py-site, expose the base
    # via PYTHONUSERBASE (so `pip install --user` lands inside the sandbox)
    # and the site-packages via PYTHONPATH (so imports resolve).
    site_dir_name = cfg.get("site_dir_name")
    if site_dir_name:
        site_base = work_dir / site_dir_name
        try:
            site_base.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return f"Error creating python site dir {site_base}: {e}", None, None, None, []
        site_packages = site_base / "lib" / "python3.12" / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        unveil_env["PYTHONUSERBASE"] = str(site_base)
        unveil_env["PYTHONPATH"] = str(site_packages)

    if unveil_env:
        env = {**os.environ, **unveil_env}

    # Optionally prepend a directory to PATH (e.g. so build commands resolve a
    # busybox `mv` from the vendored toolchain instead of GNU /usr/bin/mv).
    path_prefix = cfg.get("path_prefix")
    if path_prefix:
        prefix = path_prefix() if callable(path_prefix) else path_prefix
        if prefix:
            base = env if env is not None else dict(os.environ)
            cur = base.get("PATH", "")
            base["PATH"] = f"{prefix}:{cur}" if cur else prefix
            env = base

    return binary, sandbox_args, env, cfg, redirects


def _resolve_fd_targets(
    redirects: list[Redirect],
    default_stdout,
    default_stderr,
) -> tuple:
    """Apply redirects in order (last-wins per fd) and return fd targets.

    Returns ``(stdout_target, stderr_target, files_to_close, report_lines,
    shared_pipe_read_fd)`` where ``shared_pipe_read_fd`` is ``None`` unless
    a ``1>&2`` redirect forced creation of a shared pipe (when stderr is
    ``subprocess.PIPE``).
    """
    stdout_target = default_stdout
    stderr_target = default_stderr
    files_to_close: list = []
    report_lines: list[str] = []
    shared_pipe_read_fd = None

    for r in redirects:
        if r.op in (">", ">>"):
            mode = "wb" if r.op == ">" else "ab"
            fh = open(r.target_path, mode)
            files_to_close.append(fh)
            if r.fd == 1:
                stdout_target = fh
                arrow = "->" if r.op == ">" else "->>"
                report_lines.append(f"[stdout {arrow} {r.raw_target}]")
            else:  # fd == 2
                stderr_target = fh
                arrow = "->" if r.op == ">" else "->>"
                report_lines.append(f"[stderr {arrow} {r.raw_target}]")
        elif r.op == ">&":
            if r.fd == 2 and r.target_fd == 1:  # 2>&1
                stderr_target = subprocess.STDOUT
                report_lines.append("[stderr -> stdout]")
            elif r.fd == 1 and r.target_fd == 2:  # 1>&2
                if isinstance(stderr_target, int) and stderr_target == subprocess.PIPE:
                    # Create a shared pipe so both stdout and stderr write to
                    # the same fd; parent reads from the read end.
                    rfd, wfd = os.pipe()
                    shared_pipe_read_fd = rfd
                    stdout_target = wfd
                    stderr_target = wfd
                    files_to_close.append(wfd)
                else:
                    # stderr is a real file handle (or STDOUT sentinel);
                    # just point stdout at the same target.
                    stdout_target = stderr_target
                report_lines.append("[stdout -> stderr]")

    return stdout_target, stderr_target, files_to_close, report_lines, shared_pipe_read_fd


def _run_segment(command: str, work_dir: Path, timeout: int) -> tuple[int, str]:
    """Run a single operator-free command segment in the sandbox.

    Returns ``(returncode, output_string)``. ``returncode`` is 0 on success and
    non-zero on failure, an error, or an invalid/denied command, so callers can
    apply ``&&``/``||`` short-circuit semantics. ``output_string`` is the
    formatted output, or ``""`` when the segment produced nothing to report.
    """
    binary, sandbox_args, env, cfg, redirects = _build_invocation(command, work_dir)
    if sandbox_args is None:
        if binary is None:
            return 0, ""  # empty command
        return 1, binary  # error message

    # Narrow the TOCTOU window for local binaries: re-verify the resolved
    # path is still an executable contained within the work dir right before
    # exec, in case a directory component was swapped in the meantime.
    if cfg.get("is_local_binary") and not _binary_still_contained(binary, work_dir):
        return 1, f"Local binary no longer valid inside working directory: {binary}"

    stdout_t, stderr_t, to_close, report, shared_read_fd = _resolve_fd_targets(
        redirects, subprocess.PIPE, subprocess.PIPE,
    )

    try:
        result = subprocess.run(
            sandbox_args,
            stdout=stdout_t,
            stderr=stderr_t,
            timeout=timeout,
            cwd=str(work_dir),
            env=env,
        )

        # When stdout / stderr are file handles or fds, result.stdout/stderr
        # are None; treat as empty bytes.
        stdout_bytes = result.stdout if result.stdout is not None else b""
        stderr_bytes = result.stderr if result.stderr is not None else b""

        # If a shared pipe was created by 1>&2, read the combined output from it.
        if shared_read_fd is not None:
            try:
                combined = os.read(shared_read_fd, MAX_OUTPUT + 1)
            finally:
                try:
                    os.close(shared_read_fd)
                except OSError:
                    pass
            stdout_bytes = combined
            stderr_bytes = b""

        stdout = stdout_bytes[:MAX_OUTPUT].decode("utf-8", errors="replace")
        stderr_out = stderr_bytes[:MAX_OUTPUT].decode("utf-8", errors="replace")

        output = []
        if result.returncode != 0:
            output.append(f"Exit code: {result.returncode}")
        if stdout:
            output.append(stdout.rstrip())
        if stderr_out:
            output.append(f"[stderr]\n{stderr_out.rstrip()}")
        if report:
            output.extend(report)
        if len(stdout_bytes) > MAX_OUTPUT:
            output.append(f"\n[output truncated at {MAX_OUTPUT} bytes]")
        if len(stderr_bytes) > MAX_OUTPUT:
            output.append(f"\n[stderr truncated at {MAX_OUTPUT} bytes]")

        return result.returncode, "\n".join(output)

    except subprocess.TimeoutExpired:
        return 1, f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return 1, f"Sandbox binary not found: {SANDBOX_BIN}"
    except OSError as e:
        return 1, f"Error running command: {e}"
    finally:
        for fh in to_close:
            try:
                fh.close()
            except OSError:
                pass


def _run_pipeline(
    segments: list[str],
    work_dir: Path,
    timeout: int,
) -> tuple[int, str]:
    """Run a pipe-connected sequence of segments concurrently in the sandbox.

    Each segment's stdout is connected to the next segment's stdin, so data
    flows through the pipeline as in a real shell pipe. Every segment is still
    run through its own pledge sandbox and checked against the allowlist
    independently.

    Returns ``(returncode, output_string)``. ``returncode`` is the exit code of
    the *last* segment (shell default; no pipefail). Intermediate segments'
    stdout is consumed by the next stage; their stderr, plus the last stage's
    stdout and stderr, are surfaced in ``output_string``.
    """
    invocations: list[tuple[list[str], Optional[dict], list[Redirect]]] = []
    for seg in segments:
        binary, sandbox_args, env, cfg, redirects = _build_invocation(seg, work_dir)
        if sandbox_args is None:
            if binary is None:
                continue  # empty segment inside a pipeline
            return 1, binary  # error message
        if cfg.get("is_local_binary") and not _binary_still_contained(binary, work_dir):
            return 1, f"Local binary no longer valid inside working directory: {binary}"
        invocations.append((sandbox_args, env, redirects))

    if not invocations:
        return 0, ""

    # Reject stdout redirects on intermediate pipe stages.
    for i, (_sa, _env, redirects) in enumerate(invocations[:-1]):
        for r in redirects:
            if r.fd == 1:
                return 1, (
                    f"Cannot redirect stdout of intermediate pipe stage: "
                    f"{segments[i]}"
                )

    # Resolve fd targets per stage.
    stdout_targets: list = []
    stderr_targets: list = []
    all_to_close: list = []
    all_report: list[list[str]] = []
    last_shared_read_fd = None
    for i, (_sa, _env, redirects) in enumerate(invocations):
        is_last = i == len(invocations) - 1
        st, et, tc, rpt, srf = _resolve_fd_targets(
            redirects, subprocess.PIPE, subprocess.PIPE,
        )
        stdout_targets.append(st)
        stderr_targets.append(et)
        all_to_close.extend(tc)
        all_report.append(rpt)
        if is_last:
            last_shared_read_fd = srf

    # Launch every stage, chaining each one's stdout into the next one's stdin.
    procs: list[subprocess.Popen] = []
    prev: Optional[subprocess.Popen] = None
    for i, (sandbox_args, env, _redirects) in enumerate(invocations):
        p = subprocess.Popen(
            sandbox_args,
            stdin=prev.stdout if prev is not None else None,
            stdout=stdout_targets[i],
            stderr=stderr_targets[i],
            cwd=str(work_dir),
            env=env,
        )
        if prev is not None:
            prev.stdout.close()  # parent no longer holds the read end
        procs.append(p)
        prev = p

    # Drain the stderr of every stage but the last on a thread, so a chatty
    # producer can't deadlock on a full stderr pipe. Skip stages whose stderr
    # is not PIPE (file handle or STDOUT => p.stderr is None).
    last = procs[-1]
    stderr_bufs: dict[int, bytes] = {}
    bufs_lock = threading.Lock()

    def _drain_stderr(i: int, p: subprocess.Popen) -> None:
        if p.stderr is None:
            return
        data = p.stderr.read()
        p.stderr.close()  # intermediate pipes aren't closed by communicate()
        with bufs_lock:
            stderr_bufs[i] = data

    threads: list[threading.Thread] = []
    for i, p in enumerate(procs[:-1]):
        if p.stderr is not None:
            t = threading.Thread(target=_drain_stderr, args=(i, p), daemon=True)
            t.start()
            threads.append(t)

    timed_out = False
    try:
        stdout_bytes, last_stderr = last.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True

    # Reap every process that is still running. This is essential: the last
    # stage may finish while an upstream stage (e.g. a chatty producer) keeps
    # running, and we must close its stderr write-end so its drain thread gets
    # EOF and completes — otherwise the thread stays alive while we read
    # `stderr_bufs` below (a race), and the child becomes a zombie.
    for p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass
    for p in procs:
        p.wait()

    # Now that every process is dead and its pipe fds are closed, the drain
    # threads are guaranteed to finish (their read() returns EOF).
    for t in threads:
        t.join(timeout=1)

    # Close any file handles opened for redirects.
    for fh in all_to_close:
        try:
            fh.close()
        except OSError:
            pass
    if last_shared_read_fd is not None:
        try:
            os.close(last_shared_read_fd)
        except OSError:
            pass

    if timed_out:
        # communicate() timed out and left the last stage's pipes open; close
        # them so we don't leak fds or trip ResourceWarning.
        if last.stdout is not None:
            try:
                last.stdout.close()
            except OSError:
                pass
        if last.stderr is not None:
            try:
                last.stderr.close()
            except OSError:
                pass
        return 1, f"Pipeline timed out after {timeout}s"

    rc = last.returncode
    with bufs_lock:
        intermediate_err = b"\n".join(stderr_bufs.values())
    combined_err = (intermediate_err + b"\n" + (last_stderr or b"")).strip()

    # Handle last-stage stdout: may be None when redirected to a file.
    if stdout_bytes is None:
        stdout_bytes = b""

    stdout = stdout_bytes[:MAX_OUTPUT].decode("utf-8", errors="replace")
    stderr_out = combined_err[:MAX_OUTPUT].decode("utf-8", errors="replace")

    output = []
    if rc != 0:
        output.append(f"Exit code: {rc}")
    if stdout:
        output.append(stdout.rstrip())
    if stderr_out:
        output.append(f"[stderr]\n{stderr_out.rstrip()}")

    # Per-stage report lines (prefixed for intermediate stages).
    for i, rpt in enumerate(all_report):
        if not rpt:
            continue
        if i == len(all_report) - 1:
            output.extend(rpt)
        else:
            for line in rpt:
                output.append(f"[stage {i + 1} {line}]")

    if len(stdout_bytes) > MAX_OUTPUT:
        output.append(f"\n[output truncated at {MAX_OUTPUT} bytes]")
    if len(combined_err) > MAX_OUTPUT:
        output.append(f"\n[stderr truncated at {MAX_OUTPUT} bytes]")

    return rc, "\n".join(output)


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------

_reaper_lock = threading.Lock()
_reaper_started = False


def _start_reaper() -> None:
    """Start a daemon thread that reaps zombie background children."""
    global _reaper_started
    with _reaper_lock:
        if _reaper_started:
            return
        _reaper_started = True

    def _reap() -> None:
        while True:
            try:
                while True:
                    try:
                        pid, _ = os.waitpid(-1, os.WNOHANG)
                        if pid == 0:
                            break
                    except ChildProcessError:
                        break
            except Exception:
                pass
            time.sleep(5)

    t = threading.Thread(target=_reap, daemon=True)
    t.start()


def _run_background(
    segments: list[str],
    work_dir: Path,
) -> tuple[int, str]:
    """Launch a pipe-connected pipeline in the background and return immediately.

    Each segment is built through ``_build_invocation`` (same allowlist/sandbox
    checks as ``_run_pipeline``).  Intermediate stages' stdout feeds the next
    stage's stdin exactly as in a foreground pipeline; all stderr, plus the
    last stage's stdout, are redirected to a timestamped log file under
    ``work_dir`` so the parent never blocks on pipe buffers.

    Returns ``(0, message)`` with the PID and log path.
    """
    invocations: list[tuple[list[str], Optional[dict], list[Redirect]]] = []
    for seg in segments:
        binary, sandbox_args, env, cfg, redirects = _build_invocation(seg, work_dir)
        if sandbox_args is None:
            if binary is None:
                continue  # empty segment inside a pipeline
            return 1, binary  # error message
        if cfg.get("is_local_binary") and not _binary_still_contained(binary, work_dir):
            return 1, f"Local binary no longer valid inside working directory: {binary}"
        invocations.append((sandbox_args, env, redirects))

    if not invocations:
        return 0, ""

    # Reject stdout redirects on intermediate pipe stages.
    for i, (_sa, _env, redirects) in enumerate(invocations[:-1]):
        for r in redirects:
            if r.fd == 1:
                return 1, (
                    f"Cannot redirect stdout of intermediate pipe stage: "
                    f"{segments[i]}"
                )

    log_path = work_dir / f".bg-{int(time.time() * 1000)}.log"
    log_fh = open(str(log_path), "wb")

    # Resolve fd targets per stage; last stage defaults to the log file.
    stdout_targets: list = []
    stderr_targets: list = []
    all_to_close: list = [log_fh]
    all_report: list[list[str]] = []
    for i, (_sa, _env, redirects) in enumerate(invocations):
        is_last = i == len(invocations) - 1
        def_stdout = log_fh if is_last else subprocess.PIPE
        def_stderr = log_fh if is_last else subprocess.PIPE
        st, et, tc, rpt, _srf = _resolve_fd_targets(redirects, def_stdout, def_stderr)
        stdout_targets.append(st)
        stderr_targets.append(et)
        all_to_close.extend(tc)
        all_report.append(rpt)

    procs: list[subprocess.Popen] = []
    prev: Optional[subprocess.Popen] = None
    for i, (sandbox_args, env, _redirects) in enumerate(invocations):
        p = subprocess.Popen(
            sandbox_args,
            stdin=prev.stdout if prev is not None else None,
            stdout=stdout_targets[i],
            stderr=stderr_targets[i],
            cwd=str(work_dir),
            env=env,
            start_new_session=True,
        )
        if prev is not None:
            prev.stdout.close()
        procs.append(p)
        prev = p

    # Parent releases its handles; children hold their own copies.
    for fh in all_to_close:
        try:
            fh.close()
        except OSError:
            pass

    # Build the message with report details.
    msg_parts = [f"Backgrounded PID {procs[0].pid}; output -> {log_path}"]
    # Collect all unique report lines across stages.
    seen: set[str] = set()
    for rpt in all_report:
        for line in rpt:
            if line not in seen:
                msg_parts.append(line)
                seen.add(line)

    _start_reaper()
    return 0, "\n".join(msg_parts)


@mcp.tool()
def shell_run(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Run a shell command in a pledge sandbox.

    Commands are executed via the pledge-wrapped sandbox binary with
    restricted system call promises. Only allowlisted commands can run.

    Pipes (`|`) are supported: the command string is split on top-level `|`
    into pipeline stages, and each stage's stdout is fed to the next stage's
    stdin, exactly as in a real shell pipe. Pipelines can be chained with `;`,
    `&&`, and `||`; every stage/segment is run through its own sandbox and
    checked against the allowlist independently. `&&` runs its pipeline only if
    the previous one succeeded; `||` only if it failed; `;` always runs.

    Backgrounding (`&`) is supported: a bare `&` terminates the current
    pipeline, launches it in the background (detached via a new session), and
    returns immediately with the PID and a log path. The next pipeline always
    runs (like `;`). `&&`/`||` short-circuit is still honoured before a
    backgrounded pipeline is launched.

    Redirects (``>``, ``>>``, ``2>``, ``2>>``, ``2>&1``, ``1>&2``) are supported
    per command segment. Redirect targets must stay inside the working
    directory. Quoted operators (``echo ">"``) are treated as literal
    arguments, not redirects. Only output redirects are supported; input
    redirects (``<``, ``<<``) are rejected. Redirecting stdout of an
    intermediate pipe stage is rejected (stderr redirects and ``2>&1`` are
    allowed on intermediate stages).

    Args:
        command: The command to run (e.g., "git status", "ls | grep foo",
            "cd build && make test")
        cwd:     Working directory (must be within allowed paths)
        timeout: Timeout in seconds (default 30, max 300)
    """
    timeout = min(timeout, MAX_TIMEOUT)

    # Validate cwd first — resolve once, use for validation, resolution and
    # execution. Required before resolving so local binaries can be located.
    raw_cwd = cwd or "."
    try:
        work_dir = Path(raw_cwd).expanduser().resolve()
    except Exception:
        return f"Invalid path: {raw_cwd}"

    err = _validate_cwd(work_dir, raw_cwd)
    if err:
        return err

    # Split into allowlist-checked pipelines on ; / && / || / &, with each
    # pipeline being a list of `|`-separated stages.
    try:
        pipelines = _split_command(command)
    except ValueError as e:
        return str(e)
    if not pipelines:
        return "Empty command."

    # Single-command fast path — preserves the exact prior behaviour.
    if len(pipelines) == 1 and pipelines[0][0] is None and len(pipelines[0][1]) == 1:
        if pipelines[0][2]:
            _rc, out = _run_background(pipelines[0][1], work_dir)
            return out if out else "(no output)"
        _rc, out = _run_segment(pipelines[0][1][0], work_dir, timeout)
        return out if out else "(no output)"

    # Multi-pipeline chain: run each pipeline through the sandbox, applying
    # `&&` / `||` short-circuit semantics based on the previous pipeline's exit
    # code. Pipelines are independent processes (no shared shell variables), but
    # they share the same cwd, so file-based state persists across stages.
    outputs: list[str] = []
    prev_rc = 0
    ran_any = False

    for op, stages, backgrounded in pipelines:
        joined = " | ".join(stages)
        if op == "&&" and ran_any and prev_rc != 0:
            outputs.append(f"(skipped: previous command exited {prev_rc}) — {joined}")
            continue
        if op == "||" and ran_any and prev_rc == 0:
            outputs.append("(skipped: previous command succeeded) — " + joined)
            continue

        if backgrounded:
            rc, out = _run_background(stages, work_dir)
            ran_any = True
            # Leave prev_rc unchanged — backgrounded exit code is unknown.
        elif len(stages) == 1:
            rc, out = _run_segment(stages[0], work_dir, timeout)
            prev_rc = rc
            ran_any = True
        else:
            rc, out = _run_pipeline(stages, work_dir, timeout)
            prev_rc = rc
            ran_any = True
        if out:
            outputs.append(out)

    if not outputs:
        return "(no output)"

    return "\n".join(outputs)


@mcp.tool()
def shell_list() -> str:
    """List all allowed commands and their descriptions."""
    lines = ["Allowed commands:"]
    for name, cfg in COMMANDS.items():
        lines.append(f"  {name}: {cfg['description']}")
        lines.append(f"    Promises: {cfg['promises']}")
    lines.append(f"  busybox applets ({len(BUSYBOX_APPLETS)}): {', '.join(BUSYBOX_APPLETS)}")
    lines.append(f"    Binary: {BUSYBOX_BIN.resolve()}")
    lines.append(f"    Promises: stdio rpath wpath cpath")
    lines.append("")
    lines.append("Local binaries: any executable file inside the working")
    lines.append("    directory (or below) can be run by name or path, e.g.")
    lines.append("    './scripts/foo' or 'target/release/bar'. Paths escaping")
    lines.append("    the working directory are rejected.")
    lines.append("")
    lines.append("Command chaining: ';', '&&', '||', and '&' are supported. Each")
    lines.append("    pipeline is run through the sandbox and checked against the")
    lines.append("    allowlist independently. '&&' runs only after success, '||'")
    lines.append("    only after failure, ';' always. '&' backgrounds a pipeline")
    lines.append("    and returns immediately with the PID and log path. Pipes ('|')")
    lines.append("    are supported: each stage's stdout feeds the next stage's stdin.")
    lines.append("    Redirects (>, >>, 2>, 2>>, 2>&1, 1>&2) are supported per segment;")
    lines.append("    targets must stay inside the working directory. Quoted operators")
    lines.append("    are not treated as redirects.")
    lines.append("")
    lines.append(f"Sandbox binary: {SANDBOX_BIN.resolve()}")
    lines.append(f"Allowed directories: {', '.join(DEFAULT_ALLOWED_DIRS)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the MCP server with stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
