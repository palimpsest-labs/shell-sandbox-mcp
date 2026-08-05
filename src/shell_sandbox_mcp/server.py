#!/usr/bin/env python3
"""Shell Sandbox MCP Server — safe shell command execution via pledge + busybox.

Tools:
  shell_run(command, cwd, timeout) — run a command in a pledge sandbox
  shell_list                          — list allowed commands
"""

import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

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

def _split_command(command: str) -> list[tuple[Optional[str], str]]:
    """Split a command string on top-level `;`, `&&`, and `||` operators.

    Operators nested inside single or double quotes are preserved as literal
    text of the segment (they are not shell operators). Returns a list of
    ``(operator, segment)`` pairs where ``operator`` is None for the first
    segment and ``';'``, ``'&&'``, or ``'||'`` for each subsequent one.
    Leading/trailing whitespace is stripped from each segment; empty segments
    are dropped.
    """
    segments: list[tuple[Optional[str], str]] = []
    i, n = 0, len(command)
    current: list[str] = []
    quote: Optional[str] = None
    prev_op: Optional[str] = None

    def flush() -> None:
        nonlocal prev_op
        text = "".join(current).strip()
        if text:
            segments.append((prev_op, text))
        current.clear()
        prev_op = None

    while i < n:
        c = command[i]
        if quote is not None:
            current.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            current.append(c)
            i += 1
            continue
        if c == ";":
            flush()
            prev_op = ";"
            i += 1
            continue
        if c == "&" and i + 1 < n and command[i + 1] == "&":
            flush()
            prev_op = "&&"
            i += 2
            continue
        if c == "|" and i + 1 < n and command[i + 1] == "|":
            flush()
            prev_op = "||"
            i += 2
            continue
        current.append(c)
        i += 1

    flush()
    return segments


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _run_segment(command: str, work_dir: Path, timeout: int) -> tuple[int, str]:
    """Run a single operator-free command segment in the sandbox.

    Returns ``(returncode, output_string)``. ``returncode`` is 0 on success and
    non-zero on failure, an error, or an invalid/denied command, so callers can
    apply ``&&``/``||`` short-circuit semantics. ``output_string`` is the
    formatted output, or ``""`` when the segment produced nothing to report.
    """
    # Parse command into args
    try:
        args = shlex.split(command)
    except ValueError as e:
        return 1, f"Invalid command syntax: {e}"

    if not args:
        return 0, ""

    # Resolve and validate against the allowlist (or a local binary under cwd)
    binary, final_args, cfg = _resolve_command(args, work_dir)
    if binary is None:
        return 1, final_args  # error message

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
    env = None
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
            return 1, f"Error creating python site dir {site_base}: {e}"
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

    # Narrow the TOCTOU window for local binaries: re-verify the resolved
    # path is still an executable contained within the work dir right before
    # exec, in case a directory component was swapped in the meantime.
    if cfg.get("is_local_binary") and not _binary_still_contained(binary, work_dir):
        return 1, f"Local binary no longer valid inside working directory: {binary}"

    try:
        result = subprocess.run(
            sandbox_args,
            capture_output=True,
            timeout=timeout,
            cwd=str(work_dir),
            env=env,
        )

        # Capture raw bytes and decode lossily so binary/non-UTF-8 output
        # never crashes the tool.
        stdout = result.stdout[:MAX_OUTPUT].decode("utf-8", errors="replace")
        stderr_out = result.stderr[:MAX_OUTPUT].decode("utf-8", errors="replace")

        output = []
        if result.returncode != 0:
            output.append(f"Exit code: {result.returncode}")
        if stdout:
            output.append(stdout.rstrip())
        if stderr_out:
            output.append(f"[stderr]\n{stderr_out.rstrip()}")

        if len(result.stdout) > MAX_OUTPUT:
            output.append(f"\n[output truncated at {MAX_OUTPUT} bytes]")
        if len(result.stderr) > MAX_OUTPUT:
            output.append(f"\n[stderr truncated at {MAX_OUTPUT} bytes]")

        return result.returncode, "\n".join(output)

    except subprocess.TimeoutExpired:
        return 1, f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return 1, f"Sandbox binary not found: {SANDBOX_BIN}"
    except OSError as e:
        return 1, f"Error running command: {e}"


@mcp.tool()
def shell_run(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Run a shell command in a pledge sandbox.

    Commands are executed via the pledge-wrapped sandbox binary with
    restricted system call promises. Only allowlisted commands can run.

    Command chaining with `;`, `&&`, and `||` is supported: the command string
    is split on those operators and each segment is run through the sandbox and
    checked against the allowlist independently. `&&` runs its segment only if
    the previous one succeeded; `||` only if it failed; `;` always runs. Pipes
    (`|`) and redirects (`>`, `>>`) are not supported.

    Args:
        command: The command to run (e.g., "git status", "cd build && make test")
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

    # Split into allowlist-checked segments on ; / && / ||
    segments = _split_command(command)
    if not segments:
        return "Empty command."

    # Single-command fast path — preserves the exact prior behaviour.
    if len(segments) == 1 and segments[0][0] is None:
        _rc, out = _run_segment(segments[0][1], work_dir, timeout)
        return out if out else "(no output)"

    # Multi-segment chain: run each segment through the sandbox, applying
    # `&&` / `||` short-circuit semantics based on the previous segment's exit
    # code. Segments are independent processes (no shared shell variables), but
    # they share the same cwd, so file-based state persists across segments.
    outputs: list[str] = []
    prev_rc = 0
    ran_any = False

    for op, segment in segments:
        if op == "&&" and ran_any and prev_rc != 0:
            outputs.append(f"(skipped: previous command exited {prev_rc}) — {segment}")
            continue
        if op == "||" and ran_any and prev_rc == 0:
            outputs.append("(skipped: previous command succeeded) — " + segment)
            continue

        rc, out = _run_segment(segment, work_dir, timeout)
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
    lines.append("Command chaining: ';', '&&', and '||' are supported. Each")
    lines.append("    segment is run through the sandbox and checked against the")
    lines.append("    allowlist independently. '&&' runs only after success, '||'")
    lines.append("    only after failure, ';' always. No pipes ('|') or redirects.")
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
