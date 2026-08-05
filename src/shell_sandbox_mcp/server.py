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
DEFAULT_ALLOWED_DIRS = [
    str(Path.home() / "github"),
    str(Path.home() / "projects"),
    "/tmp",
]
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300
MAX_OUTPUT = 1_000_000  # 1 MB

# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------

COMMANDS = {
    "git": {
        "binary": "/usr/bin/git",
        "promises": "stdio rpath wpath cpath prot_exec",
        "description": "Git version control",
    },
    "cargo": {
        "binary": "cargo",
        "promises": "stdio rpath wpath cpath proc prot_exec",
        "description": "Rust package manager (build, test, check, fmt, clippy)",
    },
    "make": {
        "binary": "make",
        "promises": "stdio rpath wpath cpath proc prot_exec",
        "description": "GNU make build tool (spawns compiler subprocesses)",
    },
}

BUSYBOX_APPLETS = [
    "cat", "head", "tail", "wc", "sort", "uniq",
    "grep", "ls", "echo", "test", "expr",
    "mkdir", "cp", "mv", "chmod",
    "cut", "tr", "diff", "cmp", "md5sum", "sha256sum",
    "which", "basename", "dirname", "realpath",
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


def _resolve_command(args: list[str]) -> tuple[str, list[str], dict] | tuple[None, str, None]:
    """Resolve and validate a command against the allowlist.

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

    # Check direct commands (git, cargo)
    if cmd_name not in COMMANDS:
        return None, f"Command not allowed: {cmd_name}. Use shell_list to see allowed commands.", None

    cfg = COMMANDS[cmd_name]
    binary = cfg["binary"]
    return binary, args, cfg


def _validate_cwd(cwd: str) -> Optional[str]:
    """Validate working directory. Returns error or None."""
    try:
        resolved = Path(cwd).expanduser().resolve()
    except Exception:
        return f"Invalid path: {cwd}"

    if not resolved.is_dir():
        return f"Directory not found: {cwd}"

    # Must be within an allowed tree or a subdirectory thereof
    for allowed in DEFAULT_ALLOWED_DIRS:
        allowed_path = Path(allowed).expanduser().resolve()
        try:
            resolved.relative_to(allowed_path)
            return None
        except ValueError:
            continue

    return f"Directory not in allowed paths: {cwd}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def shell_run(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Run a shell command in a pledge sandbox.

    Commands are executed via the pledge-wrapped sandbox binary with
    restricted system call promises. Only allowlisted commands can run.

    Args:
        command: The command to run (e.g., "git status", "busybox grep pattern file")
        cwd:     Working directory (must be within allowed paths)
        timeout: Timeout in seconds (default 30, max 300)
    """
    timeout = min(timeout, MAX_TIMEOUT)

    # Parse command into args
    try:
        args = shlex.split(command)
    except ValueError as e:
        return f"Invalid command syntax: {e}"

    if not args:
        return "Empty command."

    # Resolve and validate
    binary, final_args, cfg = _resolve_command(args)
    if binary is None:
        return final_args  # error message

    # Validate cwd — resolve once, use for both validation and execution
    raw_cwd = cwd or "."
    try:
        work_dir = Path(raw_cwd).expanduser().resolve()
    except Exception:
        return f"Invalid path: {raw_cwd}"

    err = _validate_cwd(str(work_dir))
    if err:
        return err

    # Build sandbox invocation via the exec wrapper (bypasses posix_spawn APE issue)
    sandbox_args = [
        str(SANDBOX_WRAPPER.resolve()),
        cfg["promises"],
        str(work_dir),  # unveil directory
        "--",
        binary,
    ] + final_args[1:]

    try:
        result = subprocess.run(
            sandbox_args,
            capture_output=True,
            timeout=timeout,
            cwd=str(work_dir),
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

        if not output:
            return "(no output)"

        return "\n".join(output)

    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return f"Sandbox binary not found: {SANDBOX_BIN}"
    except OSError as e:
        return f"Error running command: {e}"


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
