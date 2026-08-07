#!/usr/bin/env python3
"""Shell Sandbox MCP Server — safe shell command execution via pledge + busybox.

Tools:
  shell_run(command, cwd, timeout) — run a command in a pledge sandbox
  shell_list                          — list allowed commands
"""

import subprocess
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Re-export parser symbols for backward compatibility.
# Tests do `from shell_sandbox_mcp.server import Expansion, Redirect, ...`.
# ---------------------------------------------------------------------------

from .parser import (  # noqa: F401
    CommandNode,
    Expansion,
    ParseError,
    ProgramNode,
    Redirect,
    _expand_subst_in_text as _parser_expand_subst_in_text,
    _serialize_command,
    _serialize_pipeline,
    cmd_to_display,
    extract_redirects as _parser_extract_redirects,
    parse_command as _parser_parse_command,
    program_to_chain,
    split_legacy,
)

# ---------------------------------------------------------------------------
# Re-export config symbols
# ---------------------------------------------------------------------------

from .config import (  # noqa: F401
    BUSYBOX_BIN,
    COSMO_TOOLCHAIN,
    DEFAULT_ALLOWED_DIRS,
    DEFAULT_TIMEOUT,
    EXTRA_REDIRECT_ROOTS,
    MAX_HEREDOC_BODY,
    MAX_OUTPUT,
    MAX_SUBST_COUNT,
    MAX_SUBST_DEPTH,
    MAX_SUBST_OUTPUT,
    MAX_TIMEOUT,
    MUSL_TOOLCHAIN,
    REPO_ROOT,
    SANDBOX_BIN,
    SANDBOX_WRAPPER,
    _ENV_ALLOWLIST,
    _base_env,
    _python_version,
)

# ---------------------------------------------------------------------------
# Re-export containment symbols
# ---------------------------------------------------------------------------

from .containment import (  # noqa: F401
    _binary_still_contained,
    _contained_in_any,
    _contained_path,
    _validate_cwd,
    _validate_redirect_paths,
)

# ---------------------------------------------------------------------------
# Re-export redirects symbols
# ---------------------------------------------------------------------------

from .redirects import (  # noqa: F401
    FdDefaults,
    FdPlan,
    RedirectPlan,
    _extract_redirects,
    _resolve_fd_targets,
)

# ---------------------------------------------------------------------------
# Re-export policy symbols
# ---------------------------------------------------------------------------

from .policy import (  # noqa: F401
    BUSYBOX_APPLETS,
    COMMANDS,
    _cosmo_toolchain_bin,
    _cosmo_toolchain_paths,
    _git_config_paths,
    _git_credential_paths,
    _git_extra_rx_paths,
    _git_readonly_paths,
    _maybe_venv_cfg,
    _musl_toolchain_bin,
    _musl_toolchain_paths,
    _resolve_command,
    _resolve_local_binary,
    _resolve_venv_fallback,
    _stage_git_global_config,
)

# ---------------------------------------------------------------------------
# Re-export builtins symbols
# ---------------------------------------------------------------------------

from .builtins import (  # noqa: F401
    _apply_timeout_builtin,
    _try_cd,
    _try_for_loop,
)

# ---------------------------------------------------------------------------
# Re-export executor symbols
# ---------------------------------------------------------------------------

from .executor import (  # noqa: F401
    EmptyInvocation,
    Invocation,
    InvocationError,
    _build_invocation,
    _capture_stdout,
    _expand_command,
    _expand_subst_in_text,
    _format_output,
    _run_background,
    _run_pipeline,
    _run_pipeline_core,
    _run_segment,
    _run_segment_core,
    _serialize_pipeline_from_cmds,
    _start_reaper,
)

# ---------------------------------------------------------------------------
# Re-export runner symbols
# ---------------------------------------------------------------------------

from .runner import Runner  # noqa: F401

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "shell-sandbox",
    instructions="Run shell commands safely via pledge + busybox sandbox",
)


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

    Redirects (``>``, ``>>``, ``2>``, ``2>>``, ``2>&1``, ``1>&2``, ``<``) are
    supported per command segment. Redirect targets must stay inside the
    working directory or under /tmp. Quoted operators (``echo ">"``) are
    treated as literal arguments, not redirects. Output redirects and input
    redirects (``< file``) are supported; heredocs (``<<``, ``<<-``, ``<<<``)
    feed stdin. Redirect targets must stay inside the working directory or
    under /tmp. Redirecting stdout of an intermediate pipe stage is rejected
    (stderr redirects and ``2>&1`` are allowed on intermediate stages).

    Heredocs (``<<EOF``, ``<<'EOF'``, ``<<-EOF``) and here-strings
    (``<<<'literal'``) feed literal text to the command's stdin.
    Command substitution (``$(command ...)``) recursively executes the inner
    command and splices its stdout as a single argument word.

    ``cd <dir>`` is a per-call builtin: it changes the working directory for
    subsequent segments of the SAME call only (e.g. ``cd build && make``).
    It does NOT persist across separate ``shell_run`` invocations. The
    target directory is validated against the allowed-dir containment rules
    (same as ``cwd``). Bare ``cd`` with no argument is rejected.

    ``timeout N CMD…`` is a per-pipeline builtin: it overrides the timeout
    for the current pipeline to *N* seconds (clamped to MAX_TIMEOUT=300).
    ``timeout`` must be the first word of the first segment of a pipeline
    (e.g. ``timeout 5 long_cmd`` or ``timeout 3 a | b``); it is rejected
    on non-first stages and when backgrounding (``&``). The builtin is
    implemented in pure Python and does not spawn a subprocess, so it does
    not need a ``vfork``-capable pledge.

    ``for VAR [in WORD…] [;] do BODY done`` iterates over the word list,
    re-parsing the body with ``$VAR`` bound to each word's current value.
    The ``in`` clause is optional; when omitted the loop runs zero iterations
    (the sandbox has no positional parameters).  ``$()``, ``$VAR``,
    ``cd``, ``timeout``, pipes, and redirects all work inside the body.
    ``cd`` inside the body does NOT persist across iterations or out of the
    loop.  The for-loop must be the **entire command string** — it cannot
    appear mid-chain with ``;``, ``&&``, or ``||``.

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

    # for-loop builtin — detect and execute the entire command as a for-loop
    # before expansion so the body is re-parsed per iteration with the loop
    # variable bound.  Returns None when the command is NOT a for-loop.
    loop = _try_for_loop(command, work_dir, timeout, depth=0)
    if loop is not None:
        return loop[0]

    # Expand $(...) heredocs and here-strings BEFORE splitting.
    # Also parse into an AST so the execution path consumes it directly
    # without re-lexing the cleaned string.
    try:
        expanded, expansion, program = _expand_command(command, work_dir, timeout, depth=0)
    except (ParseError, ValueError) as e:
        return str(e)

    # Walk the AST directly (Option B: single-parse path).  The AST is now the
    # ONLY execution path — the legacy string-based split_legacy fallback
    # was removed in U2.
    if program is None:
        # Defensive: parse_command returns program=None only if AST building
        # failed after the scanner succeeded (see parser.parse_command). Surface
        # a clean error string rather than crashing. Do NOT route to the
        # legacy splitter.
        return "Command parse error."
    chains = program_to_chain(program)
    if not chains:
        return "Empty command."

    # Delegate chain walking (single-command fast path AND `;` / `&&` / `||`
    # multi-pipeline chains) to the Runner, which owns the mutable per-call
    # work_dir / prev_rc / outputs state.
    runner = Runner(work_dir=work_dir, default_timeout=timeout, expansion=expansion)
    return runner.run_chain(chains, timeout)


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
    lines.append("    Redirects (>, >>, 2>, 2>>, 2>&1, 1>&2, <) are supported per")
    lines.append("    segment; < reads from a file. Targets must stay inside the")
    lines.append("    working directory or under /tmp. Quoted operators")
    lines.append("    are not treated as redirects. Heredocs (<<, <<-, <<') and")
    lines.append("    here-strings (<<<) feed stdin; command substitution ($(...))")
    lines.append(f"    splices stdout as a single arg (depth {MAX_SUBST_DEPTH}, count")
    lines.append(f"    {MAX_SUBST_COUNT}, max output {MAX_SUBST_OUTPUT:,}, max body")
    lines.append(f"    {MAX_HEREDOC_BODY:,}).")
    lines.append("")
    lines.append("Builtins: 'cd <dir>' changes the working directory for the rest of the")
    lines.append("    same shell_run call only (not persisted across calls). The target")
    lines.append("    directory is validated against the same containment rules as the")
    lines.append(f"    initial cwd ({', '.join(DEFAULT_ALLOWED_DIRS)}). Bare 'cd' with no")
    lines.append("    argument is rejected (no $HOME concept in the sandbox).")
    lines.append("")
    lines.append("    'for VAR [in WORD…] [;] do BODY done' iterates over the word list,")
    lines.append("    re-parsing the body with $VAR bound to each word. $(), $VAR, cd,")
    lines.append("    timeout, pipes, and redirects all work inside the body. cd inside")
    lines.append("    the body does NOT persist across iterations. The for-loop must be")
    lines.append("    the entire command (not mid-chain with ;, &&, or ||).")
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
