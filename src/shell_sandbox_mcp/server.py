#!/usr/bin/env python3
"""Shell Sandbox MCP Server — safe shell command execution via pledge + busybox.

Tools:
  shell_run(command, cwd, timeout, structured) — run a command in a pledge sandbox
  shell_list                                   — list allowed commands
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
    ForNode,
    IfNode,
    ParseError,
    ProgramNode,
    Redirect,
    WhileNode,
    _expand_subst_in_text as _parser_expand_subst_in_text,
    _serialize_command,
    _serialize_pipeline,
    cmd_to_display,
    extract_redirects as _parser_extract_redirects,
    parse_command as _parser_parse_command,
    program_to_chain,
    segment_needs_variable_state,
    split_chains,
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
    MAX_LOOP_ITER,
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
    _split_assignment_prefix,
    _try_cd,
    _try_export,
    _try_set,
    _try_shift,
    _try_source,
    _try_unset,
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

from .runner import LoopSignal, Runner  # noqa: F401

# ---------------------------------------------------------------------------
# Re-export variables symbols
# ---------------------------------------------------------------------------

from .variables import VariableStore  # noqa: F401

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "shell-sandbox",
    instructions="Run shell commands safely via pledge + busybox sandbox",
)

# Session-level function registry — shared across all shell_run calls within
# one MCP process (FastMCP stdio has one client per process).  Declared at
# module level so it survives across invocations and persists user-defined
# function definitions for the lifetime of the server.
_SESSION_FUNCTIONS: dict[str, str] = {}


@mcp.tool()
def shell_run(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    structured: bool = False,
) -> "str | dict":
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
    loop.  The for-loop body may contain nested control flow
    (``if``/``while``/``until``/``for``).  The loop variable persists after
    the loop (matching POSIX semantics).

    ``if CONDITION; then BODY; [elif CONDITION; then BODY;] [else BODY;] fi``
    runs the first branch whose condition succeeds (exit code 0), or the
    ``else`` branch if none match.  Conditions are shell command lists whose
    exit code determines truth.

    ``while CONDITION; do BODY; done`` and ``until CONDITION; do BODY; done``
    loop while (or until) the condition succeeds, capped at
    ``MAX_LOOP_ITER`` iterations.  Conditions and bodies may contain nested
    control flow.

    ``VAR=value`` as a standalone statement sets a shell variable for the
    remainder of the same ``shell_run`` call (e.g. ``VAR=hello; echo $VAR``).
    Shell variables are NOT passed to subprocess environments unless exported
    with ``export``.

    ``VAR=x cmd`` sets *VAR* in the subprocess environment for *cmd* only
    (env-prefix).  The variable is available to *cmd* but the assignment
    does NOT persist as a shell variable after the command finishes.

    Builtins for variable management:
    - ``export [VAR[=value]]`` — mark variables exported.  ``export VAR=x``
      sets and exports; ``export VAR`` marks existing VAR (or ``""``) as
      exported.  No args prints sorted ``name=value`` of exported vars.
    - ``unset VAR [...]`` — remove variables from the store.  Silent if
      the variable does not exist.
    - ``set [name=value ...]`` — ``set VAR=x`` stores locally (NOT exported).
      Flags starting ``-`` or ``+`` (e.g. ``set -e``) are silently ignored.
      No args prints sorted ``name=value`` of ALL vars.
    - ``shift [n]`` — always returns rc=1 (no positional parameters in the
      sandbox); non-numeric arg produces an error.
    - ``source file`` / ``. file`` — execute *file* as a shell script with
      the shared variable store so mutations persist (POSIX ``source``).
      The file must be within the working directory or ``/tmp``.  Recursion
      is capped at ``MAX_SOURCE_DEPTH`` (8).

    **Limitation:** these builtins are not supported in backgrounded
    pipelines (``&``); such commands are rejected with
    ``"builtin not supported in backgrounded pipeline (&)"``.  ``cd`` is
    only intercepted as a single-command pipeline segment.  When a builtin
    like ``export``/``set`` runs as a pipeline stage, its mutation persists
    to the shell store for LATER chain segments (``export Y=2 | cat; echo
    $Y`` → Y set), but it does NOT propagate into the subprocess env of
    other stages IN THE SAME pipeline — each subprocess stage's env is
    snapshotted from the exported vars at pipeline start.

    **Positional parameters:** quoted ``"$@"`` fans out to one argv entry
    per positional parameter (POSIX-compatible).  Quoted ``"$*"`` joins
    positional parameters with spaces into a single arg.  Unquoted ``$@``
    and ``$*`` both produce a single space-joined arg (no IFS field-splitting
    is performed for any ``$VAR`` expansion).  Inside heredoc bodies and
    here-strings, ``$@`` / ``$*`` are space-joined (no fan-out).  The
    for-loop ``for x in "$@"`` correctly iterates each positional.

    By default this tool returns a plain string: the per-pipeline outputs
    joined with newlines (or ``"(no output)"`` when nothing was produced),
    with ``"Exit code: N"`` lines embedded for non-zero exits.  Set
    ``structured=True`` to instead receive a JSON object:

    .. code-block:: python

        {
          "rc": <int, final exit code of the last executed pipeline>,
          "skipped": <bool, true if any &&/|| chain was skipped>,
          "stages": [
            {"command": <serialized command>, "output": <str>, "rc": <int or null>}
          ],
          "output": <str, same joined text as the default return>,
        }

    ``output`` is identical to the default string return for back-compat
    within the payload.  ``stages`` records one entry per pipeline (or
    builtin interception): ``rc`` is null for skipped chains and backgrounded
    pipelines (whose exit code is unknown by design), and is left unchanged
    by a trailing backgrounded command.

    When ``structured=True`` a dict is returned on ALL paths — including
    error/early-return paths (invalid path, cwd validation failure, parse or
    expansion error, empty command, for-loop).  On such paths ``stages`` is
    empty and ``rc`` is 1 (or the for-loop's exit code) with ``skipped`` False.

    Args:
        command: The command to run (e.g., "git status", "ls | grep foo",
            "cd build && make test")
        cwd:     Working directory (must be within allowed paths)
        timeout: Timeout in seconds (default 30, max 300)
        structured: When True, return a structured JSON object (see above)
            instead of the plain string.  Default False (string return).
    """
    timeout = min(timeout, MAX_TIMEOUT)

    def _structured_error(rc: int, text: str) -> "str | dict":
        """Return a structured error dict when requested, else the bare string."""
        if structured:
            return {"rc": rc, "skipped": False, "stages": [], "output": text}
        return text

    # Validate cwd first — resolve once, use for validation, resolution and
    # execution. Required before resolving so local binaries can be located.
    raw_cwd = cwd or "."
    try:
        work_dir = Path(raw_cwd).expanduser().resolve()
    except Exception:
        return _structured_error(1, f"Invalid path: {raw_cwd}")

    err = _validate_cwd(work_dir, raw_cwd)
    if err:
        return _structured_error(1, err)

    # Variable assignment + builtins gate: lex-only detection to decide
    # whether to take the per-chain re-expansion path (Approach A).  If NO
    # segment starts with an assignment word or one of the new builtins,
    # take the existing single-expand path unchanged — byte-for-byte identical.
    segments = split_chains(command)
    needs_state = any(segment_needs_variable_state(seg, _SESSION_FUNCTIONS.keys()) for _, seg, _ in segments)

    if needs_state:
        runner = Runner(
            work_dir=work_dir, default_timeout=timeout,
            variables=VariableStore(functions=_SESSION_FUNCTIONS),
        )
        if structured:
            return runner.run_command(command, timeout, structured=True).to_dict()
        return runner.run_command(command, timeout)

    # Expand $(...) heredocs and here-strings BEFORE splitting.
    # Also parse into an AST so the execution path consumes it directly
    # without re-lexing the cleaned string.
    try:
        expanded, expansion, program = _expand_command(command, work_dir, timeout, depth=0)
    except (ParseError, ValueError) as e:
        return _structured_error(1, str(e))

    # Walk the AST directly (single-parse path).
    assert program is not None, "_expand_command must return a ProgramNode"
    chains = program_to_chain(program)
    if not chains:
        return _structured_error(1, "Empty command.")

    # Delegate chain walking (single-command fast path AND `;` / `&&` / `||`
    # multi-pipeline chains) to the Runner, which owns the mutable per-call
    # work_dir / prev_rc / outputs state.
    runner = Runner(work_dir=work_dir, default_timeout=timeout, expansion=expansion)
    if structured:
        return runner.run_chain(chains, timeout, structured=True).to_dict()
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
    lines.append("    the body does NOT persist across iterations. The body may contain")
    lines.append("    nested control flow (if/while/until/for). The loop variable")
    lines.append("    persists after the loop (matching POSIX semantics).")
    lines.append("")
    lines.append("    'if COND; then BODY; [elif ...] [else ...] fi',")
    lines.append("    'while COND; do BODY; done', and 'until COND; do BODY; done'")
    lines.append("    provide POSIX-compatible control flow. Conditions are shell")
    lines.append(f"    command lists; loops are capped at {MAX_LOOP_ITER} iterations.")
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
