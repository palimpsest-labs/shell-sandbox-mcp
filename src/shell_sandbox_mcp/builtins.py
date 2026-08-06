"""Builtin commands — per-call ``cd`` working directory change and
``timeout`` builtin (fork-free, implemented in Python).

The ``cd`` and ``timeout`` builtins do NOT spawn a subprocess.  ``cd``
resolves the target directory within the current sandbox invocation and
updates the work_dir for subsequent segments of the same ``shell_run`` call.
``timeout N CMD…`` overrides the per-pipeline timeout without requiring a
``vfork``-capable pledge (busybox ``timeout`` uses ``vfork``, which no
cosmocc pledge token grants).
"""

import dataclasses
from pathlib import Path
from typing import Optional

from .config import MAX_TIMEOUT
from .parser import CommandNode, Expansion


def _get_server():
    """Lazy accessor for the server module (avoids circular import at module level)."""
    from . import server
    return server


def _try_cd(cmd, work_dir: Path, expansion: Optional[Expansion] = None) -> tuple[Optional[Path], Optional[str]]:
    """Detect and execute a ``cd`` builtin without spawning a subprocess.

    ``cmd`` may be a ``str`` (legacy path) or a :class:`parser.CommandNode`
    (AST-native path).  Returns ``(new_work_dir, None)`` on success,
    ``(None, error_msg)`` on failure, or ``(None, None)`` when the command
    is NOT a ``cd`` (caller falls through to normal dispatch).
    """
    srv = _get_server()
    args, _redirects, _err = srv._extract_redirects(cmd, expansion)
    if not args or args[0] != "cd":
        return None, None

    if len(args) == 1:
        # Bare cd — no $HOME concept in the sandbox.
        return None, "cd: no directory"

    # A leading `--` is end-of-options. `cd -- <dir>` targets <dir>;
    # `cd --` alone behaves like a bare cd.
    if args[1] == "--":
        if len(args) == 3:
            target = args[2]
        elif len(args) == 2:
            return None, "cd: no directory"
        else:
            return None, "cd: too many arguments"
    elif len(args) > 2:
        return None, "cd: too many arguments"
    else:
        target = args[1]

    try:
        # Expand `~` against the target itself before joining with work_dir,
        # mirroring how the initial cwd is resolved (Path(raw_cwd).expanduser()).
        target_path = Path(target).expanduser()
        if not target_path.is_absolute():
            target_path = work_dir / target_path
        new_dir = target_path.resolve()
    except Exception:
        return None, f"cd: invalid path: {target}"

    err = srv._validate_cwd(new_dir, target)
    if err is not None:
        return None, err

    return new_dir, None


def _apply_timeout_builtin(
    cmd_nodes: list[CommandNode],
    expansion: Optional[Expansion],
    backgrounded: bool,
    default_timeout: int,
) -> tuple[Optional[list[CommandNode]], Optional[int], Optional[str]]:
    """Detect and apply a ``timeout N`` builtin prefix on a pipeline.

    ``timeout`` is only recognized as the first word of the first segment.
    When detected, the ``timeout N`` tokens are stripped and the effective
    timeout is set to ``min(N, MAX_TIMEOUT)``.  This overrides the caller's
    default timeout for the entire pipeline.

    Returns ``(nodes, effective_timeout, None)`` on success,
    ``(None, None, error_msg)`` on failure, or ``(cmd_nodes, default_timeout,
    None)`` when the command is NOT a ``timeout`` invocation (caller falls
    through to normal dispatch).
    """
    if not cmd_nodes:
        return cmd_nodes, default_timeout, None

    srv = _get_server()
    args, _redirects, _err = srv._extract_redirects(cmd_nodes[0], expansion)
    if not args or args[0] != "timeout":
        return cmd_nodes, default_timeout, None

    if backgrounded:
        return None, None, "timeout: not supported with background (&)"

    if len(args) < 2 or args[1] == "":
        return None, None, "timeout: missing duration"

    if len(args) == 2:
        return None, None, "timeout: missing command"

    n_str = args[1]
    try:
        n = int(n_str)
    except ValueError:
        return None, None, f"timeout: invalid duration '{n_str}'"

    if n <= 0:
        return None, None, "timeout: duration must be > 0"

    n = min(n, MAX_TIMEOUT)

    # Strip "timeout N" from the first segment's words, preserving redirects
    # and backgrounded flag.  Use dataclasses.replace so we keep all other
    # fields (redirects, backgrounded) intact.
    new_first = dataclasses.replace(cmd_nodes[0], words=tuple(cmd_nodes[0].words[2:]))
    new_nodes = [new_first] + list(cmd_nodes[1:])
    return new_nodes, n, None
