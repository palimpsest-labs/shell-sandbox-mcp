"""Builtin commands — per-call ``cd`` working directory change.

The ``cd`` builtin does NOT spawn a subprocess; it resolves the target
directory within the current sandbox invocation and updates the work_dir
for subsequent segments of the same ``shell_run`` call.
"""

from pathlib import Path
from typing import Optional

from .parser import Expansion


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
