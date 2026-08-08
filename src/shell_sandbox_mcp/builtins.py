"""Builtin commands — per-call ``cd`` working directory change,
``timeout`` builtin (fork-free, implemented in Python), and variable/assignment
builtins (export, unset, set, shift, source/.).

The ``cd`` and ``timeout`` builtins do NOT spawn a subprocess.  ``cd``
resolves the target directory within the current sandbox invocation and
updates the work_dir for subsequent segments of the same ``shell_run`` call.
``timeout N CMD…`` overrides the per-pipeline timeout without requiring a
``vfork``-capable pledge (busybox ``timeout`` uses ``vfork``, which no
cosmocc pledge token grants).

Control-flow constructs (``if``/``while``/``until``/``for``) are now parsed
AST-natively in :mod:`parser` and executed in :mod:`runner`.
"""

import dataclasses
import re as _re
from pathlib import Path
from typing import Optional

from .config import MAX_TIMEOUT
from .parser import CommandNode, Expansion, ParseError, program_to_chain


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
    args, _redirects, _err = srv._extract_redirects(cmd, expansion, work_dir)
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
    work_dir: Optional[Path] = None,
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
    args, _redirects, _err = srv._extract_redirects(cmd_nodes[0], expansion, work_dir)
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


# ---------------------------------------------------------------------------
# Assignment prefix detector
# ---------------------------------------------------------------------------

_ASSIGN_NAME_RE = _re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _split_assignment_prefix(
    cmd,
    expansion: Optional[Expansion],
    work_dir: Optional[Path],
) -> tuple[Optional[dict[str, str]], Optional["CommandNode"], Optional[str]]:
    """Split leading ``VAR=value`` words from *cmd*.

    Returns ``(prefix, remaining_node, err)``:

    * *prefix* — ``None`` when the first word is NOT an assignment, or a dict
      of ``{VAR: value}`` pairs for every leading assignment word.
    * *remaining_node* — the ``CommandNode`` with assignment words stripped,
      or ``None`` when ALL words are assignments (pure assignment statement).
    * *err* — string on glob-guard failure, else ``None``.

    Edge cases handled:
    * ``=foo`` → NOT an assignment (LHS must be ``[A-Za-z_][A-Za-z0-9_]*``).
    * ``VAR=`` → value ``""``.
    * ``VAR=a=b`` → value ``"a=b"`` (split on first ``=`` only).
    * Leading ``--`` stops detection.
    * Guard: if glob expansion made ``len(args) != len(cmd.words)``, fall
      back to ``(None, cmd, None)`` so the caller treats it as a normal command.
    """
    srv = _get_server()
    args, _redirects, err = srv._extract_redirects(cmd, expansion, work_dir)
    if err is not None:
        return None, cmd, err

    # Guard: glob expansion changed the arg count → fallback to normal.
    if len(args) != len(cmd.words):
        return None, cmd, None

    prefix: dict[str, str] = {}
    assign_count = 0

    for arg in args:
        if arg == "--":
            # -- stops assignment detection
            break
        eq_idx = arg.find("=")
        if eq_idx < 1:
            # Either no '=' or '=foo' (eq_idx == 0) — not an assignment
            break
        lhs = arg[:eq_idx]
        if not _ASSIGN_NAME_RE.match(lhs):
            break
        rhs = arg[eq_idx + 1:]
        prefix[lhs] = rhs
        assign_count += 1

    if assign_count == 0:
        return None, cmd, None

    # Build remaining CommandNode with leading assignment words stripped.
    from .parser import CommandNode
    remaining_words = tuple(cmd.words[assign_count:])

    if not remaining_words:
        # Pure assignment — no command follows
        return prefix, None, None

    remaining = dataclasses.replace(
        cmd,
        words=remaining_words,
    )
    return prefix, remaining, None


_BUILTIN_STAGE_NAMES = ("export", "unset", "set", "shift", "source", ".")


def _classify_builtin(cmd, expansion, work_dir) -> Optional[str]:
    """Return the builtin name if *cmd*'s first arg is one of the 5 variable
    builtins (or ``.``), else None.  Does NOT execute — safe to call as a
    probe.
    """
    srv = _get_server()
    args, _redirects, _err = srv._extract_redirects(cmd, expansion, work_dir)
    if not args:
        return None
    return args[0] if args[0] in _BUILTIN_STAGE_NAMES else None


# ---------------------------------------------------------------------------
# Variable/assignment builtins — export, unset, set, shift, source / .
# ---------------------------------------------------------------------------


def _try_export(
    cmd,
    expansion: Optional[Expansion],
    work_dir: Optional[Path],
    store,
) -> tuple[bool, Optional[str], Optional[int]]:
    """Handle ``export [name[=value] ...]`` builtin.

    Returns ``(True, output, rc)`` when *cmd* is ``export``, else
    ``(False, None, None)``.
    """
    srv = _get_server()
    args, _redirects, _err = srv._extract_redirects(cmd, expansion, work_dir)
    if not args or args[0] != "export":
        return False, None, None

    if len(args) == 1:
        # export with no args → print sorted name=value of exported vars
        lines = sorted(
            f"{k}={store.variables.get(k, '')}"
            for k in store.exported
            if k in store.variables
        )
        return True, "\n".join(lines) if lines else "", 0

    for arg in args[1:]:
        eq_idx = arg.find("=")
        if eq_idx >= 1:
            name = arg[:eq_idx]
            value = arg[eq_idx + 1:]
            if not _ASSIGN_NAME_RE.match(name):
                return True, f"export: not a valid variable name: {arg}", 1
            store.set_export(name, value)
        else:
            if not _ASSIGN_NAME_RE.match(arg):
                return True, f"export: not a valid variable name: {arg}", 1
            store.mark_export(arg)

    return True, "", 0


def _try_unset(
    cmd,
    expansion: Optional[Expansion],
    work_dir: Optional[Path],
    store,
) -> tuple[bool, Optional[str], Optional[int]]:
    """Handle ``unset [name ...]`` builtin.

    Returns ``(True, output, rc)`` when *cmd* is ``unset``, else
    ``(False, None, None)``.
    """
    srv = _get_server()
    args, _redirects, _err = srv._extract_redirects(cmd, expansion, work_dir)
    if not args or args[0] != "unset":
        return False, None, None

    for arg in args[1:]:
        store.unset(arg)
    return True, "", 0


def _try_set(
    cmd,
    expansion: Optional[Expansion],
    work_dir: Optional[Path],
    store,
) -> tuple[bool, Optional[str], Optional[int]]:
    """Handle ``set [name=value ...]`` builtin.

    Returns ``(True, output, rc)`` when *cmd* is ``set``, else
    ``(False, None, None)``.
    """
    srv = _get_server()
    args, _redirects, _err = srv._extract_redirects(cmd, expansion, work_dir)
    if not args or args[0] != "set":
        return False, None, None

    if len(args) == 1:
        # set with no args → print sorted name=value of ALL vars
        lines = sorted(
            f"{k}={v}" for k, v in store.variables.items()
        )
        return True, "\n".join(lines) if lines else "", 0

    for arg in args[1:]:
        eq_idx = arg.find("=")
        if eq_idx >= 1:
            name = arg[:eq_idx]
            value = arg[eq_idx + 1:]
            store.set_local(name, value)
        elif arg.startswith("-") or arg.startswith("+"):
            # set -e, set +x etc. → no-op, rc 0
            continue
        else:
            return True, f"set: unsupported argument: {arg}", 1

    return True, "", 0


def _try_shift(
    cmd,
    expansion: Optional[Expansion],
    work_dir: Optional[Path],
    store,
) -> tuple[bool, Optional[str], Optional[int]]:
    """Handle ``shift [n]`` builtin.

    Returns ``(True, output, rc)`` when *cmd* is ``shift``, else
    ``(False, None, None)``.  Pops from ``store.positional`` when
    positional parameters are available; returns rc=1 when empty.
    """
    srv = _get_server()
    args, _redirects, _err = srv._extract_redirects(cmd, expansion, work_dir)
    if not args or args[0] != "shift":
        return False, None, None

    n = 1
    if len(args) > 1:
        try:
            n = int(args[1])
        except ValueError:
            return True, f"shift: invalid argument: {args[1]}", 1

    if n < 1:
        return True, f"shift: non-positive argument: {n}", 1

    pos = list(store.positional)
    if len(pos) < n:
        store.positional = ()
        return True, "", 1

    store.positional = tuple(pos[n:])
    return True, "", 0


def _try_source(
    cmd,
    expansion: Optional[Expansion],
    work_dir: Optional[Path],
    store,
    timeout: int,
    depth: int,
) -> tuple[bool, Optional[str], Optional[int]]:
    """Handle ``source file`` / ``. file`` builtin.

    Returns ``(True, output, rc)`` when *cmd* is ``source`` or ``.``, else
    ``(False, None, None)``.

    Reads *file* (must be contained within the work directory / extra redirect
    roots) and executes its contents as a shell command with the shared
    ``store`` so mutations persist (POSIX ``source`` semantics).
    """
    from pathlib import Path

    from .config import EXTRA_REDIRECT_ROOTS, MAX_HEREDOC_BODY, MAX_SOURCE_DEPTH
    from .containment import _contained_in_any

    srv = _get_server()
    args, _redirects, _err = srv._extract_redirects(cmd, expansion, work_dir)
    if not args:
        return False, None, None
    if args[0] not in ("source", "."):
        return False, None, None

    if len(args) < 2:
        return True, "source: missing file argument", 1

    path_str = args[1]
    if work_dir is None:
        return True, f"source: no working directory", 1

    # Depth guard first — avoid reading a file needlessly at the boundary.
    if depth >= MAX_SOURCE_DEPTH:
        return True, f"source: recursion depth limit ({MAX_SOURCE_DEPTH}) exceeded", 1

    cand = _contained_in_any(path_str, [work_dir, *EXTRA_REDIRECT_ROOTS])
    if cand is None:
        return True, f"source: file escapes sandbox: {path_str}", 1

    file_path = Path(cand)
    if not file_path.is_file():
        return True, f"source: file not found or unreadable: {path_str}", 1

    try:
        with open(str(file_path), "r", encoding="utf-8", errors="replace") as fh:
            contents = fh.read(MAX_HEREDOC_BODY + 1)
    except OSError as e:
        return True, f"source: cannot read file: {path_str}: {e}", 1

    if len(contents) > MAX_HEREDOC_BODY:
        contents = contents[:MAX_HEREDOC_BODY]

    from .runner import Runner
    runner = Runner(work_dir=work_dir, default_timeout=timeout, variables=store)
    out = runner.run_command(contents, timeout, structured=False, depth=depth + 1)
    return True, out, runner.prev_rc
