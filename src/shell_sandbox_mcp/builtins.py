"""Builtin commands — per-call ``cd`` working directory change,
``timeout`` builtin (fork-free, implemented in Python), and ``for``-loop
builtin.

The ``cd`` and ``timeout`` builtins do NOT spawn a subprocess.  ``cd``
resolves the target directory within the current sandbox invocation and
updates the work_dir for subsequent segments of the same ``shell_run`` call.
``timeout N CMD…`` overrides the per-pipeline timeout without requiring a
``vfork``-capable pledge (busybox ``timeout`` uses ``vfork``, which no
cosmocc pledge token grants).

``for VAR [in WORD…] [;] do BODY done`` iterates over the word list,
re-parsing the body with ``$VAR`` bound to each word's value.  The body
may contain ``$()``, ``$VAR``, ``cd``, ``timeout``, and pipes/redirects,
all of which work per-iteration.
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
    ``(False, None, None)``.  Always returns rc=1 (no positional params).
    """
    srv = _get_server()
    args, _redirects, _err = srv._extract_redirects(cmd, expansion, work_dir)
    if not args or args[0] != "shift":
        return False, None, None

    if len(args) > 1:
        try:
            int(args[1])
        except ValueError:
            return True, f"shift: invalid argument: {args[1]}", 1

    return True, "", 1  # no positional parameters in the sandbox


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


# ---------------------------------------------------------------------------
# for-loop builtin — string-level detection with per-iteration re-parse
# ---------------------------------------------------------------------------

# Valid shell identifier for the loop variable.
_FOR_VAR_RE = _re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _try_for_loop(
    command: str,
    work_dir: Path,
    timeout: int,
    depth: int = 0,
) -> Optional[tuple[str, int]]:
    """Detect and execute a ``for VAR [in WORD…] [;] do BODY done`` loop.

    Returns ``(output_string, exit_code)`` when *command* is a for-loop and
    it ran, or ``None`` when *command* is NOT a for-loop (caller falls through
    to normal parse+run).

    Grammar supported (entire-command only)::

        for VAR [in WORD WORD …] [;] do BODY done

    - *VAR* must be ``[A-Za-z_][A-Za-z0-9_]*``.
    - If ``in`` is omitted the loop runs zero iterations (the sandbox has no
      positional parameters).
    - An optional ``;`` may appear before ``do``.
    - *BODY* is re-parsed per iteration with the loop variable bound in the
      env so ``$VAR``/``${VAR}`` expand to the current word.
    - ``$()`` inside *in* words and the body is expanded normally (recursive
      capture).
    - Globs in *in* words are expanded (via the work_dir-aware extract path).
    - ``cd`` inside the body does NOT persist across iterations or out of the
      loop (each iteration gets a fresh ``Runner`` with the original work_dir).
    - ``timeout`` inside the body works per-iteration (as with any pipeline).
    - Exit code: the last iteration's exit code is reported; zero iterations
      → exit code 0.
    - The for-loop must be the entire command string (not mid-chain with ``;``,
      ``&&``, or ``||``).  If the command starts with ``for`` but also contains
      chain operators outside the body, the for-loop is not detected and the
      caller falls through to normal parsing (which will reject ``for`` as an
      unknown command).
    """
    # Phase 1: scan the front matter and locate the body
    scan = _scan_for_loop(command)
    if scan is None:
        return None  # not a for-loop

    var_name, in_words_raw, body = scan

    # Validate var name
    if not _FOR_VAR_RE.match(var_name):
        return f"for: invalid variable name '{var_name}'", 1

    srv = _get_server()
    from .config import _ENV_ALLOWLIST, _base_env
    from .runner import Runner

    base_env = _base_env()

    # Phase 2: expand the word list
    in_words: list[str] = []
    for raw_word in in_words_raw:
        expanded = _expand_for_word(raw_word, work_dir, timeout, depth, base_env, srv)
        in_words.append(expanded)

    # Phase 3: iterate
    outputs: list[str] = []
    last_rc = 0

    for word_value in in_words:
        # Build per-iteration env with the loop variable bound
        iter_env = dict(base_env)
        iter_env[var_name] = word_value

        try:
            expanded, expansion, program = srv._expand_command(
                body, work_dir, timeout, depth + 1,
                env=iter_env,
            )
        except (ParseError, ValueError) as e:
            outputs.append(str(e))
            last_rc = 1
            continue

        if program is None:
            outputs.append("Command parse error.")
            last_rc = 1
            continue

        chains = program_to_chain(program)
        if not chains:
            continue  # empty body — no output, rc unchanged

        # Run with a fresh Runner per iteration (cd doesn't leak out)
        runner = Runner(
            work_dir=work_dir,
            default_timeout=timeout,
            expansion=expansion,
        )
        out = runner.run_chain(chains, timeout)
        if out and out != "(no output)":
            outputs.append(out)
        last_rc = runner.prev_rc

    if not outputs:
        return "(no output)", last_rc
    return "\n".join(outputs), last_rc


def _expand_for_word(
    raw_word: str,
    work_dir: Path,
    timeout: int,
    depth: int,
    base_env: dict[str, str],
    srv,
) -> str:
    """Expand a single word from the for-loop's ``in`` list.

    Uses the existing ``_expand_command`` machinery on a synthetic ``echo``
    command so that ``$()``, ``$VAR``, and glob expansion all apply.  Strips
    the leading ``echo`` (first arg) to recover the expanded word value.
    Returns the original *raw_word* unchanged on any expansion failure.
    """
    if raw_word == "":
        return ""

    try:
        expanded, expansion, program = srv._expand_command(
            f"__for_expand {raw_word}",
            work_dir, timeout, depth + 1,
            env=base_env,
        )
    except (ParseError, ValueError):
        return raw_word  # fallback: literal

    if program is None:
        return raw_word

    chains = program_to_chain(program)
    if not chains or not chains[0][1]:
        return raw_word

    from .parser import _extract_from_node
    args, _, _ = _extract_from_node(chains[0][1][0], expansion, work_dir)
    if not args or args[0] != "__for_expand":
        return raw_word
    # The expanded word is everything after the synthetic command name.
    # If there are multiple words (e.g. glob expansion), join them.
    if len(args) == 1:
        return ""
    return " ".join(args[1:])


# ---------------------------------------------------------------------------
# For-loop scanner helpers
# ---------------------------------------------------------------------------


def _skip_redirect_in_body(
    command: str, pos: int, n: int, expect_command: bool,
) -> tuple[int, bool]:
    """Skip a heredoc or here-string in the for-loop body scanner.

    Called when ``command[pos:pos+2] == '<<'`` at depth 0, not in quotes.
    Advances *pos* past the redirect and its body (for heredocs) so that
    ``done`` inside the heredoc body is not mistaken for the loop terminator.

    Returns ``(new_pos, new_expect_command)``, or ``(-1, False)`` on error
    (e.g. missing heredoc delimiter).
    """
    # Determine which redirect operator
    if pos + 2 < n and command[pos + 2] == '<':
        # Here-string: <<<WORD
        op_len = 3
        is_heredoc = False
    elif pos + 2 < n and command[pos + 2] == '-':
        # Heredoc with tab stripping: <<-DELIM
        op_len = 3
        is_heredoc = True
        strip_tabs = True
    else:
        # Plain heredoc: <<DELIM
        op_len = 2
        is_heredoc = True
        strip_tabs = False

    pos += op_len

    if not is_heredoc:
        # Here-string — skip whitespace and the following word (quote-aware).
        # The word is on the same line; it should not be confused with 'done'.
        while pos < n and command[pos] in (' ', '\t'):
            pos += 1
        # Read the here-string word (quote-aware, stop at space/tab/newline/|/;/&)
        quote = None
        while pos < n:
            c = command[pos]
            if quote is not None:
                pos += 1
                if c == quote:
                    quote = None
            elif c in ("'", '"'):
                quote = c
                pos += 1
            elif c in (' ', '\t', '\n', ';', '|', '&'):
                break
            else:
                pos += 1
        # After a here-string, we're still mid-command (expect_command unchanged)
        return pos, expect_command

    # --- heredoc (<< or <<-) ---
    # Skip whitespace after the operator
    while pos < n and command[pos] in (' ', '\t'):
        pos += 1

    # Read the delimiter word (quote-aware)
    delim_chars: list[str] = []
    dq = None
    while pos < n:
        c = command[pos]
        if dq is not None:
            delim_chars.append(c)
            pos += 1
            if c == dq:
                dq = None
        elif c in ("'", '"'):
            dq = c
            delim_chars.append(c)
            pos += 1
        elif c in (' ', '\t', '\n', ';', '|', '&'):
            break
        else:
            delim_chars.append(c)
            pos += 1

    raw_delim = "".join(delim_chars)
    # Strip quotes from the delimiter for matching (same as parser._strip_quotes)
    delimiter = _strip_heredoc_delim(raw_delim)

    if not delimiter:
        return -1, False  # malformed

    # Skip to end of current line (the heredoc body starts on the next line)
    while pos < n and command[pos] != '\n':
        pos += 1
    if pos < n:
        pos += 1  # consume newline

    # Scan line by line for the closing delimiter
    found = False
    while pos < n:
        line_start = pos
        while pos < n and command[pos] != '\n':
            pos += 1
        line = command[line_start:pos]
        if pos < n:
            pos += 1  # consume newline

        check = line
        if strip_tabs:
            check = line.lstrip('\t')
        if check == delimiter:
            found = True
            break

    if not found:
        return -1, False  # malformed heredoc

    # After a heredoc, the closing delimiter is on its own line, so the
    # next position is at a command boundary.
    return pos, True


def _strip_heredoc_delim(raw: str) -> str:
    """Strip surrounding quotes from a heredoc delimiter word.

    ``'EOF'`` → ``EOF``, ``"EOF"`` → ``EOF``, ``EOF`` → ``EOF``.
    """
    if len(raw) >= 2:
        if (raw[0] == "'" and raw[-1] == "'") or (raw[0] == '"' and raw[-1] == '"'):
            return raw[1:-1]
    return raw


# ---------------------------------------------------------------------------
# For-loop scanner
# ---------------------------------------------------------------------------

def _scan_for_loop(command: str) -> Optional[tuple[str, list[str], str]]:
    """Scan *command* for ``for VAR [in WORD…] [;] do BODY done``.

    Returns ``(var_name, in_words, body)`` on success, or ``None`` if
    *command* does not match the for-loop grammar.

    The scan is depth-aware: keywords (``for``, ``in``, ``do``, ``done``)
    are only recognised when they appear outside quotes and ``$()``.
    """
    n = len(command)
    pos = 0

    # ---- helpers ----

    def _skip_ws() -> None:
        nonlocal pos
        while pos < n and command[pos] in (' ', '\t', '\n'):
            pos += 1

    def _at_word(w: str) -> bool:
        """True if *w* starts at *pos* and is followed by a non-alnum,non-_ char or end."""
        if not command.startswith(w, pos):
            return False
        end = pos + len(w)
        if end < n and (command[end].isalnum() or command[end] == '_'):
            return False
        # Backward boundary: char before must not be alnum or _
        if pos > 0 and (command[pos - 1].isalnum() or command[pos - 1] == '_'):
            return False
        return True

    def _read_plain_word() -> str:
        """Read a word at depth 0 (no quotes / $() tracking needed for the
        front matter — the keywords ``for``, ``in``, ``do`` and the var name
        are plain identifiers)."""
        nonlocal pos
        _skip_ws()
        start = pos
        while pos < n and command[pos] not in (' ', '\t', '\n', ';', '|', '&', '<', '>'):
            pos += 1
        return command[start:pos]

    def _read_word_deep() -> str:
        """Read one word at the current position, tracking single/double
        quotes and ``$()`` nesting.  Returns the raw word text (including
        any quote characters)."""
        nonlocal pos
        _skip_ws()
        start = pos
        depth = 0
        quote = None
        while pos < n:
            c = command[pos]
            if quote is not None:
                if quote == "'":
                    if c == "'":
                        quote = None
                    pos += 1
                else:  # double quote
                    if c == '\\' and pos + 1 < n:
                        pos += 2
                    elif c == '"':
                        quote = None
                        pos += 1
                    else:
                        pos += 1
            elif c == "'":
                quote = "'"
                pos += 1
            elif c == '"':
                quote = '"'
                pos += 1
            elif c == '$' and pos + 1 < n and command[pos + 1] == '(':
                depth += 1
                pos += 2
            elif c == ')' and depth > 0:
                depth -= 1
                pos += 1
            elif depth > 0:
                pos += 1
            elif c in (' ', '\t', '\n', ';', '|', '&', '<', '>'):
                break
            else:
                pos += 1
        return command[start:pos]

    # ---- scan proper ----

    _skip_ws()
    if pos >= n:
        return None

    # 1. ``for``
    if not _at_word('for'):
        return None
    pos += 3  # skip 'for'

    # 2. var name
    var_name = _read_plain_word()
    if not var_name:
        return None

    # 3. optional ``in`` with word list, or optional ``;``, then ``do``
    _skip_ws()
    if pos >= n:
        return None

    in_words: list[str] = []

    if _at_word('in'):
        pos += 2  # skip 'in'
        # Collect words until ``do`` or ``;``
        while pos < n:
            _skip_ws()
            if pos >= n:
                return None  # missing do
            if command[pos] == ';':
                pos += 1
                break
            if _at_word('do'):
                break
            w = _read_word_deep()
            if w:
                in_words.append(w)
    elif command[pos] == ';':
        pos += 1
        _skip_ws()
        if not _at_word('do'):
            return None  # expected 'do' after ';'
        pos += 2  # skip 'do'
    elif _at_word('do'):
        pos += 2  # skip 'do'; no in clause → no words
    else:
        return None  # expected 'in', 'do', or ';'

    # If we found ``;`` in the in-words loop, consume the following ``do``
    if pos >= 2 and command[pos - 2:pos] != 'do':
        # We broke out on ';' — expect 'do' next
        _skip_ws()
        if not _at_word('do'):
            return None
        pos += 2

    # Skip whitespace after 'do' to find body start
    _skip_ws()
    body_start = pos

    # 4. Find the matching ``done`` at depth 0 and at a command-boundary
    #    position.  Scan the remainder character by character, tracking quote,
    #    $() depth, expect_command state, and heredoc bodies.
    depth = 0
    quote = None
    expect_command = True  # at body start, a command name is expected

    while pos < n:
        c = command[pos]

        if quote is not None:
            if quote == "'":
                if c == "'":
                    quote = None
                pos += 1
            else:  # double quote
                if c == '\\' and pos + 1 < n:
                    pos += 2
                elif c == '"':
                    quote = None
                    pos += 1
                else:
                    pos += 1
        elif c == "'":
            quote = "'"
            expect_command = False
            pos += 1
        elif c == '"':
            quote = '"'
            expect_command = False
            pos += 1
        elif c == '$' and pos + 1 < n and command[pos + 1] == '(':
            depth += 1
            pos += 2
        elif c == ')' and depth > 0:
            depth -= 1
            pos += 1
        elif depth > 0:
            pos += 1
        elif c == '\n':
            # Newline at depth 0 acts as a command separator
            expect_command = True
            pos += 1
        elif c in (';', '|', '&'):
            expect_command = True
            pos += 1
        elif c == '<' and pos + 1 < n and command[pos + 1] == '<':
            # Heredoc or here-string at depth 0 — skip to avoid false 'done'
            pos, expect_command = _skip_redirect_in_body(
                command, pos, n, expect_command,
            )
            if pos < 0:
                return None  # malformed heredoc
        elif c in (' ', '\t'):
            pos += 1
        elif _at_word('done') and expect_command:
            # Found the closing ``done`` at a command-boundary position.
            body_end = pos
            # Scan backward to trim trailing whitespace from body
            while body_end > body_start and command[body_end - 1] in (' ', '\t', '\n'):
                body_end -= 1
            body = command[body_start:body_end]
            # Reject trailing non-whitespace content after ``done``
            tail_pos = pos + 4  # skip 'done'
            while tail_pos < n and command[tail_pos] in (' ', '\t', '\n'):
                tail_pos += 1
            if tail_pos < n:
                return None  # trailing content after done — not entire-command
            return var_name, in_words, body
        else:
            # Start of a non-boundary word (not done-at-boundary)
            expect_command = False
            pos += 1

    return None  # missing done
