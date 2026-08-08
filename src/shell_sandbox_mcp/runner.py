"""Shell chain runner — owns mutable per-call state (work_dir, prev_rc).

The ``Runner`` dataclass owns the mutable working-directory and exit-code
state for a single ``shell_run`` invocation and walks a parsed chain
(a list of ``(op, cmd_nodes, backgrounded)`` tuples) applying ``;`` / ``&&``
/ ``||`` short-circuit semantics.

All executor helpers are looked up lazily at call time via
:func:`executor._get_server` so that tests monkeypatching
``server._run_segment`` / ``server._run_pipeline`` / ``server._run_background``
etc. keep firing.
"""

from __future__ import annotations

import fnmatch as _fnmatch
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .executor import _get_server
from .parser import (
    CaseNode,
    CommandNode,
    Expansion,
    ForNode,
    FuncNode,
    GroupNode,
    IfNode,
    ParseError,
    SubshellNode,
    WhileNode,
    program_to_chain,
)
from .config import MAX_LOOP_ITER
from . import config as _config

if TYPE_CHECKING:
    from .variables import VariableStore

__all__ = ["Runner", "Result", "LoopSignal"]


class LoopSignal(Exception):
    """Raised by ``break`` / ``continue`` builtins to signal loop control flow.

    Attributes:
        kind:  ``"break"`` or ``"continue"``.
        count: Nesting level (≥1).  Each loop handler decrements it; the
               loop that catches ``count ≤ 1`` acts on the signal.
    """

    def __init__(self, kind: str, count: int = 1) -> None:
        assert kind in ("break", "continue"), f"LoopSignal kind must be 'break' or 'continue', got {kind!r}"
        super().__init__(kind)
        self.kind = kind
        self.count = max(1, count)


def _expand_for_word(
    raw_word: str,
    work_dir: Path,
    timeout: int,
    depth: int,
    base_env: dict[str, str],
    srv,
    positional: tuple[str, ...] = (),
    *,
    field_split: bool = True,
    ifs: Optional[str] = None,
) -> list[str]:
    """Expand a single word from the for-loop's ``in`` list.

    Uses the existing ``_expand_command`` machinery on a synthetic ``echo``
    command so that ``$()``, ``$VAR``, and glob expansion all apply.  Strips
    the leading ``echo`` (first arg) to recover the expanded word value.
    Returns the original *raw_word* unchanged on any expansion failure.

    Returns a LIST of strings to support ``"$@"`` fan-out: a word
    containing a quoted ``$@`` may expand to multiple argv entries.

    *field_split* controls IFS field splitting on unquoted expansions
    (True for for-loop in-words, False for case-subject).
    """
    if raw_word == "":
        return [""]

    try:
        expanded, expansion, program = srv._expand_command(
            f"__for_expand {raw_word}",
            work_dir, timeout, depth + 1,
            env=base_env,
        )
    except (ParseError, ValueError):
        return [raw_word]  # fallback: literal

    if program is None:
        return [raw_word]

    expansion.positional_tuple = positional
    expansion.ifs = ifs

    chains = program_to_chain(program)
    if not chains or not chains[0][1]:
        return [raw_word]

    from .parser import _extract_from_node
    args, _, _ = _extract_from_node(
        chains[0][1][0], expansion, work_dir, field_split=field_split,
    )
    if not args or args[0] != "__for_expand":
        return [raw_word]
    # The expanded words are everything after the synthetic command name.
    # If there are multiple words (e.g. "$@" fan-out), return them all.
    if len(args) == 1:
        return [""]
    return args[1:]


def _match_case_pattern(subject: str, pattern_text: str, store: "VariableStore") -> bool:
    """Match *subject* against a case-clause *pattern_text*.

    POSIX case pattern matching with ``|`` alternation and fnmatch globbing.
    ``$()`` in patterns is NOT expanded in Phase B.

    Phases:
      1. Walk *pattern_text* char-by-char, expanding ``$VAR``/``${VAR}``
         from *store*, producing a list of ``(char, is_glob_active)`` pairs.
      2. Split on unquoted ``|`` into alternatives.
      3. For each alternative, rebuild with ``*?[`` bracketed-escaped where
         *is_glob_active* is False, then ``fnmatch.fnmatchcase``.
    """

    # Phase 1: expand $VAR, tracking glob-activity per character.
    pairs: list[tuple[str, bool]] = []  # (char, glob_active)
    i, n = 0, len(pattern_text)
    quote: str = ""  # '' = none

    def _try_expand_var(i: int, glob_active: bool) -> int | None:
        """If $VAR / ${VAR} starts at *i*, append expanded chars to *pairs*
        and return the new index.  Returns None when no expansion matches."""
        nxt = pattern_text[i + 1]
        if nxt == '{':
            end = pattern_text.find('}', i + 2)
            if end != -1:
                name = pattern_text[i + 2 : end]
                val = store.get(name) if store else ""
                for ch in val:
                    pairs.append((ch, glob_active))
                return end + 1
            return None
        if _is_valid_var_char(nxt, first=True):
            j = i + 1
            while j < n and _is_valid_var_char(pattern_text[j]):
                j += 1
            name = pattern_text[i + 1 : j]
            val = store.get(name) if store else ""
            for ch in val:
                pairs.append((ch, glob_active))
            return j
        return None

    while i < n:
        c = pattern_text[i]
        if quote == "'":
            if c == "'":
                quote = ""
            else:
                pairs.append((c, False))
            i += 1
            continue
        if quote == '"':
            if c == '"':
                quote = ""
                i += 1
                continue
            if c == '\\' and i + 1 < n and pattern_text[i + 1] in ('"', '$', '\\'):
                pairs.append((pattern_text[i + 1], False))
                i += 2
                continue
            if c == '$' and i + 1 < n:
                new_i = _try_expand_var(i, False)
                if new_i is not None:
                    i = new_i
                    continue
            pairs.append((c, False))
            i += 1
            continue
        # Outside quotes
        if c == '\\' and i + 1 < n:
            pairs.append((pattern_text[i + 1], False))
            i += 2
            continue
        if c == "'":
            quote = "'"
            i += 1
            continue
        if c == '"':
            quote = '"'
            i += 1
            continue
        if c == '$' and i + 1 < n:
            new_i = _try_expand_var(i, True)
            if new_i is not None:
                i = new_i
                continue
        pairs.append((c, True))
        i += 1

    alt_start = 0
    for j, (ch, active) in enumerate(pairs):
        if ch == '|' and active:
            escaped: list[str] = []
            for pc, pa in pairs[alt_start:j]:
                if pa:
                    escaped.append(pc)
                elif pc in '*?[':
                    escaped.append('[' + pc + ']')
                else:
                    escaped.append(pc)
            if _fnmatch.fnmatchcase(subject, "".join(escaped)):
                return True
            alt_start = j + 1
    # Last alternative
    escaped = []
    for pc, pa in pairs[alt_start:]:
        if pa:
            escaped.append(pc)
        elif pc in '*?[':
            escaped.append('[' + pc + ']')
        else:
            escaped.append(pc)
    return _fnmatch.fnmatchcase(subject, "".join(escaped))


def _is_valid_var_char(c: str, *, first: bool = False) -> bool:
    """Return True if *c* is a valid shell variable name character.

    If *first* is True, digits are rejected (POSIX: variable names start
    with alpha or underscore; ``$1`` is a positional parameter, not a
    variable).  When *first* is False (default), digits are allowed for
    subsequent characters.
    """
    if first:
        return c.isalpha() or c == '_'
    return c.isalpha() or c == '_' or c.isdigit()


@dataclass
class Result:
    """Structured result of a ``run_chain`` call.

    Produced when ``run_chain(..., structured=True)``.  ``text`` is identical
    to the default string return, so callers get both the raw joined output
    and the per-stage breakdown.

    Attributes:
        rc:      The final exit code — the rc of the last pipeline that
                 actually ran.  Backgrounded pipelines leave ``prev_rc``
                 unchanged by design, so a trailing ``&`` does not affect it.
        skipped: True if any ``&&``/``||`` chain was short-circuited.
        stages:  Per-stage dicts ``{"command", "output", "rc"}``.  ``rc`` is
                 ``None`` for stages that did not run (skipped chains,
                 backgrounded pipelines).
        text:    The joined output string (same as the default return).
    """

    rc: int
    skipped: bool
    stages: list
    text: str

    def to_dict(self) -> dict:
        """Return the JSON-serializable dict shape for ``structured=True``."""
        return {
            "rc": self.rc,
            "skipped": self.skipped,
            "stages": self.stages,
            "output": self.text,
        }


@dataclass
class Runner:
    """Per-call state and chain-walking logic for ``shell_run``.

    Attributes:
        work_dir:        The current working directory, updated by ``cd``
                         builtins within the same call.
        default_timeout: The call's overall timeout (post clamp), used when a
                         pipeline has no per-pipeline ``timeout N`` builtin.
        expansion:       The expansion context produced by ``_expand_command``.
        variables:       The variable store for ``VAR=value``, ``export``,
                         ``unset``, ``set``, ``shift``, and ``source``/``.``
                         builtins.  Lazily imported to avoid circular imports.
        command:         The original command string (set by ``run_command``
                         for re-expansion).
        prev_rc:         Exit code of the most recently run pipeline.
        ran_any:         Whether at least one pipeline has run.
        skipped:         Whether any ``&&``/``||`` chain was short-circuited.
        outputs:         Accumulated per-pipeline output strings.
        stages:          Per-stage dicts (see :class:`Result`) for structured
                         results.
    """

    work_dir: Path
    default_timeout: int
    expansion: Optional[Expansion] = None
    variables: Optional["VariableStore"] = None
    command: str = ""
    prev_rc: int = 0
    ran_any: bool = False
    skipped: bool = False
    outputs: list[str] = field(default_factory=list)
    stages: list = field(default_factory=list)
    _in_loop: bool = False  # True when Runner is executing inside a for/while/until body
    _loop_depth: int = 0    # Tracks loop nesting depth for LoopSignal propagation

    def run_chain(self, chains: list, timeout: int, structured: bool = False) -> "str | Result":
        """Walk a parsed chain and return the joined output string.

        *chains* is the list of ``(op, cmd_nodes, backgrounded)`` tuples from
        ``program_to_chain``.  A length-1 chain produces identical output to
        the former single-command fast path.  Returns ``"(no output)"`` when
        nothing was produced.

        When *structured* is True, returns a :class:`Result` carrying the
        final rc, whether any chain was skipped, a per-stage breakdown, and
        the same joined text as the default string return.  The default
        (False) return is byte-for-byte identical to the previous behaviour.
        """
        srv = _get_server()

        for op, cmd_nodes, backgrounded in chains:
            joined = srv._serialize_pipeline_from_cmds(cmd_nodes)
            if op == "&&" and self.ran_any and self.prev_rc != 0:
                self.skipped = True
                skip_text = (
                    f"(skipped: previous command exited {self.prev_rc}) — {joined}"
                )
                self.outputs.append(skip_text)
                self.stages.append({"command": joined, "output": skip_text, "rc": None})
                continue
            if op == "||" and self.ran_any and self.prev_rc == 0:
                self.skipped = True
                skip_text = "(skipped: previous command succeeded) — " + joined
                self.outputs.append(skip_text)
                self.stages.append({"command": joined, "output": skip_text, "rc": None})
                continue

            # timeout builtin: intercept before cd/allowlist dispatch so the
            # per-pipeline timeout override applies to the correct pipeline.
            nodes, eff_to, terr = srv._apply_timeout_builtin(
                cmd_nodes, self.expansion, backgrounded, timeout,
                work_dir=self.work_dir,
            )
            if terr is not None:
                self.outputs.append(terr)
                self.stages.append({"command": joined, "output": terr, "rc": 1})
                self.prev_rc = 1
                self.ran_any = True
                continue

            # cd builtin: intercept single-command non-backgrounded pipelines
            # before allowlist dispatch so the directory change applies to
            # subsequent segments of the same shell_run call.
            if not backgrounded and len(nodes) == 1:
                new_dir, cd_err = srv._try_cd(nodes[0], self.work_dir, self.expansion)
                if cd_err is not None:
                    self.outputs.append(cd_err)
                    self.stages.append({"command": joined, "output": cd_err, "rc": 1})
                    self.prev_rc = 1
                    self.ran_any = True
                    continue
                if new_dir is not None:
                    self.work_dir = new_dir
                    self.prev_rc = 0
                    self.ran_any = True
                    self.stages.append({"command": joined, "output": "", "rc": 0})
                    continue

            if backgrounded:
                _rc, out, _pid = srv._run_background(
                    nodes, self.work_dir, expansion=self.expansion,
                )
                self.ran_any = True
                # Leave prev_rc unchanged — backgrounded exit code is unknown.
                if self.variables is not None:
                    self.variables.last_bg_pid = _pid
                self.stages.append({"command": joined, "output": out, "rc": None})
            elif len(nodes) == 1:
                rc, out = srv._run_segment(
                    nodes[0], self.work_dir, eff_to, expansion=self.expansion,
                )
                self.prev_rc = rc
                self.ran_any = True
                self.stages.append({"command": joined, "output": out, "rc": rc})
            else:
                rc, out = srv._run_pipeline(
                    nodes, self.work_dir, eff_to, expansion=self.expansion,
                )
                self.prev_rc = rc
                self.ran_any = True
                self.stages.append({"command": joined, "output": out, "rc": rc})
            if out:
                self.outputs.append(out)

        if not self.outputs:
            text = "(no output)"
        else:
            text = "\n".join(self.outputs)

        if structured:
            return Result(
                rc=self.prev_rc,
                skipped=self.skipped,
                stages=self.stages,
                text=text,
            )
        return text

    def run_command(
        self, command: str, timeout: int, structured: bool = False, depth: int = 0,
    ) -> "str | Result":
        """Per-chain re-expansion path for variable assignment + builtins.

        Unlike :meth:`run_chain` (which walks a pre-parsed AST with a single
        expansion context), this method re-expands each chain segment with the
        current ``variables`` store so ``$VAR`` references pick up earlier
        assignments.  Called by the ``shell_run`` tool when the command
        contains ``VAR=value`` or any of the new builtins
        (``export``, ``unset``, ``set``, ``shift``, ``source``/``.``).

        *depth* is the nesting level (used by ``source`` to detect recursion).
        """
        from .parser import split_chains, program_to_chain, ParseError
        from .variables import VariableStore
        from .builtins import (
            _split_assignment_prefix,
            _try_export,
            _try_unset,
            _try_set,
            _try_shift,
            _try_source,
            _try_break_continue,
        )

        srv = _get_server()

        # Ensure we have a variable store.
        store = self.variables if self.variables is not None else VariableStore()
        self.variables = store

        self.command = command

        segments = split_chains(command)
        if not segments:
            text = "(no output)"
            if structured:
                return Result(rc=0, skipped=False, stages=[], text=text)
            return text

        for op, seg_text, bg in segments:
            # && / || short-circuit (mirror run_chain)
            joined = seg_text  # approximate — the real serialized form is computed later
            if op == "&&" and self.ran_any and self.prev_rc != 0:
                self.skipped = True
                skip_text = (
                    f"(skipped: previous command exited {self.prev_rc}) — {joined}"
                )
                self.outputs.append(skip_text)
                self.stages.append({"command": joined, "output": skip_text, "rc": None})
                continue
            if op == "||" and self.ran_any and self.prev_rc == 0:
                self.skipped = True
                skip_text = "(skipped: previous command succeeded) — " + joined
                self.outputs.append(skip_text)
                self.stages.append({"command": joined, "output": skip_text, "rc": None})
                continue

            # Re-expand with the current variable store.
            store.last_rc = self.prev_rc
            env = store.env_for_expansion()
            try:
                expanded, expansion, program = srv._expand_command(
                    seg_text, self.work_dir, timeout, depth, env=env,
                )
            except (ParseError, ValueError) as e:
                self.outputs.append(str(e))
                self.stages.append({"command": joined, "output": str(e), "rc": 1})
                self.prev_rc = 1
                self.ran_any = True
                continue

            if expansion is not None:
                expansion.positional_tuple = store.positional
                expansion.ifs = store.variables.get("IFS")

            if program is None:
                self.outputs.append("Command parse error.")
                self.stages.append({"command": joined, "output": "Command parse error.", "rc": 1})
                self.prev_rc = 1
                self.ran_any = True
                continue

            chains = program_to_chain(program)
            if not chains:
                continue  # empty body — no output, rc unchanged

            # Take the first chain only (run_command processes one segment at a time)
            _op2, cmd_nodes, _bg2 = chains[0]
            if not cmd_nodes:
                continue

            # Compound command dispatch (if/while/until/for).
            # Must be checked BEFORE assignment-prefix / builtin processing
            # because compounds are not regular CommandNodes.
            if not bg and len(cmd_nodes) == 1:
                node = cmd_nodes[0]
                if isinstance(node, (IfNode, WhileNode, ForNode, CaseNode, SubshellNode, FuncNode, GroupNode)):
                    try:
                        rc, out = self._run_compound(
                            node, store, self.work_dir, timeout, depth,
                            joined_for_stage=joined,
                        )
                    except LoopSignal as sig:
                        if self._loop_depth > 0:
                            raise
                        out = f"{sig.kind}: only meaningful in a `for`, `while`, or `until` loop"
                        rc = 1
                    self.prev_rc = rc
                    self.ran_any = True
                    self.stages.append({"command": joined, "output": out, "rc": rc})
                    if out:
                        self.outputs.append(out)
                    continue

            # Per-stage detection: assignment prefix, builtins, cd, timeout
            prefix_maps: list[dict[str, str]] = []    # all prefixes (for pure-assignment branch)
            remaining_nodes: list = []                 # stages with commands to run
            stage_env_overrides: list[dict[str, str]] = []  # per-stage env overrides (aligned with remaining_nodes)
            pure_assignments: list[dict[str, str]] = []     # prefixes from pure-assignment stages
            all_pure_assignment = True

            for node in cmd_nodes:
                prefix, remaining, err = _split_assignment_prefix(
                    node, expansion, self.work_dir,
                )
                if err is not None:
                    self.outputs.append(err)
                    self.stages.append({"command": joined, "output": err, "rc": 1})
                    self.prev_rc = 1
                    self.ran_any = True
                    break
                if prefix is not None:
                    prefix_maps.append(prefix)
                    if remaining is not None:
                        # env-prefix: VAR=x cmd — prefix goes to stage_env_overrides only
                        remaining_nodes.append(remaining)
                        stage_env_overrides.append(dict(prefix))
                        all_pure_assignment = False
                    else:
                        # pure assignment: VAR=x with no command
                        pure_assignments.append(dict(prefix))
                else:
                    # No assignment prefix at all for this stage
                    remaining_nodes.append(node)
                    stage_env_overrides.append({})
                    all_pure_assignment = False

            # If we broke out of the loop due to an error, continue
            if self.ran_any and self.prev_rc == 1 and self.stages:
                continue

            # All pure assignment? Handle separately.
            if all_pure_assignment and not remaining_nodes:
                if len(cmd_nodes) == 1:
                    # Single-segment pure assignment
                    for k, v in prefix_maps[0].items():
                        store.set_local(k, v)
                    self.prev_rc = 0
                    self.ran_any = True
                    self.stages.append({"command": joined, "output": "", "rc": 0})
                else:
                    # Multi-stage pure assignment → error
                    err = "assignment prefix requires a command in a pipeline"
                    self.outputs.append(err)
                    self.stages.append({"command": joined, "output": err, "rc": 1})
                    self.prev_rc = 1
                    self.ran_any = True
                continue

            # Merge ONLY pure-assignment prefixes into the store.
            # Env-prefix assignments (VAR=x cmd) do NOT persist past cmd.
            for pm in pure_assignments:
                for k, v in pm.items():
                    store.set_local(k, v)

            # timeout builtin — strip `timeout N` BEFORE builtin/cd/allowlist
            # dispatch so a single-stage `timeout N <builtin>` resolves the
            # builtin (not `timeout`).
            nodes, eff_to, terr = srv._apply_timeout_builtin(
                remaining_nodes, expansion, bg, timeout,
                work_dir=self.work_dir,
            )
            if terr is not None:
                self.outputs.append(terr)
                self.stages.append({"command": joined, "output": terr, "rc": 1})
                self.prev_rc = 1
                self.ran_any = True
                continue

            # cd builtin — skip compounds (they are handled later).
            if not bg and len(nodes) == 1 and isinstance(nodes[0], CommandNode):
                new_dir, cd_err = srv._try_cd(nodes[0], self.work_dir, expansion)
                if cd_err is not None:
                    self.outputs.append(cd_err)
                    self.stages.append({"command": joined, "output": cd_err, "rc": 1})
                    self.prev_rc = 1
                    self.ran_any = True
                    continue
                if new_dir is not None:
                    self.work_dir = new_dir
                    self.prev_rc = 0
                    self.ran_any = True
                    self.stages.append({"command": joined, "output": "", "rc": 0})
                    continue

            # Builtin interception (single-stage non-bg only)
            if not bg and len(nodes) == 1:
                cmd_node = nodes[0]

                def _builtin_result(rc_val: int, out_val: str) -> tuple[str, int]:
                    """Format builtin output: prepend 'Exit code: N' when rc != 0."""
                    if rc_val != 0 and not out_val:
                        return f"Exit code: {rc_val}", rc_val
                    return out_val, rc_val

                # --- break / continue ---
                brk_kind, brk_count, brk_rc = _try_break_continue(
                    cmd_node, expansion, self.work_dir,
                )
                if brk_kind is not None:
                    if brk_rc != 0:
                        # Bad argument — emit diagnostic with rc=1, loop continues
                        out, rc = _builtin_result(
                            1, f"{brk_kind}: invalid argument"
                        )
                        if out:
                            self.outputs.append(out)
                        self.stages.append({"command": joined, "output": out, "rc": rc})
                        self.prev_rc = rc
                        self.ran_any = True
                        continue
                    if self._in_loop:
                        raise LoopSignal(brk_kind, brk_count)
                    else:
                        msg = f"{brk_kind}: only meaningful in a `for`, `while`, or `until` loop"
                        out, rc = _builtin_result(1, msg)
                        if out:
                            self.outputs.append(out)
                        self.stages.append({"command": joined, "output": out, "rc": rc})
                        self.prev_rc = rc
                        self.ran_any = True
                        continue

                # --- function invocation dispatch ---
                args, _, _ = srv._extract_redirects(cmd_node, expansion, self.work_dir)
                if args and args[0] in store.functions:
                    if store.func_depth >= _config.MAX_FUNC_DEPTH:
                        err_msg = (
                            f"function recursion depth limit"
                            f" ({_config.MAX_FUNC_DEPTH}) exceeded"
                        )
                        out, rc = _builtin_result(1, err_msg)
                        if out:
                            self.outputs.append(out)
                        self.stages.append({"command": joined, "output": out, "rc": rc})
                        self.prev_rc = rc
                        self.ran_any = True
                        continue
                    rc, out = self._run_function_call(
                        store.functions[args[0]], args[1:], args[0],
                        store, self.work_dir, timeout, depth,
                    )
                    out, rc = _builtin_result(rc, out)
                    if out:
                        self.outputs.append(out)
                    self.stages.append({"command": joined, "output": out, "rc": rc})
                    self.prev_rc = rc
                    self.ran_any = True
                    continue

                # Try export
                handled, out, rc = _try_export(cmd_node, expansion, self.work_dir, store)
                if handled:
                    out, rc = _builtin_result(rc, out)
                    if out:
                        self.outputs.append(out)
                    self.stages.append({"command": joined, "output": out, "rc": rc})
                    self.prev_rc = rc
                    self.ran_any = True
                    continue

                # Try unset
                handled, out, rc = _try_unset(cmd_node, expansion, self.work_dir, store)
                if handled:
                    out, rc = _builtin_result(rc, out)
                    if out:
                        self.outputs.append(out)
                    self.stages.append({"command": joined, "output": out, "rc": rc})
                    self.prev_rc = rc
                    self.ran_any = True
                    continue

                # Try set
                handled, out, rc = _try_set(cmd_node, expansion, self.work_dir, store)
                if handled:
                    out, rc = _builtin_result(rc, out)
                    if out:
                        self.outputs.append(out)
                    self.stages.append({"command": joined, "output": out, "rc": rc})
                    self.prev_rc = rc
                    self.ran_any = True
                    continue

                # Try shift
                handled, out, rc = _try_shift(cmd_node, expansion, self.work_dir, store)
                if handled:
                    out, rc = _builtin_result(rc, out)
                    if out:
                        self.outputs.append(out)
                    self.stages.append({"command": joined, "output": out, "rc": rc})
                    self.prev_rc = rc
                    self.ran_any = True
                    continue

                # Try source / .
                handled, out, rc = _try_source(
                    cmd_node, expansion, self.work_dir, store, timeout, depth,
                )
                if handled:
                    out, rc = _builtin_result(rc, out)
                    if out:
                        self.outputs.append(out)
                    self.stages.append({"command": joined, "output": out, "rc": rc})
                    self.prev_rc = rc
                    self.ran_any = True
                    continue

            # Compound command dispatch (if/while/until/for).
            # Compounds must be the sole element of their pipeline and
            # cannot be backgrounded.  They route through the stateful
            # execution path so they have access to the VariableStore.
            #
            # This is the second dispatch point (the first is above, at
            # the top of the segment-processing loop).  The first dispatch
            # catches compounds immediately from cmd_nodes before any
            # prefix/builtin/cd processing.  This second dispatch handles
            # the (currently unreachable but defensive) case where a
            # compound survives through prefix-extraction and reappears
            # in *nodes*.  Consolidation would require restructuring the
            # entire prefix/builtin/cd block; keeping both is simpler and
            # harmless.
            if not bg and len(nodes) == 1:
                node = nodes[0]
                if isinstance(node, (IfNode, WhileNode, ForNode, CaseNode, SubshellNode, FuncNode, GroupNode)):
                    try:
                        rc, out = self._run_compound(
                            node, store, self.work_dir, timeout, depth,
                            joined_for_stage=joined,
                        )
                    except LoopSignal as sig:
                        if self._loop_depth > 0:
                            raise
                        out = f"{sig.kind}: only meaningful in a `for`, `while`, or `until` loop"
                        rc = 1
                    self.prev_rc = rc
                    self.ran_any = True
                    self.stages.append({"command": joined, "output": out, "rc": rc})
                    if out:
                        self.outputs.append(out)
                    continue

            # Per-stage builtin handling in multi-stage pipelines.
            # Single-stage builtins are handled above (byte-for-byte preserved).
            # Pure-subprocess pipelines fall through to the bg/single/pipeline
            # dispatch below.
            from .builtins import _classify_builtin
            kinds = [_classify_builtin(n, expansion, self.work_dir) for n in nodes]
            if any(kinds):
                if bg:
                    err = "builtin not supported in backgrounded pipeline (&)"
                    self.outputs.append(err)
                    self.stages.append({"command": joined, "output": err, "rc": 1})
                    self.prev_rc = 1
                    self.ran_any = True
                    continue
                if len(nodes) > 1:
                    rc, out = self._run_mixed_pipeline(
                        nodes, kinds, stage_env_overrides, expansion, eff_to, timeout, depth,
                    )
                    self.prev_rc = rc
                    self.ran_any = True
                    self.stages.append({"command": joined, "output": out, "rc": rc})
                    if out:
                        self.outputs.append(out)
                    continue
                # Single-stage builtin reaching this point (defense in depth —
                # normally handled by the single-stage block above, or routed to
                # run_command by the lex-gate).  Dispatch it properly instead of
                # falling through to subprocess dispatch.
                out, rc = self._exec_pipeline_builtin(
                    kinds[0], nodes[0], expansion, timeout, depth,
                )
                self.prev_rc = rc
                self.ran_any = True
                self.stages.append({"command": joined, "output": out, "rc": rc})
                if out:
                    self.outputs.append(out)
                continue

            # Build stage environment: base = exported vars; per-stage overrides
            base = store.env_for_subprocess()

            if bg:
                _rc, out, _pid = srv._run_background(
                    nodes, self.work_dir, expansion=expansion,
                    shell_env=base, stage_env_overrides=stage_env_overrides if stage_env_overrides else None,
                )
                self.ran_any = True
                store.last_bg_pid = _pid
                self.stages.append({"command": joined, "output": out, "rc": None})
            elif len(nodes) == 1:
                rc, out = srv._run_segment(
                    nodes[0], self.work_dir, eff_to, expansion=expansion,
                    shell_env=base,
                    stage_env_overrides=[stage_env_overrides[0]] if stage_env_overrides else None,
                )
                self.prev_rc = rc
                self.ran_any = True
                self.stages.append({"command": joined, "output": out, "rc": rc})
            else:
                rc, out = srv._run_pipeline(
                    nodes, self.work_dir, eff_to, expansion=expansion,
                    shell_env=base, stage_env_overrides=stage_env_overrides if stage_env_overrides else None,
                )
                self.prev_rc = rc
                self.ran_any = True
                self.stages.append({"command": joined, "output": out, "rc": rc})
            if out:
                self.outputs.append(out)

        if not self.outputs:
            text = "(no output)"
        else:
            text = "\n".join(self.outputs)

        if structured:
            return Result(
                rc=self.prev_rc,
                skipped=self.skipped,
                stages=self.stages,
                text=text,
            )
        return text

    def _run_mixed_pipeline(self, nodes, kinds, stage_env_overrides, expansion,
                            eff_to, timeout, depth):
        """Walk *nodes* left-to-right, grouping consecutive subprocess stages
        into mini-pipelines separated by builtin barriers.

        Returns ``(final_rc, output_string)``.

        *eff_to* is the per-pipeline effective timeout (after
        :func:`_apply_timeout_builtin` stripping); *timeout* is the original
        call timeout (used by ``source``).
        """
        srv = _get_server()
        store = self.variables
        n = len(nodes)
        pending_stdin_bytes: Optional[bytes] = None
        collected_stderr = bytearray()
        final_stdout_bytes = b""
        final_rc = 0
        last_was_builtin = False
        last_builtin_out = ""
        base = store.env_for_subprocess()

        i = 0
        while i < n:
            if kinds[i] is None:
                # Subprocess stage(s) — collect consecutive subprocess nodes.
                j = i + 1
                while j < n and kinds[j] is None:
                    j += 1
                sub_nodes = nodes[i:j]
                sub_overrides = (stage_env_overrides[i:j]
                                 if stage_env_overrides else None)
                is_last_run = (j == n)

                rc, stdout_b, stderr_b, report = srv._run_pipeline_core(
                    sub_nodes, self.work_dir, eff_to, expansion=expansion,
                    shell_env=base, stage_env_overrides=sub_overrides,
                    injected_first_stdin_bytes=pending_stdin_bytes,
                    start_index=i,
                )
                final_rc = rc
                if stderr_b:
                    collected_stderr.extend(stderr_b)
                if is_last_run:
                    final_stdout_bytes = stdout_b
                # else: intermediate subprocess stdout is consumed by
                # _launch_pipeline_foreground (captured to PIPE internally,
                # no block); we just drop it here.

                pending_stdin_bytes = None
                last_was_builtin = False
                i = j
            else:
                # Builtin stage.
                name = kinds[i]
                out, rc = self._exec_pipeline_builtin(
                    name, nodes[i], expansion, timeout, depth,
                )
                pending_stdin_bytes = (out or "").encode("utf-8", errors="replace")
                final_rc = rc
                last_was_builtin = True
                last_builtin_out = out
                i += 1

        # Compose final output string.
        if last_was_builtin:
            # Builtin-last: output already has "Exit code: N" applied by
            # _exec_pipeline_builtin.  Append collected stderr if any.
            out = last_builtin_out
            if collected_stderr:
                stderr_text = collected_stderr.decode("utf-8", errors="replace")
                out = out + "\n" + stderr_text if out else stderr_text
            return final_rc, out
        else:
            # Subprocess-last: use standard _format_output.
            out = srv._format_output(
                final_rc, final_stdout_bytes, bytes(collected_stderr), [],
            )
            return final_rc, out

    def _exec_pipeline_builtin(self, name, node, expansion, timeout, depth):
        """Execute a single builtin stage in a pipeline.

        Returns ``(output_string, rc)`` with ``"Exit code: N"`` formatting
        already applied (same as the single-stage builtin path).
        """
        from .builtins import (
            _try_export,
            _try_unset,
            _try_set,
            _try_shift,
            _try_source,
        )
        store = self.variables

        def _builtin_result(rc_val: int, out_val: str) -> tuple[str, int]:
            """Format builtin output: prepend 'Exit code: N' when rc != 0."""
            if rc_val != 0 and not out_val:
                return f"Exit code: {rc_val}", rc_val
            return out_val or "", rc_val

        if name == "export":
            handled, out, rc = _try_export(node, expansion, self.work_dir, store)
        elif name == "unset":
            handled, out, rc = _try_unset(node, expansion, self.work_dir, store)
        elif name == "set":
            handled, out, rc = _try_set(node, expansion, self.work_dir, store)
        elif name == "shift":
            handled, out, rc = _try_shift(node, expansion, self.work_dir, store)
        elif name in ("source", "."):
            handled, out, rc = _try_source(
                node, expansion, self.work_dir, store, timeout, depth,
            )
        else:
            return f"unknown builtin: {name}", 1

        if not handled:
            return f"builtin not handled: {name}", 1

        return _builtin_result(rc, out)

    # ------------------------------------------------------------------
    # Compound command execution (if / while / until / for)
    # ------------------------------------------------------------------

    def _run_compound(
        self,
        node: "IfNode | WhileNode | ForNode | CaseNode | SubshellNode | FuncNode | GroupNode",
        store: "VariableStore",
        work_dir: "Path",
        timeout: int,
        depth: int,
        *,
        joined_for_stage: str,
    ) -> tuple[int, str]:
        """Execute a compound command and return ``(rc, output)``.

        *node* is one of :class:`IfNode`, :class:`WhileNode`,
        :class:`ForNode`, :class:`CaseNode`, :class:`SubshellNode`,
        :class:`FuncNode`, or :class:`GroupNode`.  The body text is
        re-parsed and executed with a fresh per-body :class:`Runner` that
        shares *store* (so variable mutations persist) but uses a snapshot
        of *work_dir* (so ``cd`` inside a body does NOT leak).
        """
        if isinstance(node, IfNode):
            return self._run_if(node, store, work_dir, timeout, depth,
                                joined_for_stage=joined_for_stage)
        elif isinstance(node, WhileNode):
            return self._run_while_until(node, store, work_dir, timeout, depth,
                                         joined_for_stage=joined_for_stage)
        elif isinstance(node, ForNode):
            return self._run_for(node, store, work_dir, timeout, depth,
                                 joined_for_stage=joined_for_stage)
        elif isinstance(node, CaseNode):
            return self._run_case(node, store, work_dir, timeout, depth,
                                  joined_for_stage=joined_for_stage)
        elif isinstance(node, SubshellNode):
            return self._run_subshell(node, store, work_dir, timeout, depth,
                                      joined_for_stage=joined_for_stage)
        elif isinstance(node, FuncNode):
            # Function definition: store body, rc=0, no output.
            store.functions[node.name] = node.body
            return 0, ""
        elif isinstance(node, GroupNode):
            return self._run_group(node, store, work_dir, timeout, depth,
                                   joined_for_stage=joined_for_stage)
        else:
            return 1, f"unknown compound: {type(node).__name__}"

    def _run_if(
        self, node: IfNode, store, work_dir, timeout, depth,
        *, joined_for_stage: str,
    ) -> tuple[int, str]:
        """Execute an if/elif/else/fi construct."""
        outputs: list[str] = []
        for branch in node.branches:
            rc, out = self._run_body(
                branch.cond, store, work_dir, timeout, depth,
            )
            if out:
                outputs.append(out)
            if rc == 0:
                # Condition true — run this branch's body and return.
                brc, bout = self._run_body(
                    branch.body, store, work_dir, timeout, depth,
                )
                if bout:
                    outputs.append(bout)
                return brc, "\n".join(outputs) if outputs else ""
            # else: fall through to elif/else
        # No branch matched — run else_body if present.
        if node.else_body is not None:
            rc, out = self._run_body(
                node.else_body, store, work_dir, timeout, depth,
            )
            if out:
                outputs.append(out)
            return rc, "\n".join(outputs) if outputs else ""
        return 0, "\n".join(outputs) if outputs else ""

    def _run_while_until(
        self, node: WhileNode, store, work_dir, timeout, depth,
        *, joined_for_stage: str,
    ) -> tuple[int, str]:
        """Execute a while/until loop with MAX_LOOP_ITER cap."""
        outputs: list[str] = []
        last_rc = 0
        iterations = 0

        self._loop_depth += 1
        self._in_loop = True
        reraise_sig = None
        try:
            while iterations < MAX_LOOP_ITER:
                try:
                    rc, out = self._run_body(
                        node.cond, store, work_dir, timeout, depth,
                    )
                except LoopSignal as sig:
                    if sig.kind == "break":
                        if sig.count <= 1:
                            last_rc = 0
                            break  # exit the while loop
                        else:
                            reraise_sig = LoopSignal("break", sig.count - 1)
                            break
                    else:  # continue
                        if sig.count <= 1:
                            last_rc = 0
                            iterations += 1
                            continue  # next iteration
                        else:
                            reraise_sig = LoopSignal("continue", sig.count - 1)
                            break
                if reraise_sig is not None:
                    break  # exit while, re-raise handled after cleanup
                if out:
                    outputs.append(out)
                # while: enter body if rc==0; until: enter body if rc!=0
                enter = (rc == 0) if not node.until else (rc != 0)
                if not enter:
                    last_rc = rc
                    break
                # Enter body
                try:
                    brc, bout = self._run_body(
                        node.body, store, work_dir, timeout, depth,
                    )
                except LoopSignal as sig:
                    if sig.kind == "break":
                        if sig.count <= 1:
                            last_rc = 0
                            break  # exit the while loop
                        else:
                            reraise_sig = LoopSignal("break", sig.count - 1)
                            break
                    else:  # continue
                        if sig.count <= 1:
                            last_rc = 0
                            iterations += 1
                            continue  # next iteration
                        else:
                            reraise_sig = LoopSignal("continue", sig.count - 1)
                            break
                if reraise_sig is not None:
                    break  # exit while, re-raise handled after cleanup
                if bout:
                    outputs.append(bout)
                last_rc = brc
                iterations += 1
        except Exception:
            # Defensive cleanup: only reached for unexpected errors
            # (LoopSignal is caught by the inner handler above).
            self._loop_depth -= 1
            self._in_loop = False
            raise
        else:
            self._loop_depth -= 1
            self._in_loop = False
            if reraise_sig is not None:
                raise reraise_sig

        if iterations >= MAX_LOOP_ITER:
            outputs.append(
                f"loop exceeded MAX_LOOP_ITER ({MAX_LOOP_ITER}) iterations"
            )
            return 1, "\n".join(outputs) if outputs else ""

        return last_rc, "\n".join(outputs) if outputs else ""

    def _run_for(
        self, node: ForNode, store, work_dir, timeout, depth,
        *, joined_for_stage: str,
    ) -> tuple[int, str]:
        """Execute an AST-native for-loop.

        The loop variable persists after the loop (matching POSIX semantics):
        it is left at the value from the last iteration, or unset if the
        ``in`` word list was empty.
        """
        srv = _get_server()
        from .config import _base_env

        base_env = _base_env()

        # Expand in-words through the existing expansion machinery.
        in_words: list[str] = []
        for raw_word in node.in_words:
            expanded = _expand_for_word(raw_word, work_dir, timeout, depth,
                                        base_env, srv, positional=store.positional,
                                        ifs=store.variables.get("IFS"))
            in_words.extend(expanded)

        outputs: list[str] = []
        last_rc = 0

        self._loop_depth += 1
        self._in_loop = True
        reraise_sig = None
        try:
            for word_value in in_words:
                store.set_local(node.var_name, word_value)
                try:
                    rc, out = self._run_body(
                        node.body, store, work_dir, timeout, depth,
                    )
                except LoopSignal as sig:
                    if sig.kind == "break":
                        if sig.count <= 1:
                            last_rc = 0
                            break  # exit the for loop
                        else:
                            reraise_sig = LoopSignal("break", sig.count - 1)
                            break
                    else:  # continue
                        if sig.count <= 1:
                            last_rc = 0
                            continue  # next iteration
                        else:
                            reraise_sig = LoopSignal("continue", sig.count - 1)
                            break
                if out:
                    outputs.append(out)
                last_rc = rc
        except Exception:
            # Defensive cleanup: only reached for unexpected errors
            # (LoopSignal is caught by the inner handler above).
            self._loop_depth -= 1
            self._in_loop = False
            raise
        else:
            self._loop_depth -= 1
            self._in_loop = False
            if reraise_sig is not None:
                raise reraise_sig

        if not outputs:
            return last_rc, ""
        return last_rc, "\n".join(outputs)

    def _run_body(
        self,
        body_text: str,
        store: "VariableStore",
        work_dir: "Path",
        timeout: int,
        depth: int,
    ) -> tuple[int, str]:
        """Re-parse *body_text* and execute it with a fresh sub-Runner.

        The sub-Runner SHARES *store* (so variable mutations persist) but
        gets a snapshot of *work_dir* (so ``cd`` inside a body doesn't leak).
        """
        srv = _get_server()

        if not body_text.strip():
            return 0, ""

        store.last_rc = self.prev_rc
        env = store.env_for_expansion()
        try:
            expanded, expansion, program = srv._expand_command(
                body_text, work_dir, timeout, depth + 1,
                env=env,
            )
        except (ParseError, ValueError) as e:
            return 1, str(e)

        if program is None:
            return 1, "Command parse error."

        expansion.positional_tuple = store.positional
        expansion.ifs = store.variables.get("IFS")

        chains = program_to_chain(program)
        if not chains:
            return 0, ""

        # Create a fresh Runner that SHARES the variable store.
        sub_runner = Runner(
            work_dir=Path(str(work_dir)),  # snapshot
            default_timeout=timeout,
            expansion=expansion,
            variables=store,
        )
        sub_runner._in_loop = self._in_loop
        sub_runner._loop_depth = self._loop_depth
        result = sub_runner.run_command(
            body_text, timeout, depth=depth + 1, structured=True,
        )
        # structured=True guarantees a Result object.
        assert isinstance(result, Result)
        return result.rc, result.text

    # ------------------------------------------------------------------
    # case / esac
    # ------------------------------------------------------------------

    def _run_case(
        self,
        node: CaseNode,
        store: "VariableStore",
        work_dir: "Path",
        timeout: int,
        depth: int,
        *,
        joined_for_stage: str,
    ) -> tuple[int, str]:
        """Execute a ``case`` construct: match subject against patterns.

        The subject word is expanded via :func:`_expand_for_word`.  Patterns
        are tried in order; the first matching clause's body runs.  If no
        clause matches, returns ``(0, "")`` (POSIX).
        """
        srv = _get_server()
        # Use the store's full env (including user-set variables) for
        # subject expansion, so $VAR in the subject expands correctly.
        env = store.env_for_expansion()

        # Expand subject through existing machinery (NO field splitting).
        expanded = _expand_for_word(node.subject, work_dir, timeout, depth,
                                    env, srv, positional=store.positional,
                                    field_split=False,
                                    ifs=store.variables.get("IFS"))
        subject = " ".join(expanded) if expanded else ""

        for clause in node.clauses:
            if _match_case_pattern(subject, clause.pattern, store):
                return self._run_body(
                    clause.body, store, work_dir, timeout, depth,
                )

        # No match — POSIX: exit 0, no output.
        return 0, ""

    # ------------------------------------------------------------------
    # subshell ( ... )
    # ------------------------------------------------------------------

    def _run_subshell(
        self,
        node: SubshellNode,
        store: "VariableStore",
        work_dir: "Path",
        timeout: int,
        depth: int,
        *,
        joined_for_stage: str,
    ) -> tuple[int, str]:
        """Execute a subshell ``(command; ...)`` with variable isolation.

        Creates a deep copy of *store* so assignments inside the subshell
        do NOT leak to the parent.  ``cd`` does not leak either (already
        handled by ``_run_body``'s work_dir snapshot).
        """
        # Deep-copy the variable store for isolation.
        isolated_store = replace(
            store,
            variables=dict(store.variables),
            exported=set(store.exported),
            functions=dict(store.functions),
            positional=store.positional,
            script_name=store.script_name,
            func_depth=store.func_depth,
        )
        try:
            return self._run_body(node.body, isolated_store, work_dir, timeout, depth)
        except LoopSignal as sig:
            return 1, f"{sig.kind}: only meaningful in a `for`, `while`, or `until` loop"

    # ------------------------------------------------------------------
    # function call
    # ------------------------------------------------------------------

    def _run_function_call(
        self,
        body_text: str,
        args: list[str],
        func_name: str,
        store: "VariableStore",
        work_dir: "Path",
        timeout: int,
        depth: int,
    ) -> tuple[int, str]:
        """Execute a user-defined function call.

        Saves/restores positional parameters, bumps func_depth, and
        executes the function body.  ``$0`` (script_name) is NOT changed.
        """
        saved_positional = store.positional
        store.positional = tuple(args)
        store.func_depth += 1
        try:
            try:
                return self._run_body(body_text, store, work_dir, timeout, depth)
            except LoopSignal as sig:
                return 1, f"{sig.kind}: only meaningful in a `for`, `while`, or `until` loop"
        finally:
            store.positional = saved_positional
            store.func_depth -= 1

    # ------------------------------------------------------------------
    # brace group { ... }
    # ------------------------------------------------------------------

    def _run_group(
        self,
        node: GroupNode,
        store: "VariableStore",
        work_dir: "Path",
        timeout: int,
        depth: int,
        *,
        joined_for_stage: str,
    ) -> tuple[int, str]:
        """Execute a brace group ``{ command; ...; }``.

        No variable isolation — the body runs with the same store.
        ``cd`` does not leak (handled by ``_run_body``'s work_dir snapshot).
        """
        return self._run_body(node.body, store, work_dir, timeout, depth)
