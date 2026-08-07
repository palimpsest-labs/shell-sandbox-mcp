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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .executor import _get_server
from .parser import Expansion

if TYPE_CHECKING:
    from .variables import VariableStore

__all__ = ["Runner", "Result"]


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
                _rc, out = srv._run_background(
                    nodes, self.work_dir, expansion=self.expansion,
                )
                self.ran_any = True
                # Leave prev_rc unchanged — backgrounded exit code is unknown.
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

            # Builtin interception (single-stage non-bg only)
            if not bg and len(remaining_nodes) == 1:
                cmd_node = remaining_nodes[0]

                def _builtin_result(rc_val: int, out_val: str) -> tuple[str, int]:
                    """Format builtin output: prepend 'Exit code: N' when rc != 0."""
                    if rc_val != 0 and not out_val:
                        return f"Exit code: {rc_val}", rc_val
                    return out_val, rc_val

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

            # timeout builtin
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

            # cd builtin
            if not bg and len(nodes) == 1:
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
                # Single-stage builtin already handled above — fall through
                # (should not happen; defense in depth)

            # Build stage environment: base = exported vars; per-stage overrides
            base = store.env_for_subprocess()

            if bg:
                _rc, out = srv._run_background(
                    nodes, self.work_dir, expansion=expansion,
                    shell_env=base, stage_env_overrides=stage_env_overrides if stage_env_overrides else None,
                )
                self.ran_any = True
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
