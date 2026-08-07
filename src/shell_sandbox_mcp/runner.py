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

from dataclasses import dataclass, field
from pathlib import Path

from .executor import _get_server
from .parser import Expansion

__all__ = ["Runner"]


@dataclass
class Runner:
    """Per-call state and chain-walking logic for ``shell_run``.

    Attributes:
        work_dir:        The current working directory, updated by ``cd``
                         builtins within the same call.
        default_timeout: The call's overall timeout (post clamp), used when a
                         pipeline has no per-pipeline ``timeout N`` builtin.
        expansion:       The expansion context produced by ``_expand_command``.
        prev_rc:         Exit code of the most recently run pipeline.
        ran_any:         Whether at least one pipeline has run.
        outputs:         Accumulated per-pipeline output strings.
    """

    work_dir: Path
    default_timeout: int
    expansion: Expansion
    prev_rc: int = 0
    ran_any: bool = False
    outputs: list[str] = field(default_factory=list)

    def run_chain(self, chains: list, timeout: int) -> str:
        """Walk a parsed chain and return the joined output string.

        *chains* is the list of ``(op, cmd_nodes, backgrounded)`` tuples from
        ``program_to_chain``.  A length-1 chain produces identical output to
        the former single-command fast path.  Returns ``"(no output)"`` when
        nothing was produced.
        """
        srv = _get_server()

        for op, cmd_nodes, backgrounded in chains:
            joined = srv._serialize_pipeline_from_cmds(cmd_nodes)
            if op == "&&" and self.ran_any and self.prev_rc != 0:
                self.outputs.append(
                    f"(skipped: previous command exited {self.prev_rc}) — {joined}"
                )
                continue
            if op == "||" and self.ran_any and self.prev_rc == 0:
                self.outputs.append(
                    "(skipped: previous command succeeded) — " + joined
                )
                continue

            # timeout builtin: intercept before cd/allowlist dispatch so the
            # per-pipeline timeout override applies to the correct pipeline.
            nodes, eff_to, terr = srv._apply_timeout_builtin(
                cmd_nodes, self.expansion, backgrounded, timeout,
            )
            if terr is not None:
                self.outputs.append(terr)
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
                    self.prev_rc = 1
                    self.ran_any = True
                    continue
                if new_dir is not None:
                    self.work_dir = new_dir
                    self.prev_rc = 0
                    self.ran_any = True
                    continue

            if backgrounded:
                _rc, out = srv._run_background(
                    nodes, self.work_dir, expansion=self.expansion,
                )
                self.ran_any = True
                # Leave prev_rc unchanged — backgrounded exit code is unknown.
            elif len(nodes) == 1:
                rc, out = srv._run_segment(
                    nodes[0], self.work_dir, eff_to, expansion=self.expansion,
                )
                self.prev_rc = rc
                self.ran_any = True
            else:
                rc, out = srv._run_pipeline(
                    nodes, self.work_dir, eff_to, expansion=self.expansion,
                )
                self.prev_rc = rc
                self.ran_any = True
            if out:
                self.outputs.append(out)

        if not self.outputs:
            return "(no output)"
        return "\n".join(self.outputs)
