"""Variable store for shell-sandbox variable assignment and builtins.

Provides a :class:`VariableStore` that holds per-call shell variables
(with export tracking) so ``VAR=value`` assignments, ``export``, ``unset``,
``set``, ``shift``, and ``source`` / ``.`` can modify the environment for
subsequent segments of the same ``shell_run`` call.

Init from :func:`config._base_env` so ``$PATH`` / ``$HOME`` resolve and
reach subprocesses (all base vars exported by default). ``unset PATH``
removes from both.
"""

import os
from dataclasses import dataclass, field

from .config import _base_env


@dataclass
class VariableStore:
    """Per-call mutable shell-variable store with export tracking.

    ``variables`` holds ALL shell vars (exported + local).
    ``exported`` is the set of names that should appear in subprocess envs.

    Init from the allowlisted host env so ``$PATH`` / ``$HOME`` / etc.
    are populated and reach subprocesses (all base vars exported by default).
    """

    variables: dict[str, str] = field(default_factory=lambda: dict(_base_env()))
    exported: set[str] = field(default_factory=lambda: set(_base_env().keys()))

    # Positional parameters ($1, $2, …) and script name ($0).
    positional: tuple[str, ...] = ()
    script_name: str = ""

    # User-defined functions: name → body text.
    functions: dict[str, str] = field(default_factory=dict)

    # Function call depth counter (for recursion cap).
    func_depth: int = 0

    # Special variables — set by Runner before expansion.
    # $? = last exit code, $$ = MCP server PID, $! = last bg PID, $- = shell flags.
    last_rc: int = 0
    last_bg_pid: int = 0

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> str:
        """Return the current value of *name*, or ``""`` if unset."""
        return self.variables.get(name, "")

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------

    def set_local(self, name: str, value: str) -> None:
        """Set *name* = *value* as a local (NOT exported)."""
        self.variables[name] = value

    def set_export(self, name: str, value: str) -> None:
        """Set *name* = *value* AND mark it exported."""
        self.variables[name] = value
        self.exported.add(name)

    def mark_export(self, name: str) -> None:
        """Mark *name* as exported without changing its value.

        If *name* is not yet in the store, initialise it to ``""``
        (mirrors POSIX ``export NAME`` for previously-unset variables).
        """
        if name not in self.variables:
            self.variables[name] = ""
        self.exported.add(name)

    def unset(self, name: str) -> None:
        """Remove *name* from both the variable store and the export set."""
        self.variables.pop(name, None)
        self.exported.discard(name)

    # ------------------------------------------------------------------
    # environment views
    # ------------------------------------------------------------------

    def is_exported(self, name: str) -> bool:
        """Return True if *name* is exported."""
        return name in self.exported

    def env_for_expansion(self) -> dict[str, str]:
        """Return the FULL dict (exported + local + positional) for ``${VAR}`` expansion.

        Positional parameters are layered AFTER the base dict so they
        override.  ``$0`` is ``script_name``, ``$#`` is the count, and
        ``$@`` / ``$*`` are space-joined (word-splitting is deferred — a
        simplified space-join is used for now).
        """
        env = dict(self.variables)
        # Positional parameters
        env["0"] = self.script_name
        for i, arg in enumerate(self.positional, 1):
            env[str(i)] = arg
        env["#"] = str(len(self.positional))
        env["@"] = " ".join(self.positional)
        env["*"] = " ".join(self.positional)
        # Special variables
        env["?"] = str(self.last_rc)
        env["$"] = str(os.getpid())
        env["!"] = str(self.last_bg_pid) if self.last_bg_pid else ""
        env["-"] = ""
        return env

    def env_for_subprocess(self) -> dict[str, str]:
        """Return only exported vars for subprocess environments."""
        return {k: v for k, v in self.variables.items() if k in self.exported}
