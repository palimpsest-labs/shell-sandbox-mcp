"""Redirect helpers — extraction and fd-target resolution.

Thin wrappers around parser redirect extraction plus the fd-target resolver
that converts parsed Redirect objects into subprocess file descriptors.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from .parser import Redirect, extract_redirects as _parser_extract_redirects


@dataclass
class FdPlan:
    """Resolved file-descriptor plan for running one command segment."""
    stdout: object
    stderr: object
    to_close: list
    report: list
    shared_read_fd: Optional[int] = None
    stdin_bytes: Optional[bytes] = None
    stdin_file: Optional[object] = None

    def share_stdout_stderr_via_pipe(self) -> None:
        """Point both stdout and stderr at a single shared pipe.

        Used by ``1>&2`` and snapshot ``2>&1`` when the source fd is
        ``subprocess.PIPE``: both stdout and stderr write to the write end,
        and the parent reads from the read end (``shared_read_fd``).
        """
        rfd, wfd = os.pipe()
        self.shared_read_fd = rfd
        self.stdout = wfd
        self.stderr = wfd
        self.to_close.append(wfd)


@dataclass(frozen=True)
class FdDefaults:
    """Initial fd targets and stdin defaults for a segment's FdPlan."""
    stdout: object
    stderr: object
    snapshot_2gt1: bool = True
    stdin_bytes: Optional[bytes] = None
    stdin_file: Optional[object] = None


@dataclass(frozen=True)
class RedirectPlan:
    """A sequence of Redirects applied in order to a segment's fds."""
    redirects: tuple[Redirect, ...]

    def apply(self, defaults: FdDefaults) -> FdPlan:
        """Apply every redirect in order, returning the resolved FdPlan.

        Builds a fresh :class:`FdPlan` seeded from *defaults* and applies each
        redirect in place.  The plan is created fresh here and must NOT be
        reused across calls: each ``apply`` mutates its own plan.
        """
        plan = FdPlan(
            stdout=defaults.stdout,
            stderr=defaults.stderr,
            to_close=[],
            report=[],
            shared_read_fd=None,
            stdin_bytes=defaults.stdin_bytes,
            stdin_file=defaults.stdin_file,
        )
        has_later_stdout_file = any(
            r.fd == 1 and r.op in (">", ">>") for r in self.redirects
        )
        for r in self.redirects:
            r.apply(plan, snapshot_2gt1=has_later_stdout_file and defaults.snapshot_2gt1)
        return plan


def _extract_redirects(
    segment,
    expansion=None,
) -> tuple[list[str], list[Redirect], Optional[str]]:
    """Tokenize a command segment, extracting redirect operators.

    Thin wrapper around :func:`parser.extract_redirects`.  Accepts either a
    ``str`` (legacy path — lexes inline) or a :class:`parser.CommandNode`
    (AST-native path — uses the pre-parsed AST directly without re-lexing).
    """
    return _parser_extract_redirects(segment, expansion)


def _resolve_fd_targets(
    redirects: list[Redirect],
    default_stdout,
    default_stderr,
    *,
    snapshot_2gt1: bool = True,
) -> FdPlan:
    """Back-compat shim for the plan-based fd resolver.

    Delegates to :class:`RedirectPlan` / :class:`FdDefaults`.  Kept with its
    exact historical signature for the ~direct test call sites and the
    ``server`` re-export.
    """
    return RedirectPlan(tuple(redirects)).apply(
        FdDefaults(
            stdout=default_stdout,
            stderr=default_stderr,
            snapshot_2gt1=snapshot_2gt1,
        )
    )
