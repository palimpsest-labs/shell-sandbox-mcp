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
    """Apply redirects in order (last-wins per fd) and return an ``FdPlan``.

    Returns an :class:`FdPlan` with attributes ``stdout``, ``stderr``,
    ``to_close``, ``report``, ``shared_read_fd``, ``stdin_bytes`` and
    ``stdin_file``.  ``shared_read_fd`` is ``None`` unless a ``1>&2`` (or
    ``2>&1`` when ``snapshot_2gt1``) redirect forced creation of a shared
    pipe (when the source fd is ``subprocess.PIPE``).  ``stdin_bytes`` is
    ``None`` unless a heredoc/here-string redirect was provided;
    ``stdin_file`` is ``None`` unless a ``< file`` input redirect was
    provided (an open binary file object).

    ``snapshot_2gt1=False`` is used for intermediate pipeline stages, where
    a ``2>&1`` must merge stderr into the existing stdout pipe (so a later
    ``>file`` cannot apply to it) rather than snapshot a fresh shared pipe
    that would break the pipe chaining.
    """
    stdout_target = default_stdout
    stderr_target = default_stderr
    files_to_close: list = []
    report_lines: list[str] = []
    shared_pipe_read_fd = None
    stdin_bytes: Optional[bytes] = None
    stdin_file: Optional[object] = None

    for r in redirects:
        if r.op == "<":
            if stdin_file is not None or stdin_bytes is not None:
                raise ValueError("Multiple stdin redirects in one segment")
            try:
                fd = os.open(r.target_path, os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError:
                raise ValueError(f"Input redirect file not found: {r.raw_target}")
            except OSError as e:
                raise ValueError(f"Cannot open input redirect {r.raw_target}: {e}")
            stdin_file = os.fdopen(fd, "rb")
            files_to_close.append(stdin_file)
            report_lines.append(f"[stdin <- {r.raw_target}]")
        elif r.op in ("<<", "<<-", "<<<"):
            # Heredoc / here-string: only one stdin redirect per segment
            # (already enforced in _extract_redirects).
            if stdin_bytes is not None or stdin_file is not None:
                raise ValueError("Multiple stdin redirects in one segment")
            stdin_bytes = (r.body or "").encode("utf-8")
            if r.op == "<<<":
                report_lines.append("[stdin <<<]")
            elif r.op == "<<-":
                report_lines.append("[stdin <<-]")
            else:
                report_lines.append("[stdin <<]")
        elif r.op in (">", ">>"):
            # O_NOFOLLOW closes the symlink-swap TOCTOU: the path was
            # containment-validated earlier, so a redirect target must not be
            # a symlink pointing outside the work tree at open time.
            flags = os.O_WRONLY | os.O_CREAT
            flags |= os.O_TRUNC if r.op == ">" else os.O_APPEND
            flags |= os.O_NOFOLLOW
            fd = os.open(r.target_path, flags, 0o666)
            fh = os.fdopen(fd, "wb" if r.op == ">" else "ab")
            files_to_close.append(fh)
            if r.fd == 1:
                stdout_target = fh
                arrow = "->" if r.op == ">" else "->>"
                report_lines.append(f"[stdout {arrow} {r.raw_target}]")
            else:  # fd == 2
                stderr_target = fh
                arrow = "->" if r.op == ">" else "->>"
                report_lines.append(f"[stderr {arrow} {r.raw_target}]")
        elif r.op == ">&":
            if r.fd == 2 and r.target_fd == 1:  # 2>&1
                # If a LATER `>file` will redirect stdout, snapshot stdout's
                # current destination so stderr isn't dragged along with it
                # (matches POSIX: `2>&1 >file` puts stderr on the original
                # stdout). Otherwise a plain subprocess.STDOUT (merge stderr
                # into stdout's target) is correct and lighter.
                later_stdout_redirect = any(
                    rr.fd == 1 and rr.op in (">", ">>") for rr in redirects
                )
                if (
                    snapshot_2gt1
                    and later_stdout_redirect
                    and isinstance(stdout_target, int)
                    and stdout_target == subprocess.PIPE
                ):
                    rfd, wfd = os.pipe()
                    shared_pipe_read_fd = rfd
                    stdout_target = wfd
                    stderr_target = wfd
                    files_to_close.append(wfd)
                else:
                    stderr_target = subprocess.STDOUT
                report_lines.append("[stderr -> stdout]")
            elif r.fd == 1 and r.target_fd == 2:  # 1>&2
                if isinstance(stderr_target, int) and stderr_target == subprocess.PIPE:
                    # Create a shared pipe so both stdout and stderr write to
                    # the same fd; parent reads from the read end.
                    rfd, wfd = os.pipe()
                    shared_pipe_read_fd = rfd
                    stdout_target = wfd
                    stderr_target = wfd
                    files_to_close.append(wfd)
                else:
                    # stderr is a real file handle (or a shared-pipe fd);
                    # just point stdout at the same target.
                    stdout_target = stderr_target
                report_lines.append("[stdout -> stderr]")

    return FdPlan(
        stdout=stdout_target,
        stderr=stderr_target,
        to_close=files_to_close,
        report=report_lines,
        shared_read_fd=shared_pipe_read_fd,
        stdin_bytes=stdin_bytes,
        stdin_file=stdin_file,
    )
