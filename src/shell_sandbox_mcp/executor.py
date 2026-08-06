"""Sandbox executor — invocation building, segment/pipeline execution,
command expansion, command substitution capture, and background execution.
"""

import os
import subprocess as _stdlib_subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import (
    MAX_OUTPUT,
    MAX_SUBST_COUNT,
    MAX_SUBST_DEPTH,
    MAX_SUBST_OUTPUT,
    SANDBOX_BIN,
    SANDBOX_WRAPPER,
    _base_env,
)
from .parser import (
    CommandNode,
    Expansion,
    ProgramNode,
    _expand_subst_in_text as _parser_expand_subst_in_text,
    _serialize_command,
    parse_command as _parser_parse_command,
    split_legacy,
)


def _get_server():
    """Lazy accessor for the server module (avoids circular import at module level)."""
    from . import server
    return server


# ---------------------------------------------------------------------------
# Invocation builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Invocation:
    """A successfully built sandbox invocation."""
    binary: str
    sandbox_args: list[str]
    env: Optional[dict[str, str]] = None
    cfg: dict = field(default_factory=dict)
    redirects: list = field(default_factory=list)


@dataclass(frozen=True)
class InvocationError:
    """_build_invocation rejected the command (error message)."""
    message: str


@dataclass(frozen=True)
class EmptyInvocation:
    """Empty command — nothing to run."""


def _build_invocation(
    command,
    work_dir: Path,
    expansion: Optional[Expansion] = None,
) -> "Invocation | InvocationError | EmptyInvocation":
    """Parse, resolve, and build the sandbox invocation for one segment.

    Accepts either a ``str`` (legacy) or a :class:`parser.CommandNode`
    (AST-native — no re-lexing).  Returns an :class:`Invocation` on success
    (``env`` is always a dict — allowlisted base + per-command overrides).
    Returns an :class:`InvocationError` when the command is rejected, carrying
    the error message.  Returns an :class:`EmptyInvocation` for an empty
    command (nothing to run).
    """
    from .parser import Redirect  # for type annotation only
    srv = _get_server()
    args, raw_redirects, parse_err = srv._extract_redirects(command, expansion)
    if parse_err is not None:
        return InvocationError(parse_err)

    if not args:
        return EmptyInvocation()

    # Validate redirect paths against the working directory
    redirects, path_err = srv._validate_redirect_paths(raw_redirects, work_dir)
    if path_err is not None:
        return InvocationError(path_err)

    # Resolve and validate against the allowlist (or a local binary under cwd)
    binary, final_args, cfg = srv._resolve_command(args, work_dir)
    if binary is None:
        return InvocationError(final_args)  # error message

    # Build sandbox invocation via the exec wrapper (bypasses posix_spawn APE issue)
    # Prepend per-command args (e.g. python3's "-S") before the user's args.
    prepend = cfg.get("prepend_args", [])
    sandbox_args = [
        str(SANDBOX_WRAPPER.resolve()),
        cfg["promises"],
        str(work_dir),  # unveil directory
        "--",
        binary,
    ] + prepend + final_args[1:]

    # Extra unveil paths via env vars the sandbox honors:
    #   SANDBOX_UNVEIL_R  — read-only (e.g. git config dotfiles under $HOME)
    #   SANDBOX_UNVEIL_RW — read-write (optional)
    unveil_env: dict[str, str] = {}
    extra_unveil = cfg.get("extra_unveil")
    if extra_unveil:
        paths = extra_unveil() if callable(extra_unveil) else extra_unveil
        if paths:
            unveil_env["SANDBOX_UNVEIL_R"] = ":".join(paths)

    extra_unveil_rw = cfg.get("extra_unveil_rw")
    if extra_unveil_rw:
        paths = extra_unveil_rw() if callable(extra_unveil_rw) else extra_unveil_rw
        if paths:
            unveil_env["SANDBOX_UNVEIL_RW"] = ":".join(paths)

    # Extra read-execute unveil paths (e.g. a vendored compiler toolchain and
    # the Cosmopolitan APE loader) so build tools can exec their subprocesses.
    extra_unveil_rx = cfg.get("extra_unveil_rx")
    if extra_unveil_rx:
        paths = extra_unveil_rx(work_dir) if callable(extra_unveil_rx) else extra_unveil_rx
        if paths:
            unveil_env["SANDBOX_UNVEIL_RX"] = ":".join(paths)

    # Sandbox-local python site dir: create <cwd>/.py-site, expose the base
    # via PYTHONUSERBASE (so `pip install --user` lands inside the sandbox)
    # and the site-packages via PYTHONPATH (so imports resolve).
    site_dir_name = cfg.get("site_dir_name")
    if site_dir_name:
        site_base = work_dir / site_dir_name
        try:
            site_base.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return InvocationError(f"Error creating python site dir {site_base}: {e}")
        site_packages = site_base / "lib" / "python3.12" / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        unveil_env["PYTHONUSERBASE"] = str(site_base)
        unveil_env["PYTHONPATH"] = str(site_packages)

    # For git, stage a sandbox-global config that swaps credential.helper for
    # the read-only shim, preserving all other ~/.gitconfig settings (including
    # [filter "lfs"]). GIT_CONFIG_GLOBAL overrides the default global config
    # path, so git uses the staged copy rather than ~/.gitconfig directly.
    if cfg.get("is_git"):
        unveil_env["GIT_CONFIG_GLOBAL"] = srv._stage_git_global_config()
        # git-lfs (a Go binary spawned by git as the LFS filter) needs the
        # waitid syscall, which no cosmocc pledge token permits. Skip pledge
        # for git only; unveil (which confines FS to work_dir + /tmp + system
        # rx + read-only config/cred paths) remains the security boundary.
        unveil_env["SANDBOX_NO_PLEDGE"] = "1"

    env: dict[str, str] = _base_env()          # always allowlisted base
    env.update(unveil_env)                      # SANDBOX_*, PYTHON*, GIT_*

    # Optionally prepend a directory to PATH (e.g. so build commands resolve a
    # busybox `mv` from the vendored toolchain instead of GNU /usr/bin/mv).
    path_prefix = cfg.get("path_prefix")
    if path_prefix:
        prefix = path_prefix() if callable(path_prefix) else path_prefix
        if prefix:
            cur = env.get("PATH", "")
            env["PATH"] = f"{prefix}:{cur}" if cur else prefix

    return Invocation(binary, sandbox_args, env, cfg, redirects)


# ---------------------------------------------------------------------------
# Single-segment execution
# ---------------------------------------------------------------------------


def _run_segment_core(
    command,
    work_dir: Path,
    timeout: int,
    expansion: Optional[Expansion] = None,
) -> tuple[int, bytes, bytes, list[str]]:
    """Run a single operator-free command segment in the sandbox (raw bytes).

    *command* may be a ``str`` (legacy) or a :class:`parser.CommandNode`
    (AST-native — avoids re-lexing).  Returns ``(returncode, stdout_bytes,
    stderr_bytes, report_lines)``.  ``stdout_bytes`` and ``stderr_bytes``
    are the captured output (may be empty but never ``None``).
    """
    srv = _get_server()
    inv = srv._build_invocation(command, work_dir, expansion=expansion)
    if isinstance(inv, EmptyInvocation):
        return 0, b"", b"", []  # empty command
    if isinstance(inv, InvocationError):
        return 1, inv.message.encode("utf-8", errors="replace"), b"", []
    binary, sandbox_args, env, cfg, redirects = (
        inv.binary, inv.sandbox_args, inv.env, inv.cfg, inv.redirects,
    )

    # Narrow the TOCTOU window for local binaries.
    if cfg.get("is_local_binary") and not srv._binary_still_contained(binary, work_dir):
        msg = f"Local binary no longer valid inside working directory: {binary}"
        return 1, msg.encode("utf-8", errors="replace"), b"", []

    try:
        stdout_t, stderr_t, to_close, report, shared_read_fd, stdin_bytes, stdin_file = (
            srv._resolve_fd_targets(
                redirects, _stdlib_subprocess.PIPE, _stdlib_subprocess.PIPE,
            )
        )
    except (OSError, ValueError) as e:
        return 1, f"Error opening redirect target: {e}".encode(), b"", []

    if shared_read_fd is not None:
        to_close.append(shared_read_fd)

    try:
        run_kwargs = dict(stdout=stdout_t, stderr=stderr_t, timeout=timeout, cwd=str(work_dir), env=env)
        if stdin_file is not None:
            run_kwargs["stdin"] = stdin_file
        elif stdin_bytes is not None:
            run_kwargs["input"] = stdin_bytes
        result = srv.subprocess.run(sandbox_args, **run_kwargs)

        stdout_bytes = result.stdout if result.stdout is not None else b""
        stderr_bytes = result.stderr if result.stderr is not None else b""

        if shared_read_fd is not None:
            combined = os.read(shared_read_fd, MAX_OUTPUT + 1)
            stdout_bytes = combined
            stderr_bytes = b""

        return result.returncode, stdout_bytes, stderr_bytes, report

    except _stdlib_subprocess.TimeoutExpired:
        return 1, f"Command timed out after {timeout}s".encode(), b"", []
    except FileNotFoundError:
        return 1, f"Sandbox binary not found: {SANDBOX_BIN}".encode(), b"", []
    except OSError as e:
        return 1, f"Error running command: {e}".encode(), b"", []
    finally:
        for item in to_close:
            try:
                if isinstance(item, int):
                    os.close(item)
                else:
                    item.close()
            except OSError:
                pass


def _format_output(
    rc: int,
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    report: list[str],
) -> str:
    """Format raw segment output into a string for display."""
    stdout = stdout_bytes[:MAX_OUTPUT].decode("utf-8", errors="replace")
    stderr_out = stderr_bytes[:MAX_OUTPUT].decode("utf-8", errors="replace")

    output = []
    if rc != 0:
        output.append(f"Exit code: {rc}")
    if stdout:
        output.append(stdout.rstrip())
    if stderr_out:
        output.append(f"[stderr]\n{stderr_out.rstrip()}")
    if report:
        output.extend(report)
    if len(stdout_bytes) > MAX_OUTPUT:
        output.append(f"\n[output truncated at {MAX_OUTPUT} bytes]")
    if len(stderr_bytes) > MAX_OUTPUT:
        output.append(f"\n[stderr truncated at {MAX_OUTPUT} bytes]")

    return "\n".join(output)


def _run_segment(command, work_dir: Path, timeout: int, expansion: Optional[Expansion] = None) -> tuple[int, str]:
    """Run a single operator-free command segment in the sandbox.

    *command* may be a ``str`` (legacy) or a :class:`parser.CommandNode`
    (AST-native).  Returns ``(returncode, output_string)``.  ``returncode``
    is 0 on success and non-zero on failure, an error, or an invalid/denied
    command, so callers can apply ``&&``/``||`` short-circuit semantics.
    ``output_string`` is the formatted output, or ``""`` when the segment
    produced nothing to report.
    """
    srv = _get_server()
    rc, stdout_bytes, stderr_bytes, report = srv._run_segment_core(
        command, work_dir, timeout, expansion=expansion,
    )
    return rc, _format_output(rc, stdout_bytes, stderr_bytes, report)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _run_pipeline_core(
    segments: list[str],
    work_dir: Path,
    timeout: int,
    expansion: Optional[Expansion] = None,
) -> tuple[int, bytes, bytes, list[str]]:
    """Run a pipe-connected sequence of segments concurrently (raw bytes).

    Returns ``(returncode, stdout_bytes, stderr_bytes, report_lines)``.
    ``stdout_bytes`` is the last stage's captured stdout; ``stderr_bytes`` is
    the combined stderr from all stages.
    """
    srv = _get_server()
    invocations: list[tuple[list[str], Optional[dict[str, str]], list["Redirect"]]] = []
    for i, seg in enumerate(segments):
        inv = srv._build_invocation(seg, work_dir, expansion=expansion)
        if isinstance(inv, EmptyInvocation):
            continue  # empty segment inside a pipeline
        if isinstance(inv, InvocationError):
            return 1, inv.message.encode(), b"", []
        binary, sandbox_args, env, cfg, redirects = (
            inv.binary, inv.sandbox_args, inv.env, inv.cfg, inv.redirects,
        )
        if cfg.get("is_local_binary") and not srv._binary_still_contained(binary, work_dir):
            msg = f"Local binary no longer valid inside working directory: {binary}"
            return 1, msg.encode(), b"", []
        # Reject heredoc/here-string/input-redirect on non-first pipeline stages
        if i > 0:
            for r in redirects:
                if r.fd == 0:
                    seg_str = _serialize_command(seg) if isinstance(seg, CommandNode) else seg
                    return 1, (
                        f"heredoc/here-string/input-redirect not allowed on "
                        f"non-first pipeline stage: {seg_str}"
                    ).encode(), b"", []
        invocations.append((sandbox_args, env, redirects))

    if not invocations:
        return 0, b"", b"", []

    # Reject stdout redirects on intermediate pipe stages.
    for i, (_sa, _env, redirects) in enumerate(invocations[:-1]):
        for r in redirects:
            if r.fd == 1:
                seg_str = _serialize_command(segments[i]) if isinstance(segments[i], CommandNode) else segments[i]
                return 1, (
                    f"Cannot redirect stdout of intermediate pipe stage: "
                    f"{seg_str}"
                ).encode(), b"", []

    # Resolve fd targets per stage.
    stdout_targets: list = []
    stderr_targets: list = []
    all_to_close: list = []
    all_report: list[list[str]] = []
    last_shared_read_fd = None
    first_stdin_bytes: Optional[bytes] = None
    first_stdin_file: Optional[object] = None
    try:
        for i, (_sa, _env, redirects) in enumerate(invocations):
            is_last = i == len(invocations) - 1
            st, et, tc, rpt, srf, stdin_b, stdin_f = srv._resolve_fd_targets(
                redirects, _stdlib_subprocess.PIPE, _stdlib_subprocess.PIPE,
                snapshot_2gt1=is_last,
            )
            stdout_targets.append(st)
            stderr_targets.append(et)
            all_to_close.extend(tc)
            all_report.append(rpt)
            if is_last:
                last_shared_read_fd = srf
            if i == 0:
                if stdin_b is not None:
                    first_stdin_bytes = stdin_b
                if stdin_f is not None:
                    first_stdin_file = stdin_f
    except (OSError, ValueError) as e:
        for fh in all_to_close:
            try:
                fh.close()
            except OSError:
                pass
        return 1, f"Error opening redirect target: {e}".encode(), b"", []

    # Launch every stage, chaining each one's stdout into the next one's stdin.
    procs: list[_stdlib_subprocess.Popen] = []
    prev: Optional[_stdlib_subprocess.Popen] = None
    stdin_writer_thread: Optional[threading.Thread] = None
    try:
        for i, (sandbox_args, env, _redirects) in enumerate(invocations):
            # Determine stdin for this stage
            if prev is not None:
                stage_stdin = prev.stdout
            elif i == 0 and first_stdin_file is not None:
                stage_stdin = first_stdin_file
            elif i == 0 and first_stdin_bytes is not None:
                stage_stdin = _stdlib_subprocess.PIPE
            else:
                stage_stdin = None

            p = srv.subprocess.Popen(
                sandbox_args,
                stdin=stage_stdin,
                stdout=stdout_targets[i],
                stderr=stderr_targets[i],
                cwd=str(work_dir),
                env=env,
            )
            if prev is not None:
                prev.stdout.close()  # parent no longer holds the read end
            procs.append(p)
            prev = p

            # If first stage has stdin_bytes, write them from a daemon thread
            if i == 0 and first_stdin_bytes is not None and p.stdin is not None:
                def _write_stdin(pipe, data):
                    try:
                        pipe.write(data)
                    finally:
                        pipe.close()
                stdin_writer_thread = threading.Thread(
                    target=_write_stdin, args=(p.stdin, first_stdin_bytes), daemon=True,
                )
                stdin_writer_thread.start()
    except Exception as e:
        # Clean up already-launched procs and open handles.
        for p in procs:
            try:
                p.kill()
            except ProcessLookupError:
                pass
        for p in procs:
            p.wait()
        for fh in all_to_close:
            try:
                fh.close()
            except OSError:
                pass
        if last_shared_read_fd is not None:
            try:
                os.close(last_shared_read_fd)
            except OSError:
                pass
        return 1, f"Failed to launch pipeline: {e}".encode(), b"", []

    # Drain the stderr of every stage but the last on a thread.
    last = procs[-1]
    stderr_bufs: dict[int, bytes] = {}
    bufs_lock = threading.Lock()

    def _drain_stderr(i: int, p: _stdlib_subprocess.Popen) -> None:
        if p.stderr is None:
            return
        data = p.stderr.read()
        p.stderr.close()
        with bufs_lock:
            stderr_bufs[i] = data

    threads: list[threading.Thread] = []
    for i, p in enumerate(procs[:-1]):
        if p.stderr is not None:
            t = threading.Thread(target=_drain_stderr, args=(i, p), daemon=True)
            t.start()
            threads.append(t)

    timed_out = False
    try:
        stdout_bytes, last_stderr = last.communicate(timeout=timeout)
    except _stdlib_subprocess.TimeoutExpired:
        timed_out = True

    # Reap every process that is still running.
    for p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass
    for p in procs:
        p.wait()

    # Drain threads complete.
    for t in threads:
        t.join(timeout=1)

    # Close any file handles opened for redirects.
    for fh in all_to_close:
        try:
            fh.close()
        except OSError:
            pass
    if last_shared_read_fd is not None:
        try:
            os.close(last_shared_read_fd)
        except OSError:
            pass

    if timed_out:
        if last.stdout is not None:
            try:
                last.stdout.close()
            except OSError:
                pass
        if last.stderr is not None:
            try:
                last.stderr.close()
            except OSError:
                pass
        return 1, f"Pipeline timed out after {timeout}s".encode(), b"", []

    rc = last.returncode
    with bufs_lock:
        intermediate_err = b"\n".join(
            stderr_bufs.get(i, b"") for i in range(len(procs) - 1)
        )
    combined_err = (intermediate_err + b"\n" + (last_stderr or b"")).strip()

    if stdout_bytes is None:
        stdout_bytes = b""

    return rc, stdout_bytes, combined_err, all_report[-1] if all_report else []


def _run_pipeline(
    segments,
    work_dir: Path,
    timeout: int,
    expansion: Optional[Expansion] = None,
) -> tuple[int, str]:
    """Run a pipe-connected sequence of segments concurrently in the sandbox.

    Each element of *segments* may be a ``str`` (legacy) or a
    :class:`parser.CommandNode` (AST-native).  Each segment's stdout is
    connected to the next segment's stdin, so data flows through the
    pipeline as in a real shell pipe. Every segment is still run through its
    own pledge sandbox and checked against the allowlist independently.

    Returns ``(returncode, output_string)``. ``returncode`` is the exit code of
    the *last* segment (shell default; no pipefail). Intermediate segments'
    stdout is consumed by the next stage; their stderr, plus the last stage's
    stdout and stderr, are surfaced in ``output_string``.
    """
    srv = _get_server()
    rc, stdout_bytes, stderr_bytes, report = srv._run_pipeline_core(
        segments, work_dir, timeout, expansion=expansion,
    )
    return rc, _format_output(rc, stdout_bytes, stderr_bytes, report)


# ---------------------------------------------------------------------------
# Command substitution capture — recursive execution for $( ... )
# ---------------------------------------------------------------------------


def _capture_stdout(
    command: str,
    work_dir: Path,
    timeout: int,
    depth: int,
    deadline: Optional[float] = None,
    subst_count: Optional[list[int]] = None,
    env: Optional[dict[str, str]] = None,
) -> tuple[int, bytes]:
    """Execute *command* in the sandbox and return ``(rc, raw_stdout_bytes)``.

    Used by ``_expand_command`` to resolve ``$( ... )`` substitutions.
    Depth and count limits are enforced to prevent runaway recursion.
    The sub-command's exit code does NOT propagate (matches shell default).
    Background ``&`` inside ``$()`` is rejected.

    *env* is the allowlisted environment forwarded to recursive expansion.
    """
    if subst_count is None:
        subst_count = [0]

    if depth >= MAX_SUBST_DEPTH:
        raise ValueError(
            f"Command substitution depth limit ({MAX_SUBST_DEPTH}) exceeded"
        )
    subst_count[0] += 1
    if subst_count[0] > MAX_SUBST_COUNT:
        raise ValueError(
            f"Command substitution count limit ({MAX_SUBST_COUNT}) exceeded"
        )

    if deadline is None:
        deadline = time.time() + timeout

    # Expand inner command (recursion).  The AST is discarded here — the
    # recursive path uses the legacy string-based split (split_legacy) for
    # simplicity, since the inner command is a short $(...) snippet.
    srv = _get_server()
    expanded, expansion, _program = srv._expand_command(
        command, work_dir, timeout, depth, deadline, subst_count,
        env=env,
    )

    # Split into pipelines
    pipelines = srv._split_command(expanded)
    if not pipelines:
        return 0, b""

    collected = bytearray()
    prev_rc = 0
    ran_any = False

    for op, stages, backgrounded in pipelines:
        if backgrounded:
            raise ValueError("background not allowed in command substitution")

        if op == "&&" and ran_any and prev_rc != 0:
            break
        if op == "||" and ran_any and prev_rc == 0:
            break

        # Recompute the remaining budget before each pipeline so a long chain
        # like `$(sleep 29; sleep 29)` can't exceed the overall deadline.
        remaining = max(1, deadline - time.time())

        if len(stages) == 1:
            rc, stdout_b, stderr_b, report = srv._run_segment_core(
                stages[0], work_dir, int(remaining), expansion=expansion,
            )
        else:
            rc, stdout_b, stderr_b, report = srv._run_pipeline_core(
                stages, work_dir, int(remaining), expansion=expansion,
            )
        prev_rc = rc
        ran_any = True
        # NOTE: A denied/error sub-command inside `$()` splices its error text
        # as captured stdout output here. This is a documented limitation — it
        # mirrors nothing in real shells, where a failing command substitution
        # contributes empty output. Left as-is deliberately (behavior choice).
        collected.extend(stdout_b)

        if len(collected) > MAX_SUBST_OUTPUT:
            collected = collected[:MAX_SUBST_OUTPUT]
            break

    return prev_rc, bytes(collected)


# ---------------------------------------------------------------------------
# Expansion pre-pass
# ---------------------------------------------------------------------------


def _expand_subst_in_text(
    text: str,
    work_dir: Path,
    timeout: int,
    depth: int,
    deadline: float,
    subst_count: list[int],
    env: Optional[dict[str, str]] = None,
) -> str:
    """Scan *text* for ``$( ... )`` and replace each with its raw output.

    Thin wrapper around :func:`parser._expand_subst_in_text`.
    """
    srv = _get_server()
    def _capture(inner: str) -> tuple[int, bytes]:
        return srv._capture_stdout(inner, work_dir, timeout, depth + 1, deadline, subst_count, env=env)
    return _parser_expand_subst_in_text(text, _capture, env=env)


# ---------------------------------------------------------------------------
# AST display helpers
# ---------------------------------------------------------------------------


def _serialize_pipeline_from_cmds(cmd_nodes) -> str:
    """Join a list of CommandNode objects into a display string."""
    return " | ".join(
        _serialize_command(c) if isinstance(c, CommandNode) else str(c)
        for c in cmd_nodes
    )


def _expand_command(
    command: str,
    work_dir: Path,
    timeout: int,
    depth: int,
    deadline: Optional[float] = None,
    subst_count: Optional[list[int]] = None,
    env: Optional[dict[str, str]] = None,
) -> tuple[str, Expansion, Optional[ProgramNode]]:
    """Pre-pass: resolve ``$(...)``, heredocs, and here-strings.

    Thin wrapper around :func:`parser.parse_command`.  Returns
    ``(cleaned_command, expansion, program_ast)`` — the caller should
    use the AST directly for execution rather than re-parsing the
    cleaned string.

    *env* supplies ``$VAR`` values.  When None, defaults to
    :func:`_base_env()` so expansion sees the same allowlisted vars as
    the subprocess will later receive.
    """
    base_env = env if env is not None else _base_env()
    srv = _get_server()
    def _capture(inner: str) -> tuple[int, bytes]:
        return srv._capture_stdout(inner, work_dir, timeout, depth + 1,
                               deadline, subst_count, env=base_env)
    cleaned, expansion, program = _parser_parse_command(
        command, _capture, work_dir, timeout, depth, deadline, subst_count,
        env=base_env,
    )
    return cleaned, expansion, program


# ---------------------------------------------------------------------------
# Legacy split wrapper
# ---------------------------------------------------------------------------


def _split_command(command: str) -> list[tuple[Optional[str], list[str], bool]]:
    """Split a command string into a chain of pipe-connected pipelines.

    Thin wrapper around :func:`parser.split_legacy`.
    """
    srv = _get_server()
    return srv.split_legacy(command)


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------

_reaper_lock = threading.Lock()
_reaper_started = False

# PIDs launched by the background machinery; the reaper thread only reaps
# these children so it never races with a concurrent foreground process.
_bg_pids: set[int] = set()
_bg_pids_lock = threading.Lock()


def _start_reaper() -> None:
    """Start a daemon thread that reaps zombie background children.

    Only reaps PIDs that were added to ``_bg_pids`` by ``_run_background``
    so we never steal an exit status from a concurrent foreground process.
    """
    global _reaper_started
    with _reaper_lock:
        if _reaper_started:
            return
        _reaper_started = True

    def _reap() -> None:
        while True:
            try:
                with _bg_pids_lock:
                    pids = list(_bg_pids)
                for pid in pids:
                    try:
                        wpid, _status = os.waitpid(pid, os.WNOHANG)
                        if wpid != 0:
                            with _bg_pids_lock:
                                _bg_pids.discard(wpid)
                    except ChildProcessError:
                        with _bg_pids_lock:
                            _bg_pids.discard(pid)
            except Exception:
                pass
            time.sleep(5)

    t = threading.Thread(target=_reap, daemon=True)
    t.start()


def _run_background(
    segments,
    work_dir: Path,
    expansion: Optional[Expansion] = None,
) -> tuple[int, str]:
    """Launch a pipe-connected pipeline in the background and return immediately.

    Each element of *segments* may be a ``str`` (legacy) or a
    :class:`parser.CommandNode` (AST-native).  Each segment is built through
    ``_build_invocation`` (same allowlist/sandbox checks as
    ``_run_pipeline``).  Intermediate stages' stdout feeds the next stage's
    stdin exactly as in a foreground pipeline; all stderr, plus the last
    stage's stdout, are redirected to a timestamped log file under
    ``work_dir`` so the parent never blocks on pipe buffers.

    Returns ``(0, message)`` with the PID and log path.
    """
    srv = _get_server()
    invocations: list[tuple[list[str], Optional[dict[str, str]], list["Redirect"]]] = []
    for i, seg in enumerate(segments):
        inv = srv._build_invocation(seg, work_dir, expansion=expansion)
        if isinstance(inv, EmptyInvocation):
            continue  # empty segment inside a pipeline
        if isinstance(inv, InvocationError):
            return 1, inv.message  # error message
        binary, sandbox_args, env, cfg, redirects = (
            inv.binary, inv.sandbox_args, inv.env, inv.cfg, inv.redirects,
        )
        if cfg.get("is_local_binary") and not srv._binary_still_contained(binary, work_dir):
            return 1, f"Local binary no longer valid inside working directory: {binary}"
        # Reject heredoc/here-string/input-redirect on non-first pipeline stages
        # (matches _run_pipeline_core).
        if i > 0:
            for r in redirects:
                if r.fd == 0:
                    seg_str = _serialize_command(seg) if isinstance(seg, CommandNode) else seg
                    return 1, (
                        f"heredoc/here-string/input-redirect not allowed on "
                        f"non-first pipeline stage: {seg_str}"
                    )
        invocations.append((sandbox_args, env, redirects))

    if not invocations:
        return 0, ""

    # Reject stdout redirects on intermediate pipe stages.
    for i, (_sa, _env, redirects) in enumerate(invocations[:-1]):
        for r in redirects:
            if r.fd == 1:
                seg_str = _serialize_command(segments[i]) if isinstance(segments[i], CommandNode) else segments[i]
                return 1, (
                    f"Cannot redirect stdout of intermediate pipe stage: "
                    f"{seg_str}"
                )

    log_path = work_dir / f".bg-{int(time.time() * 1000)}.log"

    # Sentinel marking "fall through to the background log". We only open the
    # log lazily, after resolving fd targets, so we don't create an empty log
    # file when the last stage redirects both stdout and stderr to user files.
    LOG_SENTINEL = object()

    # Resolve fd targets per stage; last stage defaults to the log file.
    stdout_targets: list = []
    stderr_targets: list = []
    all_to_close: list = []
    all_report: list[list[str]] = []
    log_opened = False
    log_fh = None
    first_stdin_file: Optional[object] = None
    try:
        for i, (_sa, _env, redirects) in enumerate(invocations):
            is_last = i == len(invocations) - 1
            def_stdout = LOG_SENTINEL if is_last else _stdlib_subprocess.PIPE
            def_stderr = LOG_SENTINEL if is_last else _stdlib_subprocess.PIPE
            st, et, tc, rpt, _srf, _stdin_b, stdin_f = srv._resolve_fd_targets(
                redirects, def_stdout, def_stderr, snapshot_2gt1=is_last,
            )
            if is_last:
                # Substitute the log handle for any fd that fell through to the
                # sentinel (i.e. was not redirected to a user file).
                if st is LOG_SENTINEL or et is LOG_SENTINEL:
                    if not log_opened:
                        log_fh = open(str(log_path), "wb")
                        all_to_close.append(log_fh)
                        log_opened = True
                    if st is LOG_SENTINEL:
                        st = log_fh
                    if et is LOG_SENTINEL:
                        et = log_fh
            stdout_targets.append(st)
            stderr_targets.append(et)
            all_to_close.extend(tc)
            all_report.append(rpt)
            if i == 0 and stdin_f is not None:
                first_stdin_file = stdin_f
    except OSError as e:
        for fh in all_to_close:
            try:
                fh.close()
            except OSError:
                pass
        return 1, f"Error opening redirect target: {e}"

    procs: list[_stdlib_subprocess.Popen] = []
    prev: Optional[_stdlib_subprocess.Popen] = None
    try:
        for i, (sandbox_args, env, _redirects) in enumerate(invocations):
            if prev is not None:
                stage_stdin = prev.stdout
            elif i == 0 and first_stdin_file is not None:
                stage_stdin = first_stdin_file
            else:
                stage_stdin = None
            p = srv.subprocess.Popen(
                sandbox_args,
                stdin=stage_stdin,
                stdout=stdout_targets[i],
                stderr=stderr_targets[i],
                cwd=str(work_dir),
                env=env,
                start_new_session=True,
            )
            if prev is not None:
                prev.stdout.close()
            procs.append(p)
            prev = p
    except Exception as e:
        # Clean up already-launched procs and open handles so nothing leaks.
        for p in procs:
            try:
                p.kill()
            except ProcessLookupError:
                pass
        for p in procs:
            p.wait()
        for fh in all_to_close:
            try:
                fh.close()
            except OSError:
                pass
        return 1, f"Failed to launch background pipeline: {e}"

    # Parent releases its handles; children hold their own copies.
    for fh in all_to_close:
        try:
            fh.close()
        except OSError:
            pass

    # Register every launched pid so the reaper thread only reaps our own
    # children and never races with a concurrent foreground process.
    with _bg_pids_lock:
        for p in procs:
            _bg_pids.add(p.pid)

    # Build the message with report details.
    if log_opened:
        msg_parts = [f"Backgrounded PID {procs[0].pid}; output -> {log_path}"]
    else:
        msg_parts = [f"Backgrounded PID {procs[0].pid}"]
    # Collect all unique report lines across stages.
    seen: set[str] = set()
    for rpt in all_report:
        for line in rpt:
            if line not in seen:
                msg_parts.append(line)
                seen.add(line)

    srv._start_reaper()
    return 0, "\n".join(msg_parts)
