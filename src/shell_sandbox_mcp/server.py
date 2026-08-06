#!/usr/bin/env python3
"""Shell Sandbox MCP Server — safe shell command execution via pledge + busybox.

Tools:
  shell_run(command, cwd, timeout) — run a command in a pledge sandbox
  shell_list                          — list allowed commands
"""

import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Redirect dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Redirect:
    """A parsed shell redirect operator extracted from a command segment."""
    fd: int                                  # 0 (stdin), 1 (stdout), or 2 (stderr)
    op: Literal[">", ">>", ">&", "<<", "<<<", "<<-"]
    target_path: Optional[str] = None        # resolved absolute path (None for ">&")
    target_fd: Optional[int] = None          # source fd for ">&" (1 or 2); else None
    raw_target: Optional[str] = None         # user-typed target (for messages)
    body: Optional[str] = None               # literal stdin content for heredoc/here-string
    strip_tabs: bool = False                 # <<- semantics (strip leading TABs)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SANDBOX_BIN = REPO_ROOT / "bin" / "sandbox"
SANDBOX_WRAPPER = REPO_ROOT / "bin" / "run-sandbox"
BUSYBOX_BIN = REPO_ROOT / "bin" / "busybox"
COSMO_TOOLCHAIN = REPO_ROOT / "bin" / "cosmo-toolchain"
DEFAULT_ALLOWED_DIRS = [
    str(Path.home() / "projects"),
    "/tmp",
]
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300
MAX_OUTPUT = 1_000_000  # 1 MB

# ---------------------------------------------------------------------------
# Command substitution / heredoc limits
# ---------------------------------------------------------------------------

MAX_SUBST_DEPTH = 8
MAX_SUBST_COUNT = 256
MAX_SUBST_OUTPUT = 64_000
MAX_HEREDOC_BODY = 256_000
SENTINEL_ARG = re.compile(r"\x01A(\d+)\x01")
SENTINEL_HD = re.compile(r"\x01H(\d+)\x01")


@dataclass
class Expansion:
    """Side table holding resolved $() output words and heredoc/here-string bodies."""
    arg_values: dict[str, str]      # sentinel -> single-word $() result
    heredoc_bodies: dict[str, str]  # sentinel -> literal stdin body text

# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------


def _git_config_paths() -> list[str]:
    """Return the git config dotfiles under $HOME that must be unveiled.

    Covers the user-global config (~/.gitconfig) and the XDG config file
    (~/.config/git/config). Missing paths are ignored by the sandbox.
    Paths are resolved to their canonical form, matching the rest of the code.
    """
    home = Path.home().resolve()
    return [
        str((home / ".gitconfig").resolve()),
        str((home / ".config" / "git" / "config").resolve()),
    ]


def _git_credential_paths() -> list[str]:
    """Return the git credential store file under $HOME that must be unveiled.

    Now unveiled read-ONLY: the read-only credential helper shim
    (bin/git-cred-readonly) handles `get` and no-ops `store`, so the
    sandboxed git never writes ~/.git-credentials (which would need wc on
    $HOME — not granted).
    """
    return [
        str((Path.home().resolve() / ".git-credentials").resolve()),
    ]


def _git_extra_rx_paths(work_dir: Path) -> list[str]:
    """Read-execute paths git needs inside the sandbox.

    - <work_dir>/.git/hooks: so git can exec hook scripts (pre-push,
      post-commit, ...) installed by tools like git-lfs. rx only (no write):
      git runs existing hooks, it never writes them here. The dir may not
      exist yet; unveil tolerates ENOENT.
    - bin/git-cred-readonly: the read-only credential helper shim (Issue 2),
      which git must exec when credential.helper fires.
    """
    return [
        str((work_dir / ".git" / "hooks").resolve()),
        str((REPO_ROOT / "bin" / "git-cred-readonly").resolve()),
    ]


# Staged sandbox-global git config (idempotent, atomic write).
# Written once per python process under /tmp; reused across invocations.
_GIT_GLOBAL_CONFIG = Path(tempfile.gettempdir()) / f"sbx-git-global-{os.getuid()}-{os.getpid()}.config"
_GIT_CRED_SHIM = REPO_ROOT / "bin" / "git-cred-readonly"


def _git_readonly_paths() -> list[str]:
    """Read-only $HOME paths git needs: global/XDG config AND the credential
    store file. The cred file is read-ONLY here because the read-only helper
    shim (bin/git-cred-readonly) handles `get` and no-ops `store`, so the
    sandboxed git never writes ~/.git-credentials (which would need wc on
    $HOME — not granted)."""
    return _git_config_paths() + _git_credential_paths()


def _stage_git_global_config() -> str:
    """Write a sandbox-global git config = copy of ~/.gitconfig with
    `credential.helper` replaced by the read-only shim. Idempotent: content
    is identical across invocations, so concurrent writes are safe. Returns
    the path to set as GIT_CONFIG_GLOBAL."""
    cfg = _GIT_GLOBAL_CONFIG
    src = Path.home() / ".gitconfig"
    tmp = cfg.with_suffix(".tmp")
    if src.is_file():
        tmp.write_bytes(src.read_bytes())
    else:
        tmp.write_text("")
    subprocess.run(
        ["/usr/bin/git", "config", "--file", str(tmp),
         "--replace-all", "credential.helper", str(_GIT_CRED_SHIM.resolve())],
        check=True,
    )
    os.chmod(tmp, 0o600)
    os.replace(tmp, cfg)  # atomic
    return str(cfg)


def _cosmo_toolchain_paths(work_dir: Optional[Path] = None) -> list[str]:
    """Return the paths that must be unveiled read-execute for the vendored
    cosmocc toolchain to run: the toolchain tree itself (its compilers read
    headers/libs from it), the busybox binary (used as a non-preserving `mv`
    by build tools), and the Cosmopolitan APE loader under $HOME. The loaders
    at /usr/bin are already unveiled by the sandbox default.
    """
    return [
        str(COSMO_TOOLCHAIN.resolve()),
        str(BUSYBOX_BIN.resolve()),
        str((Path.home().resolve() / ".ape-1.10").resolve()),
    ]


def _cosmo_toolchain_bin() -> str:
    """Return the vendored toolchain's bin directory.

    Prepended to PATH for build commands so that `mv` (a busybox applet
    symlinked into the toolchain bin) is resolved instead of GNU /usr/bin/mv,
    which would otherwise try to preserve file attributes that the sandbox
    blocks (noisy but harmless warnings during builds).
    """
    return str((COSMO_TOOLCHAIN / "bin").resolve())


COMMANDS = {
    "git": {
        "binary": "/usr/bin/git",
        "promises": "stdio rpath wpath cpath prot_exec inet dns proc",
        "description": "Git version control",
        "extra_unveil": _git_readonly_paths,      # config + cred file, READ-ONLY
        "extra_unveil_rx": _git_extra_rx_paths,   # .git/hooks + cred shim
        "is_git": True,
    },
    "cargo": {
        "binary": "cargo",
        "promises": "stdio rpath wpath cpath proc prot_exec",
        "description": "Rust package manager (build, test, check, fmt, clippy)",
    },
    "make": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "make").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": (
            "GNU make build tool (vendored). Spawns compiler subprocesses, "
            "so it needs exec access to the toolchain."
        ),
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    "cosmocc": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "cosmocc").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": (
            "Cosmopolitan C/C++ compiler (vendored toolchain). Compiles "
            "portable APE binaries for both x86_64 and aarch64."
        ),
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    # Compiler aliases — all map to the same vendored cosmocc driver.
    "gcc": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "cosmocc").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": "Alias for cosmocc (Cosmopolitan C/C++ compiler)",
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    "cc": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "cosmocc").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": "Alias for cosmocc (Cosmopolitan C/C++ compiler)",
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    "clang": {
        "binary": str((COSMO_TOOLCHAIN / "bin" / "cosmocc").resolve()),
        "promises": "stdio rpath wpath cpath proc prot_exec fattr chown",
        "description": "Alias for cosmocc (Cosmopolitan C/C++ compiler)",
        "extra_unveil_rx": _cosmo_toolchain_paths,
        "path_prefix": _cosmo_toolchain_bin,
    },
    "python3": {
        "binary": str(REPO_ROOT / "bin" / "cosmo" / "python"),
        # fattr/chown: pip must set file mtimes (utime) and ownership when
        # unpacking wheels/sdists during `pip install` — without them package
        # extraction fails with "Operation not permitted" at tarfile.utime.
        # proc/prot_exec: let python spawn subprocesses (needed by the unit
        # test suite's real-subprocess integration tests, and by build tools
        # that shell out to python).
        "promises": "stdio rpath wpath cpath inet dns recvfd fattr chown proc prot_exec",
        "description": (
            "Python 3 interpreter (Cosmopolitan static build). Runs with "
            "-S so PYTHONPATH (a sandbox-local site dir inside the cwd) "
            "is honored; supports pip/network installs."
        ),
        # Prepend -S: the cosmo python's patched site module wipes PYTHONPATH
        # entries, so we disable it and rely on PYTHONPATH for the sandbox-
        # local site-packages dir.
        "prepend_args": ["-S"],
        # A per-invocation sandbox-local site base dir, created under the
        # cwd and exposed via PYTHONPATH/PYTHONUSERBASE. `pip install --user`
        # installs into <dir>/lib/python3.12/site-packages, and imports pick
        # it up via PYTHONPATH. Keeps packages inside the sandbox workspace
        # rather than unveiling the host's ~/.local.
        "site_dir_name": ".py-site",
    },
    "file": {
        "binary": str(REPO_ROOT / "bin" / "cosmo" / "file"),
        "promises": "stdio rpath",
        "description": "Determine file type (Cosmopolitan static build)",
    },
}

BUSYBOX_APPLETS = [
    # text/read
    "cat", "head", "tail", "wc", "sort", "uniq", "nl", "tac", "tee",
    "grep", "ls", "echo", "printf", "test", "expr",
    "cut", "tr", "diff", "cmp", "md5sum", "sha256sum",
    "which", "basename", "dirname", "realpath",
    # file manipulation
    "mkdir", "cp", "mv", "rm", "touch", "chmod",
    # inspection
    "find", "sed", "awk", "stat", "readlink", "df", "du", "uname",
    "id", "date", "env", "seq", "shuf", "xargs", "unzip",
    "true", "false", "sleep",
]


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "shell-sandbox",
    instructions="Run shell commands safely via pledge + busybox sandbox",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _resolve_command(
    args: list[str],
    work_dir: Optional[Path] = None,
) -> tuple[str, list[str], dict] | tuple[None, str, None]:
    """Resolve and validate a command against the allowlist.

    If the command is not a known allowlisted command, and a working
    directory is provided, tries to resolve the first token as a local
    binary path relative to that directory (e.g. "./scripts/foo" or
    "target/release/bar"). Such a binary is only allowed if it resolves
    to an executable file inside (or below) the working directory.

    Returns (binary_path, final_args, command_config) or (None, error_msg, None).
    """
    if not args:
        return None, "No command provided.", None

    cmd_name = args[0]

    # Check if it's a busybox applet alias
    if cmd_name in BUSYBOX_APPLETS:
        cfg = {
            "binary": str(BUSYBOX_BIN.resolve()),
            "promises": "stdio rpath wpath cpath",
            "description": f"BusyBox {cmd_name}",
        }
        binary = cfg["binary"]
        full_args = [binary, cmd_name] + args[1:]
        return binary, full_args, cfg

    # Reject direct busybox invocation — must use applet aliases
    if cmd_name == "busybox":
        return None, (
            f"Direct 'busybox' invocation not allowed. "
            f"Use individual applets instead: {', '.join(BUSYBOX_APPLETS[:10])}..."
        ), None

    # Check direct commands (git, cargo, ...)
    if cmd_name in COMMANDS:
        cfg = COMMANDS[cmd_name]
        return cfg["binary"], args, cfg

    # Fall back to resolving a local binary under the working directory.
    if work_dir is not None:
        binary = _resolve_local_binary(cmd_name, work_dir)
        if binary is not None:
            cfg = {
                "binary": binary,
                "promises": "stdio rpath wpath cpath prot_exec",
                "description": f"Local binary under cwd: {binary}",
                "is_local_binary": True,
            }
            return binary, args, cfg

    return None, f"Command not allowed: {cmd_name}. Use shell_list to see allowed commands.", None


def _resolve_local_binary(cmd_name: str, work_dir: Path) -> Optional[str]:
    """Try to resolve `cmd_name` to an executable file under `work_dir`.

    Accepts a bare name, a relative path (e.g. "scripts/foo" or
    "./target/bar"), or an absolute path, as long as it stays within the
    working tree. Returns the resolved binary path, or None if it is not
    an allowed local executable.
    """
    if cmd_name in (".", ".."):
        return None

    candidate = _contained_path(cmd_name, work_dir)
    if candidate is None:
        return None

    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        return None

    return str(candidate)


def _contained_path(cmd_name: str, work_dir: Path) -> Optional[Path]:
    """Resolve `cmd_name` relative to `work_dir`, returning the resolved
    path only if it stays within the working tree; else None.

    Uses `resolve()` (which follows symlinks), so a symlink inside the cwd
    pointing outside it is caught by the containment check.
    """
    try:
        raw = Path(cmd_name)
        candidate = raw if raw.is_absolute() else (work_dir / raw)
        candidate = candidate.resolve()
        # Must stay inside the working directory (or below it).
        candidate.relative_to(work_dir)
    except (ValueError, OSError):
        return None
    return candidate


def _binary_still_contained(binary: str, work_dir: Path) -> bool:
    """Re-verify a previously-resolved local binary path is still an
    executable file contained within the work dir, right before exec.

    Narrows the TOCTOU window between initial resolution and exec: if a
    directory component was swapped for a symlink escaping the tree in the
    meantime, the re-resolve + containment check catches it. Called with the
    path actually passed to the sandbox.
    """
    candidate = _contained_path(binary, work_dir)
    if candidate is None:
        return False
    return candidate.is_file() and os.access(candidate, os.X_OK)



def _validate_cwd(resolved: Path, raw: str) -> Optional[str]:
    """Validate a working directory. `resolved` must be an already-resolved
    absolute path; `raw` is the user's original input for error messages.

    Returns an error string, or None if valid.
    """
    if not resolved.is_dir():
        return f"Directory not found: {raw}"

    # Must be within an allowed tree or a subdirectory thereof
    for allowed in DEFAULT_ALLOWED_DIRS:
        allowed_path = Path(allowed).expanduser().resolve()
        try:
            resolved.relative_to(allowed_path)
            return None
        except ValueError:
            continue

    return f"Directory not in allowed paths: {raw}"


# ---------------------------------------------------------------------------
# Command chaining
# ---------------------------------------------------------------------------

def _split_command(command: str) -> list[tuple[Optional[str], list[str], bool]]:
    """Split a command string into a chain of pipe-connected pipelines.

    A pipeline is a list of segments joined by a single top-level `|` (pipe),
    where each segment's stdout feeds the next segment's stdin. Pipelines are
    joined by top-level `;`, `&&`, `||`, or `&` operators. Operators nested
    inside single or double quotes are preserved as literal text of the segment
    (they are not shell operators).

    Returns a list of ``(operator, pipeline, backgrounded)`` triples where
    ``operator`` is None for the first pipeline and ``';'``, ``'&&'``, or
    ``'||'`` for each subsequent one (never ``'&'`` — bare ``&`` acts like
    ``;`` semantically but sets ``backgrounded=True`` on the preceding
    pipeline). Each ``pipeline`` is a list of pipe-separated segment strings.
    Leading/trailing whitespace is stripped from each segment; empty segments
    are dropped.
    """
    pipelines: list[tuple[Optional[str], list[str], bool]] = []
    current_pipeline: list[str] = []
    current_seg: list[str] = []
    i, n = 0, len(command)
    quote: Optional[str] = None
    prev_op: Optional[str] = None

    def flush_segment() -> None:
        text = "".join(current_seg).strip()
        if text:
            current_pipeline.append(text)
        current_seg.clear()

    def flush_pipeline(backgrounded: bool = False) -> None:
        nonlocal prev_op
        flush_segment()
        if current_pipeline:
            pipelines.append((prev_op, current_pipeline[:], backgrounded))
        del current_pipeline[:]
        prev_op = None

    while i < n:
        c = command[i]
        if quote is not None:
            current_seg.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            current_seg.append(c)
            i += 1
            continue
        if c == ";":
            flush_pipeline()
            prev_op = ";"
            i += 1
            continue
        if c == "&" and i + 1 < n and command[i + 1] == "&":
            flush_pipeline()
            prev_op = "&&"
            i += 2
            continue
        if c == "&" and i > 0 and command[i - 1] == ">" and i + 1 < n and command[i + 1].isdigit():
            # fd-dup redirect like `2>&1` / `1>&2`: the `&` here is part of a
            # redirect operator, not a backgrounding operator. Leave it in the
            # current segment so _extract_redirects can parse it.
            current_seg.append(c)
            i += 1
            continue
        if c == "&":
            # Bare '&' backgrounding: flush the current pipeline marked as
            # backgrounded and reset prev_op (like ';' — next pipeline always
            # runs).
            flush_pipeline(backgrounded=True)
            i += 1
            continue
        if c == "|" and i + 1 < n and command[i + 1] == "|":
            flush_pipeline()
            prev_op = "||"
            i += 2
            continue
        if c == "|":
            # single pipe — end the current segment within the pipeline
            flush_segment()
            i += 1
            continue
        current_seg.append(c)
        i += 1

    flush_pipeline()
    return pipelines


# ---------------------------------------------------------------------------
# Redirect parsing
# ---------------------------------------------------------------------------


def _resolve_heredoc_body(
    token: str,
    expansion: Optional[Expansion],
    kind: str = "heredoc",
) -> Optional[str]:
    """Resolve a ``\\x01H<n>\\x01`` sentinel to its body text from *expansion*."""
    if not expansion:
        return None
    m = SENTINEL_HD.fullmatch(token)
    if not m:
        return None
    key = f"\x01H{m.group(1)}\x01"
    return expansion.heredoc_bodies.get(key)


def _strip_quotes(s: str) -> str:
    """Strip one level of single or double quotes from *s*.

    Returns the inner text if *s* is fully quoted (e.g. ``'hello'`` → ``hello``,
    ``"world"`` → ``world``), otherwise returns *s* unchanged.
    """
    if len(s) >= 2:
        if (s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"'):
            return s[1:-1]
    return s


def _extract_redirects(
    segment: str,
    expansion: Optional[Expansion] = None,
) -> tuple[list[str], list[Redirect], Optional[str]]:
    """Tokenize a command segment, extracting redirect operators.

    Single-pass char-by-char tokenizer that:
    - Splits on unquoted whitespace
    - Strips quote characters (``'`` and ``"``) from words (POSIX-style)
    - Recognizes unquoted redirect operators only at word boundaries:
      ``> file``, ``>> file``, ``2> file``, ``2>> file``, ``2>&1``, ``1>&2``
    - Recognizes heredoc/here-string operators (``<<<``, ``<<-``, ``<<``)
      when their target is a sentinel token (``\\x01H<n>\\x01``)
    - Resolves ``\\x01A<n>\\x01`` sentinels to single-word $() results
    - Glued forms like ``foo>bar`` are treated as literal words (a deliberate
      divergence from POSIX; this avoids ambiguous parsing with commands that
      embed ``>`` in their arguments).

    Returns ``(args, redirects, None)`` on success, or
    ``([], [], error_msg)`` on error.
    ``args`` is the tokenized command with quotes stripped and redirect
    operators (plus their targets) removed.
    """
    args: list[str] = []
    redirects: list[Redirect] = []
    i = 0
    n = len(segment)

    def _resolve_word(w: str) -> str:
        """Resolve ``\\x01A<n>\\x01`` sentinels anywhere within a word.

        Substitutes every sentinel found (so compound words like
        ``a\x01A0\x01c`` resolve to ``a<val>c``), not just a word that is
        exactly one sentinel. A sentinel with no stored value is left as-is.
        """
        if not expansion:
            return w

        def _replace(m: re.Match) -> str:
            key = f"\x01A{m.group(1)}\x01"
            val = expansion.arg_values.get(key)
            return val if val is not None else m.group(0)

        return SENTINEL_ARG.sub(_replace, w)

    def _read_word() -> Optional[str]:
        """Read a shell word starting at *i*, stripping quotes.
        Stops at unquoted whitespace.  Returns ``None`` on unbalanced quotes.
        """
        nonlocal i
        chars: list[str] = []
        quote: Optional[str] = None
        while i < n:
            c = segment[i]
            if quote is not None:
                if c == quote:
                    quote = None
                else:
                    chars.append(c)
                i += 1
            elif c in ("'", '"'):
                quote = c
                i += 1
            elif c in (' ', '\t'):
                break
            else:
                chars.append(c)
                i += 1
        if quote is not None:
            return None
        return ''.join(chars)

    def _read_redirect_target() -> Optional[str]:
        """Read the target of a ``>`` / ``>>`` operator.
        *i* is positioned right after the operator characters.
        The target may be glued (``>out.txt``) or space-separated
        (``> out.txt``).  Returns the target word, or ``None`` if missing.
        """
        nonlocal i
        if i < n and segment[i] not in (' ', '\t'):
            target = _read_word()
            return _resolve_word(target) if target is not None else None
        while i < n and segment[i] in (' ', '\t'):
            i += 1
        if i >= n:
            return None
        target = _read_word()
        return _resolve_word(target) if target is not None else None

    while i < n:
        while i < n and segment[i] in (' ', '\t'):
            i += 1
        if i >= n:
            break

        rem = segment[i:]

        # -- longest-match-first: 4-char fd-dup patterns --
        # Only treat `2>&1` / `1>&2` as fd-dup when the char after the
        # 4-char sequence (if any) is NOT alphanumeric or underscore
        # (whitespace, end-of-input, or another operator).  Otherwise
        # `cmd 2>&1x` is a `2>` redirect to file `&1x`, not fd-dup.
        if rem.startswith('2>&1') and not (len(rem) > 4 and (rem[4].isalnum() or rem[4] == '_')):
            i += 4
            redirects.append(Redirect(fd=2, op='>&', target_path=None, target_fd=1, raw_target='1'))
            continue
        if rem.startswith('1>&2') and not (len(rem) > 4 and (rem[4].isalnum() or rem[4] == '_')):
            i += 4
            redirects.append(Redirect(fd=1, op='>&', target_path=None, target_fd=2, raw_target='2'))
            continue

        # -- 4-char fd-dup patterns for disallowed target fds --
        # e.g. `2>&3`, `1>&4` — only 1 and 2 are valid dup targets.
        # Require a word boundary after the 4-char sequence, otherwise
        # `2>&3x` is a `2>` redirect to file `&3x`, not fd-dup.
        if (len(rem) >= 4 and rem[0].isdigit() and rem[1:3] == '>&'
                and rem[3].isdigit()
                and not (len(rem) > 4 and (rem[4].isalnum() or rem[4] == '_'))):
            return [], [], "Redirect dup target fd must be 1 or 2"

        # -- 3-char patterns --
        if rem.startswith('2>>'):
            i += 3
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=2, op='>>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('1>>'):
            i += 3
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=1, op='>>', target_path=None, target_fd=None, raw_target=t))
            continue

        # -- here-string: <<<  (3-char, checked before 2-char <<) --
        if rem.startswith('<<<'):
            i += 3
            t = _read_redirect_target()
            if t is None:
                return [], [], "Here-string missing target"
            # Resolve sentinel to body
            body = _resolve_heredoc_body(t, expansion, "here-string")
            if body is None:
                return [], [], "Here-string body not found"
            # Check for duplicate stdin redirects
            for r in redirects:
                if r.fd == 0:
                    return [], [], "Multiple stdin redirects in one segment"
            redirects.append(Redirect(fd=0, op='<<<', body=body))
            continue

        # -- heredoc with tab strip: <<-  (3-char, before <<) --
        if rem.startswith('<<-'):
            i += 3
            t = _read_redirect_target()
            if t is None:
                return [], [], "Heredoc missing delimiter sentinel"
            body = _resolve_heredoc_body(t, expansion, "heredoc")
            if body is None:
                return [], [], "Heredoc body not found"
            for r in redirects:
                if r.fd == 0:
                    return [], [], "Multiple stdin redirects in one segment"
            redirects.append(Redirect(fd=0, op='<<-', body=body, strip_tabs=True))
            continue

        # -- 2-char patterns --
        if rem.startswith('>>'):
            i += 2
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=1, op='>>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('2>'):
            i += 2
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=2, op='>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('1>'):
            i += 2
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target=t))
            continue

        # -- heredoc: <<  (2-char, after <<- and <<<) --
        if rem.startswith('<<'):
            i += 2
            t = _read_redirect_target()
            if t is None:
                return [], [], "Heredoc missing delimiter sentinel"
            body = _resolve_heredoc_body(t, expansion, "heredoc")
            if body is None:
                return [], [], "Heredoc body not found"
            for r in redirects:
                if r.fd == 0:
                    return [], [], "Multiple stdin redirects in one segment"
            redirects.append(Redirect(fd=0, op='<<', body=body))
            continue

        # -- 1-char patterns --
        if rem.startswith('>'):
            i += 1
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('<'):
            return [], [], "Input redirects are not supported"

        # -- digit + >  patterns that aren't the allowed 1> / 2> → error --
        if len(rem) >= 2 and rem[0].isdigit() and rem[1] == '>':
            fd = int(rem[0])
            return [], [], f"Redirects only support fds 1 and 2 (got {fd})"

        # -- regular word --
        w = _read_word()
        if w is None:
            return [], [], "Unbalanced quotes in command"
        if w:
            args.append(_resolve_word(w))

    return args, redirects, None


def _validate_redirect_paths(
    redirects: list[Redirect],
    work_dir: Path,
) -> tuple[list[Redirect], Optional[str]]:
    """Resolve and validate the paths in a list of redirects.

    For each ``>`` / ``>>`` redirect, resolves ``raw_target`` against
    ``work_dir`` via :func:`_contained_path`.  Returns the updated list
    (with ``target_path`` populated) or ``([], error_msg)`` if a target
    escapes the working directory.  ``>&`` redirects pass through unchanged.
    """
    from dataclasses import replace

    validated: list[Redirect] = []
    for r in redirects:
        if r.op in (">", ">>"):
            cand = _contained_path(r.raw_target, work_dir)
            if cand is None:
                return [], f"Redirect target escapes working directory: {r.raw_target}"
            validated.append(replace(r, target_path=str(cand)))
        else:
            validated.append(r)
    return validated, None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _build_invocation(
    command: str,
    work_dir: Path,
    expansion: Optional[Expansion] = None,
) -> tuple[Optional[str], Optional[list[str]], Optional[dict], Optional[dict], list[Redirect]]:
    """Parse, resolve, and build the sandbox invocation for one segment.

    Returns ``(binary, sandbox_args, env, cfg, redirects)`` on success.
    ``env`` is ``None`` when no env overrides are needed.
    On failure, returns a tuple whose first element is the error message
    (string) and whose remaining elements are ``None`` / empty:
    ``(error_msg, None, None, None, [])``.
    An empty command returns ``(None, None, None, None, [])``.
    """
    args, raw_redirects, parse_err = _extract_redirects(command, expansion)
    if parse_err is not None:
        return parse_err, None, None, None, []

    if not args:
        return None, None, None, None, []

    # Validate redirect paths against the working directory
    redirects, path_err = _validate_redirect_paths(raw_redirects, work_dir)
    if path_err is not None:
        return path_err, None, None, None, []

    # Resolve and validate against the allowlist (or a local binary under cwd)
    binary, final_args, cfg = _resolve_command(args, work_dir)
    if binary is None:
        return final_args, None, None, None, []  # error message

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
    env: Optional[dict] = None
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
            return f"Error creating python site dir {site_base}: {e}", None, None, None, []
        site_packages = site_base / "lib" / "python3.12" / "site-packages"
        site_packages.mkdir(parents=True, exist_ok=True)
        unveil_env["PYTHONUSERBASE"] = str(site_base)
        unveil_env["PYTHONPATH"] = str(site_packages)

    # For git, stage a sandbox-global config that swaps credential.helper for
    # the read-only shim, preserving all other ~/.gitconfig settings (including
    # [filter "lfs"]). GIT_CONFIG_GLOBAL overrides the default global config
    # path, so git uses the staged copy rather than ~/.gitconfig directly.
    if cfg.get("is_git"):
        unveil_env["GIT_CONFIG_GLOBAL"] = _stage_git_global_config()

    if unveil_env:
        env = {**os.environ, **unveil_env}

    # Optionally prepend a directory to PATH (e.g. so build commands resolve a
    # busybox `mv` from the vendored toolchain instead of GNU /usr/bin/mv).
    path_prefix = cfg.get("path_prefix")
    if path_prefix:
        prefix = path_prefix() if callable(path_prefix) else path_prefix
        if prefix:
            base = env if env is not None else dict(os.environ)
            cur = base.get("PATH", "")
            base["PATH"] = f"{prefix}:{cur}" if cur else prefix
            env = base

    return binary, sandbox_args, env, cfg, redirects


def _resolve_fd_targets(
    redirects: list[Redirect],
    default_stdout,
    default_stderr,
    *,
    snapshot_2gt1: bool = True,
) -> tuple:
    """Apply redirects in order (last-wins per fd) and return fd targets.

    Returns ``(stdout_target, stderr_target, files_to_close, report_lines,
    shared_pipe_read_fd, stdin_bytes)`` where ``shared_pipe_read_fd`` is
    ``None`` unless a ``1>&2`` (or ``2>&1`` when ``snapshot_2gt1``) redirect
    forced creation of a shared pipe (when the source fd is
    ``subprocess.PIPE``).  ``stdin_bytes`` is ``None`` unless a heredoc/
    here-string redirect was provided.

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

    for r in redirects:
        if r.op in ("<<", "<<-", "<<<"):
            # Heredoc / here-string: only one stdin redirect per segment
            # (already enforced in _extract_redirects).
            if stdin_bytes is not None:
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

    return (
        stdout_target, stderr_target, files_to_close, report_lines,
        shared_pipe_read_fd, stdin_bytes,
    )


def _run_segment_core(
    command: str,
    work_dir: Path,
    timeout: int,
    expansion: Optional[Expansion] = None,
) -> tuple[int, bytes, bytes, list[str]]:
    """Run a single operator-free command segment in the sandbox (raw bytes).

    Returns ``(returncode, stdout_bytes, stderr_bytes, report_lines)``.
    ``stdout_bytes`` and ``stderr_bytes`` are the captured output (may be
    empty but never ``None``).
    """
    binary, sandbox_args, env, cfg, redirects = _build_invocation(
        command, work_dir, expansion=expansion,
    )
    if sandbox_args is None:
        if binary is None:
            return 0, b"", b"", []  # empty command
        return 1, binary.encode("utf-8", errors="replace"), b"", []  # error

    # Narrow the TOCTOU window for local binaries.
    if cfg.get("is_local_binary") and not _binary_still_contained(binary, work_dir):
        msg = f"Local binary no longer valid inside working directory: {binary}"
        return 1, msg.encode("utf-8", errors="replace"), b"", []

    try:
        stdout_t, stderr_t, to_close, report, shared_read_fd, stdin_bytes = (
            _resolve_fd_targets(
                redirects, subprocess.PIPE, subprocess.PIPE,
            )
        )
    except (OSError, ValueError) as e:
        return 1, f"Error opening redirect target: {e}".encode(), b"", []

    if shared_read_fd is not None:
        to_close.append(shared_read_fd)

    try:
        result = subprocess.run(
            sandbox_args,
            stdout=stdout_t,
            stderr=stderr_t,
            input=stdin_bytes,
            timeout=timeout,
            cwd=str(work_dir),
            env=env,
        )

        stdout_bytes = result.stdout if result.stdout is not None else b""
        stderr_bytes = result.stderr if result.stderr is not None else b""

        if shared_read_fd is not None:
            combined = os.read(shared_read_fd, MAX_OUTPUT + 1)
            stdout_bytes = combined
            stderr_bytes = b""

        return result.returncode, stdout_bytes, stderr_bytes, report

    except subprocess.TimeoutExpired:
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


def _run_segment(command: str, work_dir: Path, timeout: int, expansion: Optional[Expansion] = None) -> tuple[int, str]:
    """Run a single operator-free command segment in the sandbox.

    Returns ``(returncode, output_string)``. ``returncode`` is 0 on success and
    non-zero on failure, an error, or an invalid/denied command, so callers can
    apply ``&&``/``||`` short-circuit semantics. ``output_string`` is the
    formatted output, or ``""`` when the segment produced nothing to report.
    """
    rc, stdout_bytes, stderr_bytes, report = _run_segment_core(
        command, work_dir, timeout, expansion=expansion,
    )
    return rc, _format_output(rc, stdout_bytes, stderr_bytes, report)


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
    invocations: list[tuple[list[str], Optional[dict], list[Redirect]]] = []
    for i, seg in enumerate(segments):
        binary, sandbox_args, env, cfg, redirects = _build_invocation(
            seg, work_dir, expansion=expansion,
        )
        if sandbox_args is None:
            if binary is None:
                continue  # empty segment inside a pipeline
            # error message
            return 1, binary.encode(), b"", []
        if cfg.get("is_local_binary") and not _binary_still_contained(binary, work_dir):
            msg = f"Local binary no longer valid inside working directory: {binary}"
            return 1, msg.encode(), b"", []
        # Reject heredoc/here-string on non-first pipeline stages
        if i > 0:
            for r in redirects:
                if r.fd == 0:
                    return 1, (
                        f"heredoc/here-string not allowed on non-first "
                        f"pipeline stage: {seg}"
                    ).encode(), b"", []
        invocations.append((sandbox_args, env, redirects))

    if not invocations:
        return 0, b"", b"", []

    # Reject stdout redirects on intermediate pipe stages.
    for i, (_sa, _env, redirects) in enumerate(invocations[:-1]):
        for r in redirects:
            if r.fd == 1:
                return 1, (
                    f"Cannot redirect stdout of intermediate pipe stage: "
                    f"{segments[i]}"
                ).encode(), b"", []

    # Resolve fd targets per stage.
    stdout_targets: list = []
    stderr_targets: list = []
    all_to_close: list = []
    all_report: list[list[str]] = []
    last_shared_read_fd = None
    first_stdin_bytes: Optional[bytes] = None
    try:
        for i, (_sa, _env, redirects) in enumerate(invocations):
            is_last = i == len(invocations) - 1
            st, et, tc, rpt, srf, stdin_b = _resolve_fd_targets(
                redirects, subprocess.PIPE, subprocess.PIPE,
                snapshot_2gt1=is_last,
            )
            stdout_targets.append(st)
            stderr_targets.append(et)
            all_to_close.extend(tc)
            all_report.append(rpt)
            if is_last:
                last_shared_read_fd = srf
            if i == 0 and stdin_b is not None:
                first_stdin_bytes = stdin_b
    except (OSError, ValueError) as e:
        for fh in all_to_close:
            try:
                fh.close()
            except OSError:
                pass
        return 1, f"Error opening redirect target: {e}".encode(), b"", []

    # Launch every stage, chaining each one's stdout into the next one's stdin.
    procs: list[subprocess.Popen] = []
    prev: Optional[subprocess.Popen] = None
    stdin_writer_thread: Optional[threading.Thread] = None
    try:
        for i, (sandbox_args, env, _redirects) in enumerate(invocations):
            # Determine stdin for this stage
            if prev is not None:
                stage_stdin = prev.stdout
            elif i == 0 and first_stdin_bytes is not None:
                stage_stdin = subprocess.PIPE
            else:
                stage_stdin = None

            p = subprocess.Popen(
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

    def _drain_stderr(i: int, p: subprocess.Popen) -> None:
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
    except subprocess.TimeoutExpired:
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
    segments: list[str],
    work_dir: Path,
    timeout: int,
    expansion: Optional[Expansion] = None,
) -> tuple[int, str]:
    """Run a pipe-connected sequence of segments concurrently in the sandbox.

    Each segment's stdout is connected to the next segment's stdin, so data
    flows through the pipeline as in a real shell pipe. Every segment is still
    run through its own pledge sandbox and checked against the allowlist
    independently.

    Returns ``(returncode, output_string)``. ``returncode`` is the exit code of
    the *last* segment (shell default; no pipefail). Intermediate segments'
    stdout is consumed by the next stage; their stderr, plus the last stage's
    stdout and stderr, are surfaced in ``output_string``.
    """
    rc, stdout_bytes, stderr_bytes, report = _run_pipeline_core(
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
) -> tuple[int, bytes]:
    """Execute *command* in the sandbox and return ``(rc, raw_stdout_bytes)``.

    Used by ``_expand_command`` to resolve ``$( ... )`` substitutions.
    Depth and count limits are enforced to prevent runaway recursion.
    The sub-command's exit code does NOT propagate (matches shell default).
    Background ``&`` inside ``$()`` is rejected.
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

    # Expand inner command (recursion)
    expanded, expansion = _expand_command(
        command, work_dir, timeout, depth, deadline, subst_count,
    )

    # Split into pipelines
    pipelines = _split_command(expanded)
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
            rc, stdout_b, stderr_b, report = _run_segment_core(
                stages[0], work_dir, int(remaining), expansion=expansion,
            )
        else:
            rc, stdout_b, stderr_b, report = _run_pipeline_core(
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
) -> str:
    """Scan *text* for ``$( ... )`` and replace each with its raw output.

    Used for expanding ``$()`` inside unquoted heredoc bodies and unquoted
    here-string words. No sentinel tokens — the output is spliced directly
    into the body text.
    """
    result: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if (c == '$' and i + 1 < n and text[i + 1] == '('
                and not (i > 0 and text[i - 1] == '\\')):
            # Find matching close paren with quote tracking
            j = i + 2
            paren_depth = 1
            inner_quote: Optional[str] = None
            while j < n and paren_depth > 0:
                ch = text[j]
                if inner_quote is not None:
                    if ch == inner_quote:
                        inner_quote = None
                elif ch in ("'", '"'):
                    inner_quote = ch
                elif ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
                j += 1
            if paren_depth != 0:
                raise ValueError("Unbalanced $( ... )")
            inner = text[i + 2 : j - 1]
            _rc, stdout_bytes = _capture_stdout(
                inner, work_dir, timeout, depth + 1, deadline, subst_count,
            )
            expanded = stdout_bytes.decode("utf-8", errors="replace").rstrip("\n")
            result.append(expanded)
            i = j
        else:
            result.append(c)
            i += 1
    return "".join(result)


def _expand_command(
    command: str,
    work_dir: Path,
    timeout: int,
    depth: int,
    deadline: Optional[float] = None,
    subst_count: Optional[list[int]] = None,
) -> tuple[str, Expansion]:
    """Pre-pass: resolve ``$(...)``, heredocs, and here-strings.

    Scans *command* left-to-right, quote-aware, and replaces each ``$(...)``
    with a sentinel token (``\\x01A<n>\\x01``) while storing the captured
    output in *expansion.arg_values*. Heredoc and here-string bodies are moved
    to *expansion.heredoc_bodies*, keyed by sentinel tokens
    (``\\x01H<n>\\x01``), and the operators are replaced with ``<<<`` /
    ``<<-`` / ``<<`` + sentinel.

    Returns ``(cleaned_command, expansion)``.  The cleaned command contains
    only sentinels where bodies/results were removed — it is safe to pass to
    the normal ``_split_command`` / ``_extract_redirects`` tokenizers.
    """
    if subst_count is None:
        subst_count = [0]
    if deadline is None:
        deadline = time.time() + timeout

    expansion = Expansion(arg_values={}, heredoc_bodies={})
    output: list[str] = []
    i, n = 0, len(command)
    quote: Optional[str] = None
    next_arg_id = 0
    next_hd_id = 0

    while i < n:
        c = command[i]

        # ---- inside a quote: copy verbatim ----
        if quote is not None:
            output.append(c)
            if c == quote:
                quote = None
            i += 1
            continue

        # ---- quote start ----
        if c in ("'", '"'):
            quote = c
            output.append(c)
            i += 1
            continue

        # ---- $( ... ) command substitution ----
        if c == '$' and i + 1 < n and command[i + 1] == '(':
            # Find matching ')' with paren + quote tracking
            j = i + 2
            paren_depth = 1
            inner_quote: Optional[str] = None
            while j < n and paren_depth > 0:
                ch = command[j]
                if inner_quote is not None:
                    if ch == inner_quote:
                        inner_quote = None
                elif ch in ("'", '"'):
                    inner_quote = ch
                elif ch == '(':
                    paren_depth += 1
                elif ch == ')':
                    paren_depth -= 1
                j += 1
            if paren_depth != 0:
                raise ValueError("Unbalanced $( ... )")
            inner = command[i + 2 : j - 1]
            rc, stdout_bytes = _capture_stdout(
                inner, work_dir, timeout, depth + 1, deadline, subst_count,
            )
            result = stdout_bytes.decode("utf-8", errors="replace").rstrip("\n")
            result = result[:MAX_SUBST_OUTPUT]
            sentinel = f"\x01A{next_arg_id}\x01"
            next_arg_id += 1
            expansion.arg_values[sentinel] = result
            output.append(sentinel)
            i = j
            continue

        # ---- here-string: <<< (3-char, before <<- and <<) ----
        if command[i : i + 3] == '<<<':
            output.append('<<<')
            i += 3
            # skip whitespace after <<<
            while i < n and command[i] in (' ', '\t'):
                i += 1
            # read the word (quote-aware)
            word_chars: list[str] = []
            wq: Optional[str] = None
            single_quoted = False
            while i < n:
                ch = command[i]
                if wq is not None:
                    word_chars.append(ch)
                    if ch == wq:
                        wq = None
                    i += 1
                elif ch in ("'", '"'):
                    if ch == "'":
                        single_quoted = True
                    wq = ch
                    word_chars.append(ch)
                    i += 1
                elif ch == '\n':
                    break
                else:
                    word_chars.append(ch)
                    i += 1
            raw_word = "".join(word_chars)
            body_word = _strip_quotes(raw_word) if raw_word else ""

            if not single_quoted:
                body_word = _expand_subst_in_text(
                    body_word, work_dir, timeout, depth + 1, deadline, subst_count,
                )

            body = body_word + "\n"
            body = body[:MAX_HEREDOC_BODY]
            sentinel = f"\x01H{next_hd_id}\x01"
            next_hd_id += 1
            expansion.heredoc_bodies[sentinel] = body
            output.append(" " + sentinel)
            continue

        # ---- heredoc: <<- (3-char, before <<) ----
        if command[i : i + 3] == '<<-':
            output.append('<<-')
            i += 3
            # skip whitespace after <<-
            while i < n and command[i] in (' ', '\t'):
                i += 1
            # read delimiter word (quote-aware)
            delim_chars: list[str] = []
            dq: Optional[str] = None
            delim_quoted = False
            while i < n:
                ch = command[i]
                if dq is not None:
                    delim_chars.append(ch)
                    if ch == dq:
                        dq = None
                    i += 1
                elif ch in ("'", '"'):
                    delim_quoted = True
                    dq = ch
                    delim_chars.append(ch)
                    i += 1
                elif ch in (' ', '\t', '\n', ';', '|', '&'):
                    break
                else:
                    delim_chars.append(ch)
                    i += 1
            delimiter = _strip_quotes("".join(delim_chars))

            # Skip to end of current line
            while i < n and command[i] != '\n':
                i += 1
            if i < n:
                i += 1  # consume newline

            # Collect body lines until delimiter
            body_lines: list[str] = []
            found = False
            while i < n:
                line_start = i
                while i < n and command[i] != '\n':
                    i += 1
                line = command[line_start:i]
                if i < n:
                    i += 1  # consume newline

                # Strip leading TABs from line for comparison
                tab_count = 0
                for ch_val in line:
                    if ch_val == '\t':
                        tab_count += 1
                    else:
                        break
                stripped_line = line[tab_count:]

                if stripped_line == delimiter:
                    found = True
                    break

                body_lines.append(line[tab_count:])

            if not found:
                raise ValueError(f"heredoc delimiter {delimiter!r} not found")

            body = "\n".join(body_lines) + "\n"
            body = body[:MAX_HEREDOC_BODY]

            if not delim_quoted:
                body = _expand_subst_in_text(
                    body, work_dir, timeout, depth + 1, deadline, subst_count,
                )

            sentinel = f"\x01H{next_hd_id}\x01"
            next_hd_id += 1
            expansion.heredoc_bodies[sentinel] = body
            output.append(" " + sentinel)
            continue

        # ---- heredoc: << (2-char, after <<- and <<<) ----
        if command[i : i + 2] == '<<':
            output.append('<<')
            i += 2
            # skip whitespace after <<
            while i < n and command[i] in (' ', '\t'):
                i += 1
            # read delimiter word (quote-aware)
            delim_chars = []
            dq = None
            delim_quoted = False
            while i < n:
                ch = command[i]
                if dq is not None:
                    delim_chars.append(ch)
                    if ch == dq:
                        dq = None
                    i += 1
                elif ch in ("'", '"'):
                    delim_quoted = True
                    dq = ch
                    delim_chars.append(ch)
                    i += 1
                elif ch in (' ', '\t', '\n', ';', '|', '&'):
                    break
                else:
                    delim_chars.append(ch)
                    i += 1
            delimiter = _strip_quotes("".join(delim_chars))

            # Skip to end of current line
            while i < n and command[i] != '\n':
                i += 1
            if i < n:
                i += 1  # consume newline

            # Collect body lines until delimiter
            body_lines = []
            found = False
            while i < n:
                line_start = i
                while i < n and command[i] != '\n':
                    i += 1
                line = command[line_start:i]
                if i < n:
                    i += 1  # consume newline

                if line == delimiter:
                    found = True
                    break

                body_lines.append(line)

            if not found:
                raise ValueError(f"heredoc delimiter {delimiter!r} not found")

            body = "\n".join(body_lines) + "\n"
            body = body[:MAX_HEREDOC_BODY]

            if not delim_quoted:
                body = _expand_subst_in_text(
                    body, work_dir, timeout, depth + 1, deadline, subst_count,
                )

            sentinel = f"\x01H{next_hd_id}\x01"
            next_hd_id += 1
            expansion.heredoc_bodies[sentinel] = body
            output.append(" " + sentinel)
            continue

        # ---- regular character ----
        output.append(c)
        i += 1

    if quote is not None:
        raise ValueError("Unbalanced quotes in command")

    return "".join(output), expansion


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
    segments: list[str],
    work_dir: Path,
    expansion: Optional[Expansion] = None,
) -> tuple[int, str]:
    """Launch a pipe-connected pipeline in the background and return immediately.

    Each segment is built through ``_build_invocation`` (same allowlist/sandbox
    checks as ``_run_pipeline``).  Intermediate stages' stdout feeds the next
    stage's stdin exactly as in a foreground pipeline; all stderr, plus the
    last stage's stdout, are redirected to a timestamped log file under
    ``work_dir`` so the parent never blocks on pipe buffers.

    Returns ``(0, message)`` with the PID and log path.
    """
    invocations: list[tuple[list[str], Optional[dict], list[Redirect]]] = []
    for i, seg in enumerate(segments):
        binary, sandbox_args, env, cfg, redirects = _build_invocation(
            seg, work_dir, expansion=expansion,
        )
        if sandbox_args is None:
            if binary is None:
                continue  # empty segment inside a pipeline
            return 1, binary  # error message
        if cfg.get("is_local_binary") and not _binary_still_contained(binary, work_dir):
            return 1, f"Local binary no longer valid inside working directory: {binary}"
        # Reject heredoc/here-string on non-first pipeline stages (matches
        # _run_pipeline_core).
        if i > 0:
            for r in redirects:
                if r.fd == 0:
                    return 1, (
                        f"heredoc/here-string not allowed on non-first "
                        f"pipeline stage: {seg}"
                    )
        invocations.append((sandbox_args, env, redirects))

    if not invocations:
        return 0, ""

    # Reject stdout redirects on intermediate pipe stages.
    for i, (_sa, _env, redirects) in enumerate(invocations[:-1]):
        for r in redirects:
            if r.fd == 1:
                return 1, (
                    f"Cannot redirect stdout of intermediate pipe stage: "
                    f"{segments[i]}"
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
    try:
        for i, (_sa, _env, redirects) in enumerate(invocations):
            is_last = i == len(invocations) - 1
            def_stdout = LOG_SENTINEL if is_last else subprocess.PIPE
            def_stderr = LOG_SENTINEL if is_last else subprocess.PIPE
            st, et, tc, rpt, _srf, _stdin_b = _resolve_fd_targets(
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
    except OSError as e:
        for fh in all_to_close:
            try:
                fh.close()
            except OSError:
                pass
        return 1, f"Error opening redirect target: {e}"

    procs: list[subprocess.Popen] = []
    prev: Optional[subprocess.Popen] = None
    try:
        for i, (sandbox_args, env, _redirects) in enumerate(invocations):
            p = subprocess.Popen(
                sandbox_args,
                stdin=prev.stdout if prev is not None else None,
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

    _start_reaper()
    return 0, "\n".join(msg_parts)


@mcp.tool()
def shell_run(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Run a shell command in a pledge sandbox.

    Commands are executed via the pledge-wrapped sandbox binary with
    restricted system call promises. Only allowlisted commands can run.

    Pipes (`|`) are supported: the command string is split on top-level `|`
    into pipeline stages, and each stage's stdout is fed to the next stage's
    stdin, exactly as in a real shell pipe. Pipelines can be chained with `;`,
    `&&`, and `||`; every stage/segment is run through its own sandbox and
    checked against the allowlist independently. `&&` runs its pipeline only if
    the previous one succeeded; `||` only if it failed; `;` always runs.

    Backgrounding (`&`) is supported: a bare `&` terminates the current
    pipeline, launches it in the background (detached via a new session), and
    returns immediately with the PID and a log path. The next pipeline always
    runs (like `;`). `&&`/`||` short-circuit is still honoured before a
    backgrounded pipeline is launched.

    Redirects (``>``, ``>>``, ``2>``, ``2>>``, ``2>&1``, ``1>&2``) are supported
    per command segment. Redirect targets must stay inside the working
    directory. Quoted operators (``echo ">"``) are treated as literal
    arguments, not redirects. Only output redirects are supported; input
    redirects (``<``, ``<<``) are rejected. Redirecting stdout of an
    intermediate pipe stage is rejected (stderr redirects and ``2>&1`` are
    allowed on intermediate stages).

    Heredocs (``<<EOF``, ``<<'EOF'``, ``<<-EOF``) and here-strings
    (``<<<'literal'``) feed literal text to the command's stdin.
    Command substitution (``$(command ...)``) recursively executes the inner
    command and splices its stdout as a single argument word.

    Args:
        command: The command to run (e.g., "git status", "ls | grep foo",
            "cd build && make test")
        cwd:     Working directory (must be within allowed paths)
        timeout: Timeout in seconds (default 30, max 300)
    """
    timeout = min(timeout, MAX_TIMEOUT)

    # Validate cwd first — resolve once, use for validation, resolution and
    # execution. Required before resolving so local binaries can be located.
    raw_cwd = cwd or "."
    try:
        work_dir = Path(raw_cwd).expanduser().resolve()
    except Exception:
        return f"Invalid path: {raw_cwd}"

    err = _validate_cwd(work_dir, raw_cwd)
    if err:
        return err

    # Expand $(...) heredocs and here-strings BEFORE splitting.
    try:
        expanded, expansion = _expand_command(command, work_dir, timeout, depth=0)
    except ValueError as e:
        return str(e)

    # Split into allowlist-checked pipelines on ; / && / || / &, with each
    # pipeline being a list of `|`-separated stages.
    try:
        pipelines = _split_command(expanded)
    except ValueError as e:
        return str(e)
    if not pipelines:
        return "Empty command."

    # Single-command fast path — preserves the exact prior behaviour.
    if len(pipelines) == 1 and pipelines[0][0] is None and len(pipelines[0][1]) == 1:
        if pipelines[0][2]:
            _rc, out = _run_background(pipelines[0][1], work_dir, expansion=expansion)
            return out if out else "(no output)"
        _rc, out = _run_segment(pipelines[0][1][0], work_dir, timeout,
                                expansion=expansion)
        return out if out else "(no output)"

    # Multi-pipeline chain: run each pipeline through the sandbox, applying
    # `&&` / `||` short-circuit semantics based on the previous pipeline's exit
    # code. Pipelines are independent processes (no shared shell variables), but
    # they share the same cwd, so file-based state persists across stages.
    outputs: list[str] = []
    prev_rc = 0
    ran_any = False

    for op, stages, backgrounded in pipelines:
        joined = " | ".join(stages)
        if op == "&&" and ran_any and prev_rc != 0:
            outputs.append(f"(skipped: previous command exited {prev_rc}) — {joined}")
            continue
        if op == "||" and ran_any and prev_rc == 0:
            outputs.append("(skipped: previous command succeeded) — " + joined)
            continue

        if backgrounded:
            rc, out = _run_background(stages, work_dir, expansion=expansion)
            ran_any = True
            # Leave prev_rc unchanged — backgrounded exit code is unknown.
        elif len(stages) == 1:
            rc, out = _run_segment(stages[0], work_dir, timeout, expansion=expansion)
            prev_rc = rc
            ran_any = True
        else:
            rc, out = _run_pipeline(stages, work_dir, timeout, expansion=expansion)
            prev_rc = rc
            ran_any = True
        if out:
            outputs.append(out)

    if not outputs:
        return "(no output)"

    return "\n".join(outputs)


@mcp.tool()
def shell_list() -> str:
    """List all allowed commands and their descriptions."""
    lines = ["Allowed commands:"]
    for name, cfg in COMMANDS.items():
        lines.append(f"  {name}: {cfg['description']}")
        lines.append(f"    Promises: {cfg['promises']}")
    lines.append(f"  busybox applets ({len(BUSYBOX_APPLETS)}): {', '.join(BUSYBOX_APPLETS)}")
    lines.append(f"    Binary: {BUSYBOX_BIN.resolve()}")
    lines.append(f"    Promises: stdio rpath wpath cpath")
    lines.append("")
    lines.append("Local binaries: any executable file inside the working")
    lines.append("    directory (or below) can be run by name or path, e.g.")
    lines.append("    './scripts/foo' or 'target/release/bar'. Paths escaping")
    lines.append("    the working directory are rejected.")
    lines.append("")
    lines.append("Command chaining: ';', '&&', '||', and '&' are supported. Each")
    lines.append("    pipeline is run through the sandbox and checked against the")
    lines.append("    allowlist independently. '&&' runs only after success, '||'")
    lines.append("    only after failure, ';' always. '&' backgrounds a pipeline")
    lines.append("    and returns immediately with the PID and log path. Pipes ('|')")
    lines.append("    are supported: each stage's stdout feeds the next stage's stdin.")
    lines.append("    Redirects (>, >>, 2>, 2>>, 2>&1, 1>&2) are supported per segment;")
    lines.append("    targets must stay inside the working directory. Quoted operators")
    lines.append("    are not treated as redirects. Heredocs (<<, <<-, <<') and")
    lines.append("    here-strings (<<<) feed stdin; command substitution ($(...))")
    lines.append(f"    splices stdout as a single arg (depth {MAX_SUBST_DEPTH}, count")
    lines.append(f"    {MAX_SUBST_COUNT}, max output {MAX_SUBST_OUTPUT:,}, max body")
    lines.append(f"    {MAX_HEREDOC_BODY:,}).")
    lines.append("")
    lines.append(f"Sandbox binary: {SANDBOX_BIN.resolve()}")
    lines.append(f"Allowed directories: {', '.join(DEFAULT_ALLOWED_DIRS)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the MCP server with stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
