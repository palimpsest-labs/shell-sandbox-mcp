"""Shell command parser — lexer, AST, and parse functions.

Replaces the hand-rolled char-by-char parsing passes in server.py with a
proper lexer + recursive-descent parser.  Exports Redirect and Expansion
(which exposes opaque lookups ``arg_for(part)`` / ``heredoc_for(part)``)
so existing imports keep working.  The ``\x01A`` / ``\x01H`` sentinel scheme
is encapsulated inside this module — no callers outside parser.py should
reconstruct sentinel keys.

Public API
----------
- ParseError, Redirect, Expansion
- parse_command(text, capture_fn, ...) → (cleaned, expansion, program)
- split_legacy(text) → list[(op|None, [stages], bg)]
- extract_redirects(segment, expansion=None, work_dir=None) → (args, redirects, err)
- serialize_program(program) → str
- program_to_chain(program) → list[(op|None, [CommandNode], bg)]
- Lexer, Token, TokenKind, CommandNode, PipelineNode, AndOrNode, ProgramNode (AST)
"""

from __future__ import annotations

import enum
import glob as _glob
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Mapping, Optional

from .config import MAX_HEREDOC_BODY, MAX_SUBST_COUNT, MAX_SUBST_DEPTH, MAX_SUBST_OUTPUT


# ---------------------------------------------------------------------------
# Sentinel patterns (moved from server.py; same values)
# ---------------------------------------------------------------------------

_SENTINEL_ARG = re.compile(r"\x01A(\d+)\x01")
_SENTINEL_HD  = re.compile(r"\x01H(\d+)\x01")
_BRACED_VAR_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_VAR_NAME_RE   = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ---------------------------------------------------------------------------
# Expansion (side table)
# ---------------------------------------------------------------------------

@dataclass
class Expansion:
    """Side table holding resolved $() output words and heredoc/here-string bodies.

    Lookups MUST go through the opaque ``arg_for(part)`` /
    ``heredoc_for(part)`` methods.  Do NOT access the internal dicts directly.
    """
    _arg_values: dict[str, str] = field(default_factory=dict)
    _heredoc_bodies: dict[str, str] = field(default_factory=dict)

    # -- public opaque API ------------------------------------------------

    def arg_for(self, part) -> Optional[str]:
        """Return the resolved $() value for *part*, or None.

        *part* must be a :class:`WordPart` whose ``is_arg_sentinel`` is True.
        """
        if not part.is_arg_sentinel:
            return None
        return self._arg_values.get(part.text)

    def heredoc_for(self, part) -> Optional[str]:
        """Return the resolved heredoc/here-string body for *part*, or None.

        *part* must be a :class:`WordPart` whose ``is_hd_sentinel`` is True.
        """
        if not part.is_hd_sentinel:
            return None
        return self._heredoc_bodies.get(part.text)

    # -- internal write helpers (for parser.py only) ----------------------

    def _set_arg_for(self, sentinel: str, value: str) -> None:
        """Record a $() output for a sentinel key."""
        self._arg_values[sentinel] = value

    def _set_heredoc_for(self, sentinel: str, body: str) -> None:
        """Record a heredoc/here-string body for a sentinel key."""
        self._heredoc_bodies[sentinel] = body


# ---------------------------------------------------------------------------
# Redirect dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Redirect:
    """A parsed shell redirect operator extracted from a command segment."""
    fd: int                                  # 0 (stdin), 1 (stdout), or 2 (stderr)
    op: Literal[">", ">>", ">&", "<", "<<", "<<<", "<<-"]
    target_path: Optional[str] = None        # resolved absolute path (None for ">&")
    target_fd: Optional[int] = None          # source fd for ">&" (1 or 2); else None
    raw_target: Optional[str] = None         # user-typed target (for messages)
    body: Optional[str] = None               # literal stdin content for heredoc/here-string
    strip_tabs: bool = False                 # <<- semantics (strip leading TABs)

    def apply(self, plan, *, snapshot_2gt1: bool) -> None:
        """Apply this redirect to an :class:`FdPlan` in place.

        Mutates *plan* in place (stdout/stderr targets, ``to_close``,
        ``report``, stdin fields, ``shared_read_fd``).  ``snapshot_2gt1``
        tells a ``2>&1`` whether a later ``>file`` will redirect stdout, in
        which case stderr should snapshot stdout's current destination via a
        shared pipe rather than merge into it (POSIX ``2>&1 >file``
        semantics).
        """
        if self.op == "<":
            if plan.stdin_file is not None or plan.stdin_bytes is not None:
                raise ValueError("Multiple stdin redirects in one segment")
            try:
                fd = os.open(self.target_path, os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError:
                raise ValueError(f"Input redirect file not found: {self.raw_target}")
            except OSError as e:
                raise ValueError(f"Cannot open input redirect {self.raw_target}: {e}")
            plan.stdin_file = os.fdopen(fd, "rb")
            plan.to_close.append(plan.stdin_file)
            plan.report.append(f"[stdin <- {self.raw_target}]")
        elif self.op in ("<<", "<<-", "<<<"):
            # Heredoc / here-string: only one stdin redirect per segment
            # (already enforced in _extract_redirects).
            if plan.stdin_bytes is not None or plan.stdin_file is not None:
                raise ValueError("Multiple stdin redirects in one segment")
            plan.stdin_bytes = (self.body or "").encode("utf-8")
            if self.op == "<<<":
                plan.report.append("[stdin <<<]")
            elif self.op == "<<-":
                plan.report.append("[stdin <<-]")
            else:
                plan.report.append("[stdin <<]")
        elif self.op in (">", ">>"):
            # O_NOFOLLOW closes the symlink-swap TOCTOU: the path was
            # containment-validated earlier, so a redirect target must not be
            # a symlink pointing outside the work tree at open time.
            flags = os.O_WRONLY | os.O_CREAT
            flags |= os.O_TRUNC if self.op == ">" else os.O_APPEND
            flags |= os.O_NOFOLLOW
            # Contrast with the secret 0o600 in policy.py: a redirect target is
            # NOT secret, so use 0o666 — the mode is masked by the process umask
            # at open, so a restrictive umask still applies.
            fd = os.open(self.target_path, flags, 0o666)
            fh = os.fdopen(fd, "wb" if self.op == ">" else "ab")
            plan.to_close.append(fh)
            if self.fd == 1:
                plan.stdout = fh
                arrow = "->" if self.op == ">" else "->>"
                plan.report.append(f"[stdout {arrow} {self.raw_target}]")
            else:  # fd == 2
                plan.stderr = fh
                arrow = "->" if self.op == ">" else "->>"
                plan.report.append(f"[stderr {arrow} {self.raw_target}]")
        elif self.op == ">&":
            if self.fd == 2 and self.target_fd == 1:  # 2>&1
                # If a LATER `>file` will redirect stdout, snapshot stdout's
                # current destination so stderr isn't dragged along with it
                # (matches POSIX: `2>&1 >file` puts stderr on the original
                # stdout). Otherwise a plain subprocess.STDOUT (merge stderr
                # into stdout's target) is correct and lighter.  The
                # ``later_stdout_redirect`` lookahead is supplied via the
                # ``snapshot_2gt1`` parameter (computed by RedirectPlan).
                if (
                    snapshot_2gt1
                    and isinstance(plan.stdout, int)
                    and plan.stdout == subprocess.PIPE
                ):
                    plan.share_stdout_stderr_via_pipe()
                else:
                    plan.stderr = subprocess.STDOUT
                plan.report.append("[stderr -> stdout]")
            elif self.fd == 1 and self.target_fd == 2:  # 1>&2
                if isinstance(plan.stderr, int) and plan.stderr == subprocess.PIPE:
                    # Create a shared pipe so both stdout and stderr write to
                    # the same fd; parent reads from the read end.
                    plan.share_stdout_stderr_via_pipe()
                else:
                    # stderr is a real file handle (or a shared-pipe fd);
                    # just point stdout at the same target.
                    plan.stdout = plan.stderr
                plan.report.append("[stdout -> stderr]")


# ---------------------------------------------------------------------------
# ParseError
# ---------------------------------------------------------------------------

class ParseError(ValueError):
    """Raised when a shell construct is rejected by the parser."""
    pass


# ---------------------------------------------------------------------------
# Token kinds
# ---------------------------------------------------------------------------

class TokenKind(enum.Enum):
    # Structural
    WORD = 1            # a shell word (with quote chars preserved)
    WS = 2              # whitespace (space or tab)
    NEWLINE = 3         # literal newline

    # Operators (chain level)
    PIPE = 10           # |
    SEMI = 11           # ;
    AND_AND = 12        # &&
    OR_OR = 13          # ||
    BG = 14             # bare & (background)

    # Redirect operators
    R_OUT = 20          # > or 1> or 2>
    R_APPEND = 21       # >> or 1>> or 2>>
    R_IN = 22           # <
    R_FD_DUP = 23       # >& (2>&1 or 1>&2)
    R_HEREDOC = 24      # <<
    R_HEREDOC_STRIP = 25  # <<-
    R_HERESTRING = 26   # <<<

    # Subst
    SUBST = 30          # $(...) — value is raw inner text
    VARREF = 31         # $VAR or ${VAR} — value is the variable name


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    value: str          # literal text
    pos: int            # byte offset in original input
    fd: int = 0         # redirect fd number (0=default, 1 or 2 for 1> / 2>)
    target_fd: int = 0  # for R_FD_DUP: target fd (1 or 2)
    quoted_delim: bool = False  # for heredoc: delimiter is quoted (body literal)
    strip_tabs: bool = False    # for <<- heredocs
    body: Optional[str] = None  # for heredoc/here-string: the raw body text


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WordPart:
    """One component of a shell word."""
    text: str                              # quote-stripped text (or sentinel key)
    raw: str = ""                          # raw text for serialization (defaults to text)
    is_sentinel: bool = False              # True if this is a subst/heredoc sentinel
    is_quoted: bool = False                # True if originated inside quotes

    @property
    def is_arg_sentinel(self) -> bool:
        return self.is_sentinel and "\x01A" in self.text

    @property
    def is_hd_sentinel(self) -> bool:
        return self.is_sentinel and "\x01H" in self.text

    def serialized(self) -> str:
        """Return the text for use in serialize_program (sentinel form)."""
        if self.is_sentinel:
            return self.text
        return self.raw if self.raw else self.text

    def display_serialized(self) -> str:
        """Return the human-readable form for error/display output."""
        if self.is_sentinel:
            return self.raw
        return self.raw if self.raw else self.text


@dataclass(frozen=True)
class Word:
    """A full shell word (may be formed from multiple parts)."""
    parts: tuple[WordPart, ...] = ()

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts)

    def serialized(self) -> str:
        return "".join(p.serialized() for p in self.parts)

    def display_serialized(self) -> str:
        return "".join(p.display_serialized() for p in self.parts)


@dataclass(frozen=True)
class RedirectSpec:
    """A redirect operator plus its target word."""
    fd: int
    op: Literal[">", ">>", ">&", "<", "<<", "<<<", "<<-"]
    target: Word               # the target word (may be a sentinel)
    strip_tabs: bool = False
    raw_operator: str = ""     # e.g. "2>&1" or "1>&2" or ">" or ">>"
    glued_target: bool = False # True when target was adjacent to operator (e.g. 2>err)


@dataclass(frozen=True)
class CommandNode:
    """A single command: words + redirects."""
    words: tuple[Word, ...] = ()
    redirects: tuple[RedirectSpec, ...] = ()
    backgrounded: bool = False


@dataclass(frozen=True)
class PipelineNode:
    """A pipe-connected sequence of commands."""
    commands: tuple[CommandNode, ...] = ()


@dataclass(frozen=True)
class AndOrNode:
    """One element of an and-or list."""
    operator: Optional[str]   # None, ";", "&&", "||"
    pipeline: PipelineNode
    backgrounded: bool = False


@dataclass(frozen=True)
class ProgramNode:
    """Top-level program: list of and-or chains."""
    chains: tuple[AndOrNode, ...] = ()


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class Lexer:
    """Escape-aware single-pass tokenizer for shell commands.

    Produces a flat list of Token objects.  Backslash escapes, quoting,
    ``$(...)`` substitution spans, heredoc body collection, and redirect
    operators are all recognized.

    Unsupported constructs (backticks, ``<(...)``, ``>(...)``) raise
    ``ParseError`` immediately, but only when appearing *outside* quotes.
    """

    def __init__(self, command: str, *, replay_mode: bool = False):
        self._cmd = command
        self._n = len(command)
        self._pos = 0
        self._tokens: list[Token] = []
        self._replay_mode = replay_mode

    def tokenize(self) -> list[Token]:
        """Run the lexer and return the token list."""
        self._pos = 0
        self._tokens = []
        while self._pos < self._n:
            self._lex_one()
        return self._tokens

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        p = self._pos + offset
        if p < self._n:
            return self._cmd[p]
        return ""

    def _advance(self, n: int = 1) -> None:
        self._pos += n

    def _emit(self, kind: TokenKind, value: str, **kw) -> None:
        kw.setdefault("pos", self._pos - len(value))
        self._tokens.append(Token(kind=kind, value=value, **kw))

    # ------------------------------------------------------------------
    # main dispatch
    # ------------------------------------------------------------------

    def _lex_one(self) -> None:
        c = self._cmd[self._pos]

        # whitespace
        if c in (' ', '\t'):
            start = self._pos
            while self._pos < self._n and self._cmd[self._pos] in (' ', '\t'):
                self._pos += 1
            self._tokens.append(Token(TokenKind.WS, self._cmd[start:self._pos], start))
            return

        # newline
        if c == '\n':
            self._emit(TokenKind.NEWLINE, '\n')
            self._advance()
            return

        # --- unsupported: backtick ---
        if c == '`':
            raise ParseError("Backtick command substitution is not supported; use $(...)")

        # At a word-start position — check for operators / redirects / subst
        rem = self._cmd[self._pos:]

        # --- $( ... ) ---
        if c == '$' and self._peek(1) == '(':
            if self._peek(2) == '(':
                raise ParseError("Arithmetic expansion $((...)) is not supported")
            self._lex_subst()
            return

        # --- ${VAR} / $VAR (variable reference) ---
        if c == '$':
            nxt = self._peek(1)
            if nxt == '{' and _BRACED_VAR_RE.match(self._cmd[self._pos:]):
                self._lex_varref_braced()
                return
            if nxt and (nxt.isalpha() or nxt == '_'):
                self._lex_varref()
                return
            # else fall through to _lex_word (literal $)

        # --- chain operators (longest-match first) ---
        if rem.startswith('&&'):
            self._emit(TokenKind.AND_AND, '&&')
            self._advance(2)
            return
        if rem.startswith('||'):
            self._emit(TokenKind.OR_OR, '||')
            self._advance(2)
            return
        if c == '|':
            self._emit(TokenKind.PIPE, '|')
            self._advance()
            return
        if c == ';':
            self._emit(TokenKind.SEMI, ';')
            self._advance()
            return
        if c == '&':
            self._emit(TokenKind.BG, '&')
            self._advance()
            return

        # --- redirect operators (checked at word-start only) ---
        # --- unsupported: process substitution ---
        if c == '<' and self._peek(1) == '(':
            raise ParseError("Process substitution <(...) is not supported")
        if c == '>' and self._peek(1) == '(':
            raise ParseError("Process substitution >(...) is not supported")

        # fd-dup: N>&M
        if (len(rem) >= 4 and rem[0].isdigit() and rem[1:3] == '>&'
                and rem[3].isdigit()):
            fd = int(rem[0])
            target = int(rem[3])
            # Must NOT be followed by alphanumeric or _ (so 2>&1x is not fd-dup)
            if len(rem) > 4 and (rem[4].isalnum() or rem[4] == '_'):
                # Partial fd-dup: N>&Mx (e.g. 2>&1x, 1>&2y).
                # Emit N> as a redirect operator, then the remainder
                # (&Mx…) becomes the redirect target word (matching the
                # string-path behaviour in _extract_from_string).
                if fd not in (1, 2):
                    raise ParseError(
                        f"Redirects only support fds 1 and 2 (got {fd})"
                    )
                self._emit(TokenKind.R_OUT, rem[:2], fd=fd)
                self._advance(2)
                # Read the remaining &Mx… as a bare WORD token.
                # First char is always '&' (literal, not backgrounding).
                # Continue until whitespace / pipe / semicolon / another '&'.
                start = self._pos
                chars: list[str] = []
                while self._pos < self._n:
                    c = self._cmd[self._pos]
                    if c in (' ', '\t', '\n', '|', ';'):
                        break
                    if c == '&' and len(chars) > 0:
                        break  # second & → && or BG
                    chars.append(c)
                    self._pos += 1
                value = "".join(chars)
                if value:
                    self._tokens.append(Token(TokenKind.WORD, value, start))
                return
            elif fd not in (1, 2):
                raise ParseError(f"Redirects only support fds 1 and 2 (got {fd})")
            elif target not in (1, 2):
                raise ParseError("Redirect dup target fd must be 1 or 2")
            else:
                self._emit(TokenKind.R_FD_DUP, rem[:4], fd=fd, target_fd=target)
                self._advance(4)
                return

        # 2>>
        if rem.startswith('2>>'):
            self._emit(TokenKind.R_APPEND, '2>>', fd=2)
            self._advance(3)
            return
        # 1>>
        if rem.startswith('1>>'):
            self._emit(TokenKind.R_APPEND, '1>>', fd=1)
            self._advance(3)
            return
        # 2>
        if rem.startswith('2>'):
            self._emit(TokenKind.R_OUT, '2>', fd=2)
            self._advance(2)
            return
        # 1>
        if rem.startswith('1>'):
            self._emit(TokenKind.R_OUT, '1>', fd=1)
            self._advance(2)
            return

        # <<< (here-string) — before <<
        if rem.startswith('<<<'):
            self._lex_herestring()
            return

        # <<- (heredoc strip tabs) — before <<
        if rem.startswith('<<-'):
            self._lex_heredoc(strip_tabs=True)
            return

        # >>
        if rem.startswith('>>'):
            self._emit(TokenKind.R_APPEND, '>>', fd=1)
            self._advance(2)
            return

        # << (heredoc)
        if rem.startswith('<<'):
            self._lex_heredoc(strip_tabs=False)
            return

        # >
        if c == '>':
            self._emit(TokenKind.R_OUT, '>', fd=1)
            self._advance()
            return

        # <
        if c == '<':
            self._emit(TokenKind.R_IN, '<', fd=0)
            self._advance()
            return

        # digit + > that aren't 1> / 2> (e.g. 3> or 0>)
        if len(rem) >= 2 and rem[0].isdigit() and rem[1] == '>':
            fd = int(rem[0])
            raise ParseError(f"Redirects only support fds 1 and 2 (got {fd})")

        # -- none of the above: it's a regular word --
        self._lex_word()

    # ------------------------------------------------------------------
    # word lexing (escape-aware)
    # ------------------------------------------------------------------

    def _lex_word(self) -> None:
        """Read a shell word, handling escapes and quotes."""
        start = self._pos
        chars: list[str] = []
        i = self._pos

        while i < self._n:
            c = self._cmd[i]

            if c == '\\':
                # backslash outside quotes: escape next character
                if i + 1 < self._n:
                    nxt = self._cmd[i + 1]
                    if nxt == '$':
                        chars.append('\\')
                        chars.append('$')
                    else:
                        chars.append(nxt)
                    i += 2
                else:
                    chars.append('\\')
                    i += 1
                continue

            if c == "'":
                # single quote: read until closing ', everything literal
                # Store raw form with quotes for serialization
                sq_start = i
                i += 1
                while i < self._n and self._cmd[i] != "'":
                    i += 1
                if i < self._n:
                    i += 1  # closing quote
                chars.append(self._cmd[sq_start:i])
                continue

            if c == '"':
                # double quote: read until closing ", handle escapes
                dq_start = i
                i += 1
                paren_depth = 0  # track $(...) nesting inside dq
                while i < self._n:
                    ch = self._cmd[i]
                    if ch == '\\' and i + 1 < self._n:
                        nxt = self._cmd[i + 1]
                        if nxt in ('"', '$', '\\'):
                            i += 2  # skip the backslash, keep the escaped char
                        elif nxt == '\n':
                            i += 2  # line continuation
                        else:
                            i += 2  # \X stays literal inside double quotes
                    elif ch == '$' and i + 1 < self._n and self._cmd[i + 1] == '(':
                        paren_depth += 1
                        i += 1  # skip $
                    elif ch == ')' and paren_depth > 0:
                        paren_depth -= 1
                        i += 1
                    elif ch == '"' and paren_depth == 0:
                        break  # closing quote (not inside $())
                    else:
                        i += 1
                if i < self._n:
                    i += 1  # closing quote
                chars.append(self._cmd[dq_start:i])
                continue

            # word terminators (outside quotes)
            if c in (' ', '\t', '\n', '|', ';', '&'):
                break

            # $ outside quotes → let the main loop handle it (not mid-word)
            if c == '$' and i + 1 < self._n:
                nxt = self._cmd[i + 1]
                if nxt == '(':
                    break
                if nxt == '{' and _BRACED_VAR_RE.match(self._cmd[i:]):
                    break
                if nxt and (nxt.isalpha() or nxt == '_'):
                    break

            # redirect-like chars mid-word → stay part of word
            chars.append(c)
            i += 1

        self._pos = i
        value = "".join(chars)
        if value:
            self._tokens.append(Token(TokenKind.WORD, value, start))

    # ------------------------------------------------------------------
    # $( ... )  substitution
    # ------------------------------------------------------------------

    def _lex_subst(self) -> None:
        """Lex a $( ... ) token; raises on $((."""
        start = self._pos
        self._pos += 2  # skip $(
        depth = 1
        quote: Optional[str] = None

        while self._pos < self._n and depth > 0:
            c = self._cmd[self._pos]
            if quote is not None:
                if c == quote:
                    quote = None
            elif c in ("'", '"'):
                quote = c
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            self._pos += 1

        if depth != 0:
            raise ValueError("Unbalanced $( ... )")

        inner = self._cmd[start + 2 : self._pos - 1]
        self._tokens.append(Token(TokenKind.SUBST, inner, start))

    # ------------------------------------------------------------------
    # $VAR / ${VAR}  variable reference
    # ------------------------------------------------------------------

    def _lex_varref(self) -> None:
        """Lex a bare $VAR token — read the identifier name."""
        start = self._pos
        self._pos += 1  # skip $
        m = _VAR_NAME_RE.match(self._cmd, self._pos)
        assert m is not None  # precondition: next char is [A-Za-z_]
        name = m.group(0)
        self._pos = m.end()
        self._tokens.append(Token(TokenKind.VARREF, name, start))

    def _lex_varref_braced(self) -> None:
        """Lex a ${VAR} token — read the identifier between braces."""
        start = self._pos
        m = _BRACED_VAR_RE.match(self._cmd[self._pos:])
        assert m is not None  # precondition: matches ^\$\{[A-Za-z_][A-Za-z0-9_]*\}
        full = m.group(0)
        name = full[2:-1]  # strip ${ and }
        self._pos += m.end()
        self._tokens.append(Token(TokenKind.VARREF, name, start))

    # ------------------------------------------------------------------
    # here-string  <<< word
    # ------------------------------------------------------------------

    def _lex_herestring(self) -> None:
        """Lex a here-string: <<< followed by a word."""
        start = self._pos
        self._pos += 3  # skip <<<

        # skip whitespace
        while self._pos < self._n and self._cmd[self._pos] in (' ', '\t'):
            self._pos += 1

        # read the target word (quote-aware, stop at newline)
        chars: list[str] = []
        quote: Optional[str] = None
        quoted_delim = False
        while self._pos < self._n:
            c = self._cmd[self._pos]
            if quote is not None:
                chars.append(c)
                if c == quote:
                    quote = None
                self._pos += 1
            elif c in ("'", '"'):
                if c == "'":
                    quoted_delim = True  # single-quoted word → literal
                quote = c
                chars.append(c)
                self._pos += 1
            elif c == '\n':
                break
            else:
                chars.append(c)
                self._pos += 1

        raw_target = "".join(chars)
        self._tokens.append(Token(
            TokenKind.R_HERESTRING, '<<<', start,
            fd=0, body=raw_target,
            quoted_delim=quoted_delim,
        ))

    # ------------------------------------------------------------------
    # heredoc  <<[-]DELIM
    # ------------------------------------------------------------------

    def _lex_heredoc(self, strip_tabs: bool) -> None:
        """Lex a heredoc: << or <<- followed by delimiter, then collect body."""
        start = self._pos
        op_len = 3 if strip_tabs else 2
        op_text = "<<-" if strip_tabs else "<<"
        self._pos += op_len

        # skip whitespace after operator
        while self._pos < self._n and self._cmd[self._pos] in (' ', '\t'):
            self._pos += 1

        # read delimiter (quote-aware, backslash-aware)
        delim_start = self._pos  # saved for sentinel-shortcut rewind
        delim_chars: list[str] = []
        dq: Optional[str] = None
        quoted = False

        # Backslash before delimiter → quoted
        if self._pos < self._n and self._cmd[self._pos] == '\\':
            quoted = True
            self._pos += 1

        while self._pos < self._n:
            c = self._cmd[self._pos]
            if dq is not None:
                delim_chars.append(c)
                if c == dq:
                    dq = None
                self._pos += 1
            elif c in ("'", '"'):
                quoted = True
                dq = c
                delim_chars.append(c)
                self._pos += 1
            elif c in (' ', '\t', '\n', ';', '|', '&'):
                break
            else:
                delim_chars.append(c)
                self._pos += 1

        raw_delim = "".join(delim_chars)
        delimiter = _strip_quotes(raw_delim)

        # --- sentinel-delimiter shortcut ---
        # When the delimiter is a sentinel (e.g. \x01H0\x01) the body has
        # already been extracted and stored in the Expansion side table.
        # Rewind so the sentinel text is re-lexed as a plain WORD token,
        # allowing _build_redirect_spec to detect and reuse the original ID.
        # ONLY in replay mode — in populate mode a literal \x01H<N>\x01
        # delimiter is a real user-typed delimiter whose body must be
        # collected normally.
        if self._replay_mode and _SENTINEL_HD.fullmatch(delimiter):
            self._tokens.append(Token(
                TokenKind.R_HEREDOC_STRIP if strip_tabs else TokenKind.R_HEREDOC,
                op_text, start, fd=0,
                quoted_delim=quoted, strip_tabs=strip_tabs,
                body=None,
            ))
            self._pos = delim_start  # rewind so sentinel is re-lexed as WORD
            return

        # Skip to end of current line
        while self._pos < self._n and self._cmd[self._pos] != '\n':
            self._pos += 1
        if self._pos < self._n:
            self._pos += 1  # consume newline

        # Collect body lines until delimiter
        body_lines: list[str] = []
        found = False
        while self._pos < self._n:
            line_start = self._pos
            while self._pos < self._n and self._cmd[self._pos] != '\n':
                self._pos += 1
            line = self._cmd[line_start:self._pos]
            if self._pos < self._n:
                self._pos += 1  # consume newline

            if strip_tabs:
                # Strip leading TABs from line for comparison
                tab_count = 0
                for ch in line:
                    if ch == '\t':
                        tab_count += 1
                    else:
                        break
                stripped = line[tab_count:]
            else:
                stripped = line

            if stripped == delimiter:
                found = True
                break

            if strip_tabs:
                body_lines.append(line.lstrip('\t'))
            else:
                body_lines.append(line)

        if not found:
            raise ValueError(f"heredoc delimiter {delimiter!r} not found")

        body = "\n".join(body_lines) + "\n"
        body = body[:MAX_HEREDOC_BODY]

        self._tokens.append(Token(
            TokenKind.R_HEREDOC_STRIP if strip_tabs else TokenKind.R_HEREDOC,
            op_text, start, fd=0,
            quoted_delim=quoted, strip_tabs=strip_tabs,
            body=body,
        ))


# ---------------------------------------------------------------------------
# _detect_sentinels_in_text — split plain text into WordParts with sentinel
#   detection (needed for replay-mode AST construction from cleaned segments)
# ---------------------------------------------------------------------------


def _detect_sentinels_in_text(text: str) -> list[WordPart]:
    """Split *text* into WordParts, marking ``\\x01A<N>\\x01`` and
    ``\\x01H<N>\\x01`` patterns as ``is_sentinel=True`` so that the
    downstream :func:`_extract_from_node` can resolve them from the
    :class:`Expansion` side table."""
    parts: list[WordPart] = []
    i = 0
    n = len(text)
    while i < n:
        m_arg = _SENTINEL_ARG.match(text, i)
        m_hd = _SENTINEL_HD.match(text, i)
        if m_arg:
            sentinel = m_arg.group(0)
            parts.append(WordPart(text=sentinel, raw=sentinel, is_sentinel=True))
            i = m_arg.end()
        elif m_hd:
            sentinel = m_hd.group(0)
            parts.append(WordPart(text=sentinel, raw=sentinel, is_sentinel=True))
            i = m_hd.end()
        else:
            j = i
            while j < n and text[j] != '\x01':
                j += 1
            if j > i:
                chunk = text[i:j]
                parts.append(WordPart(text=chunk, raw=chunk))
            elif j < n:
                # At \x01 but not a valid sentinel — treat as literal byte
                parts.append(WordPart(text='\x01', raw='\x01'))
                j += 1
            i = j
    return parts


# ---------------------------------------------------------------------------
# AST builder — builds AST from tokens + pre-populated expansion
# ---------------------------------------------------------------------------

def _build_ast(
    tokens: list[Token],
    expansion: Expansion,
    *,
    capture_fn=None,
    env=None,
    depth: int = 0,
    subst_count=None,
    deadline=None,
) -> ProgramNode:
    """Build an AST from the token stream.

    Two modes:

    - **replay mode** (*capture_fn* is None): assigns sentinel IDs in
      left-to-right token order, expecting *expansion* to already be
      pre-populated by the char-by-char scanner.  (Today's behaviour.)

    - **populate mode** (*capture_fn* is not None): performs ``$()``
      capture, ``$VAR``/``${VAR}`` expansion, and heredoc/here-string
      body resolution itself and stores the results directly into
      *expansion*.
    """
    pos = 0
    n = len(tokens)
    next_arg_id = 0
    next_hd_id = 0

    if subst_count is None:
        subst_count = [0]

    def _emit_arg_sentinel(raw_src: str, name: str, *, is_subst: bool) -> str:
        """Assign the next arg-sentinel ID and, in populate mode, resolve the value.

        *raw_src* is the source text span (for SUBST the inner text, for
        VARREF the ``$VAR``/``${VAR}`` text).  *name* is the lookup key:
        the inner text for ``$(…)`` or the bare variable name for ``$VAR``.
        """
        nonlocal next_arg_id
        sentinel = f"\x01A{next_arg_id}\x01"
        next_arg_id += 1

        if capture_fn is not None:                      # populate mode
            if is_subst:
                if depth + 1 > MAX_SUBST_DEPTH:
                    raise ValueError(
                        f"Command substitution depth limit ({MAX_SUBST_DEPTH}) exceeded"
                    )
                subst_count[0] += 1
                if subst_count[0] > MAX_SUBST_COUNT:
                    raise ValueError(
                        f"Command substitution count limit ({MAX_SUBST_COUNT}) exceeded"
                    )
                _rc, stdout_bytes = capture_fn(name)
                value: str = stdout_bytes.decode("utf-8", errors="replace").rstrip("\n")
                value = value[:MAX_SUBST_OUTPUT]
            else:
                value = env.get(name, "") if env else ""
            expansion._set_arg_for(sentinel, value)

        return sentinel

    def _peek() -> Optional[Token]:
        if pos < n:
            return tokens[pos]
        return None

    def _consume() -> Optional[Token]:
        nonlocal pos
        t = _peek()
        if t is not None:
            pos += 1
        return t

    def _skip_ws() -> None:
        nonlocal pos
        while pos < n:
            k = tokens[pos].kind
            if k in (TokenKind.WS, TokenKind.NEWLINE):
                pos += 1
            else:
                break

    def _split_word_parts(raw: str) -> list[WordPart]:
        """Split a raw word token value into WordParts with quote stripping.

        ``$(...)``, ``$VAR``, and ``${VAR}`` inside double-quoted spans are
        detected and emitted as sentinel ``WordPart`` entries via
        :func:`_emit_arg_sentinel`.  Single-quoted spans stay fully literal.
        """
        parts: list[WordPart] = []
        i, n2 = 0, len(raw)

        # Quick path: no quotes and no $ (no substitution sentinels needed).
        # But sentinel patterns (\x01A<N>\x01, \x01H<N>\x01) may appear in
        # cleaned segments that have already been through expansion.
        # Only detect them in replay mode (capture_fn is None).
        if '"' not in raw and "'" not in raw and '$' not in raw:
            if '\x01' in raw and capture_fn is None:
                parts.extend(_detect_sentinels_in_text(raw))
            else:
                parts.append(WordPart(text=raw, raw=raw))
            return parts

        current_text: list[str] = []
        current_raw: list[str] = []
        in_quotes = False   # True while accumulating text inside '...' or "..."

        def flush() -> None:
            if current_text or current_raw:
                t = "".join(current_text)
                r = "".join(current_raw)
                parts.append(WordPart(text=t, raw=r if r else t, is_quoted=in_quotes))
                current_text.clear()
                current_raw.clear()

        while i < n2:
            c = raw[i]
            if c == "'":
                # Single quote — fully literal, no $() expansion
                flush()
                current_raw.append(c)
                in_quotes = True
                i += 1
                while i < n2 and raw[i] != "'":
                    current_text.append(raw[i])
                    current_raw.append(raw[i])
                    i += 1
                if i < n2:
                    current_raw.append("'")
                    i += 1
                flush()
                in_quotes = False
                continue
            if c == '"':
                flush()
                current_raw.append(c)
                in_quotes = True
                i += 1
                while i < n2 and raw[i] != '"':
                    ch = raw[i]
                    # Handle backslash escapes inside double quotes
                    if ch == '\\' and i + 1 < n2:
                        nxt = raw[i + 1]
                        if nxt in ('"', '$', '\\'):
                            current_text.append(nxt)
                            current_raw.append('\\')
                            current_raw.append(nxt)
                            i += 2
                        elif nxt == '\n':
                            i += 2
                        else:
                            current_text.append('\\')
                            current_text.append(nxt)
                            current_raw.append('\\')
                            current_raw.append(nxt)
                            i += 2
                    # Detect $( ... ) inside double quotes
                    elif ch == '$' and i + 1 < n2 and raw[i + 1] == '(':
                        # Check for $(( arithmetic (reject)
                        if i + 2 < n2 and raw[i + 2] == '(':
                            raise ParseError(
                                "Arithmetic expansion $((...)) is not supported"
                            )
                        flush()
                        # Find matching ')' with paren + quote tracking
                        j = i + 2
                        paren_depth = 1
                        inner_q: Optional[str] = None
                        while j < n2 and paren_depth > 0:
                            c2 = raw[j]
                            if inner_q is not None:
                                if c2 == inner_q:
                                    inner_q = None
                            elif c2 in ("'", '"'):
                                inner_q = c2
                            elif c2 == '(':
                                paren_depth += 1
                            elif c2 == ')':
                                paren_depth -= 1
                            j += 1
                        if paren_depth != 0:
                            raise ValueError("Unbalanced $( ... )")
                        inner_text = raw[i + 2 : j - 1]
                        raw_subst = raw[i:j]  # "$(inner)"
                        sentinel = _emit_arg_sentinel(inner_text, inner_text, is_subst=True)
                        wp = WordPart(
                            text=sentinel, raw=raw_subst,
                            is_sentinel=True, is_quoted=True,
                        )
                        parts.append(wp)
                        i = j
                    # Detect $VAR / ${VAR} inside double quotes
                    elif ch == '$' and raw[i + 1] != '(':
                        nxt2 = raw[i + 1]
                        if nxt2 == '{' and _BRACED_VAR_RE.match(raw[i:]):
                            flush()
                            m = _BRACED_VAR_RE.match(raw[i:])
                            assert m is not None
                            raw_subst = m.group(0)
                            var_name = raw_subst[2:-1]  # strip ${ and }
                            sentinel = _emit_arg_sentinel(raw_subst, var_name, is_subst=False)
                            wp = WordPart(
                                text=sentinel, raw=raw_subst,
                                is_sentinel=True, is_quoted=True,
                            )
                            parts.append(wp)
                            i += m.end()
                        elif nxt2 and (nxt2.isalpha() or nxt2 == '_'):
                            flush()
                            m = _VAR_NAME_RE.match(raw, i + 1)
                            assert m is not None
                            var_name = m.group(0)
                            raw_subst = raw[i:i + 1 + len(var_name)]
                            sentinel = _emit_arg_sentinel(raw_subst, var_name, is_subst=False)
                            wp = WordPart(
                                text=sentinel, raw=raw_subst,
                                is_sentinel=True, is_quoted=True,
                            )
                            parts.append(wp)
                            i = i + 1 + len(var_name)
                        else:
                            current_text.append(ch)
                            current_raw.append(ch)
                            i += 1
                    else:
                        # Check for sentinel patterns inside double quotes
                        # ONLY in replay mode (capture_fn is None).
                        if ch == '\x01' and capture_fn is None:
                            m_arg = _SENTINEL_ARG.match(raw, i)
                            m_hd = _SENTINEL_HD.match(raw, i)
                            if m_arg:
                                flush()
                                raw_match = m_arg.group(0)
                                wp = WordPart(
                                    text=raw_match, raw=raw_match,
                                    is_sentinel=True, is_quoted=True,
                                )
                                parts.append(wp)
                                i = m_arg.end()
                                continue
                            elif m_hd:
                                flush()
                                raw_match = m_hd.group(0)
                                wp = WordPart(
                                    text=raw_match, raw=raw_match,
                                    is_sentinel=True, is_quoted=True,
                                )
                                parts.append(wp)
                                i = m_hd.end()
                                continue
                        current_text.append(ch)
                        current_raw.append(ch)
                        i += 1
                if i < n2:
                    current_raw.append('"')
                    i += 1
                flush()
                in_quotes = False
                continue
            # Backslash outside quotes: escape next character.
            # In _lex_word, \$ keeps the backslash so _split_word_parts
            # can detect the escape.  Emit a literal $ (no expansion).
            if c == '\\' and i + 1 < n2 and raw[i + 1] == '$':
                current_text.append('$')
                current_raw.append('\\')
                current_raw.append('$')
                i += 2
                continue

            # Check for sentinel patterns in unquoted text
            # ONLY in replay mode (capture_fn is None).
            if c == '\x01' and capture_fn is None:
                m_arg = _SENTINEL_ARG.match(raw, i)
                m_hd = _SENTINEL_HD.match(raw, i)
                if m_arg or m_hd:
                    flush()
                    parts.extend(_detect_sentinels_in_text(raw[i:]))
                    i = n2  # consumed everything
                    continue

            current_text.append(c)
            current_raw.append(c)
            i += 1

        flush()
        return parts

    def _parse_and_or() -> Optional[AndOrNode]:
        nonlocal pos
        _skip_ws()
        t = _peek()
        if t is None:
            return None

        # Collect consecutive chain operators (last wins, matching
        # split_legacy's behaviour:  ;;  ,  ;&&  ,  ||;  etc.).
        operator: Optional[str] = None
        while t is not None and t.kind in (TokenKind.SEMI,
                                             TokenKind.AND_AND,
                                             TokenKind.OR_OR):
            if t.kind == TokenKind.SEMI:
                operator = ";"
            elif t.kind == TokenKind.AND_AND:
                operator = "&&"
            else:
                operator = "||"
            _consume()
            _skip_ws()
            t = _peek()

        pipeline = _parse_pipeline()
        if pipeline is None:
            # Trailing operator(s) with nothing after — drop
            # (matching split_legacy empty-drop semantics).
            return None

        # Check for backgrounding
        bg = False
        _skip_ws()
        t = _peek()
        if t is not None and t.kind == TokenKind.BG:
            _consume()
            bg = True

        return AndOrNode(operator=operator, pipeline=pipeline, backgrounded=bg)

    def _parse_pipeline() -> Optional[PipelineNode]:
        nonlocal pos
        commands: list[CommandNode] = []
        while True:
            _skip_ws()
            cmd = _parse_command()
            if cmd is not None:
                commands.append(cmd)

            _skip_ws()
            t = _peek()
            if t is not None and t.kind == TokenKind.PIPE:
                _consume()
                continue
            break

        if not commands:
            return None
        return PipelineNode(commands=tuple(commands))

    def _parse_command() -> Optional[CommandNode]:
        nonlocal pos
        words: list[Word] = []
        redirects: list[RedirectSpec] = []
        current_parts: list[WordPart] = []

        def _flush_word() -> None:
            if current_parts:
                words.append(Word(parts=tuple(current_parts)))
                current_parts.clear()

        while True:
            # Track whether we consumed whitespace (word boundary)
            ws_seen = False
            while pos < n:
                k = tokens[pos].kind
                if k in (TokenKind.WS, TokenKind.NEWLINE):
                    ws_seen = True
                    pos += 1
                else:
                    break

            t = _peek()
            if t is None:
                break

            kind = t.kind

            # End of command at operator boundaries
            if kind in (TokenKind.PIPE, TokenKind.SEMI, TokenKind.AND_AND,
                        TokenKind.OR_OR, TokenKind.BG):
                break

            # SUBST — if preceded by WS, start new word; else continue current
            if kind == TokenKind.SUBST:
                if ws_seen and current_parts:
                    _flush_word()
                tok = _consume()
                sentinel = _emit_arg_sentinel(tok.value, tok.value, is_subst=True)
                wp = WordPart(text=sentinel, raw=sentinel, is_sentinel=True)
                current_parts.append(wp)
                continue

            # VARREF — same arg-sentinel mechanism as SUBST
            if kind == TokenKind.VARREF:
                if ws_seen and current_parts:
                    _flush_word()
                tok = _consume()
                sentinel = _emit_arg_sentinel("$" + tok.value, tok.value, is_subst=False)
                wp = WordPart(text=sentinel, raw=sentinel, is_sentinel=True)
                current_parts.append(wp)
                continue

            # Redirect operators — always flush current word first
            if kind in (TokenKind.R_OUT, TokenKind.R_APPEND, TokenKind.R_IN,
                        TokenKind.R_FD_DUP, TokenKind.R_HEREDOC,
                        TokenKind.R_HEREDOC_STRIP, TokenKind.R_HERESTRING):
                _flush_word()
                self_tok = _consume()
                assert self_tok is not None
                rspec = _build_redirect_spec(self_tok)
                if rspec is not None:
                    redirects.append(rspec)
                continue

            # WORD — if preceded by WS, start new word; else continue current
            if kind == TokenKind.WORD:
                if ws_seen and current_parts:
                    _flush_word()
                tok = _consume()
                parts = _split_word_parts(tok.value)
                current_parts.extend(parts)
                continue

            break

        _flush_word()

        if not words and not redirects:
            return None

        return CommandNode(
            words=tuple(words),
            redirects=tuple(redirects),
            backgrounded=False,
        )

    def _build_redirect_spec(tok: Token) -> Optional[RedirectSpec]:
        nonlocal pos, next_hd_id

        kind = tok.kind

        # fd-dup: 2>&1, 1>&2
        if kind == TokenKind.R_FD_DUP:
            target_word = Word(parts=(
                WordPart(text=str(tok.target_fd), raw=str(tok.target_fd)),
            ))
            return RedirectSpec(
                fd=tok.fd, op=">&", target=target_word,
                raw_operator=tok.value,
            )

        # heredoc / here-string
        if kind in (TokenKind.R_HEREDOC, TokenKind.R_HEREDOC_STRIP,
                     TokenKind.R_HERESTRING):
            # ---------- sentinel-ID reuse (replay mode) ----------
            # When processing a cleaned segment that already contains
            # sentinel keys (e.g. \x01H0\x01), reuse the original ID
            # so the Expansion lookup in _extract_from_node matches.
            sentinel: Optional[str] = None
            if kind in (TokenKind.R_HEREDOC, TokenKind.R_HEREDOC_STRIP):
                if tok.body is None:
                    # Sentinel shortcut from _lex_heredoc: the next WORD
                    # token carries the original sentinel text.
                    _skip_ws()
                    tgt = _peek()
                    if tgt is not None and tgt.kind == TokenKind.WORD:
                        tgt_text = tgt.value
                        if _SENTINEL_HD.fullmatch(tgt_text):
                            _consume()
                            sentinel = tgt_text
            else:  # R_HERESTRING
                # _lex_herestring stores the target word in tok.body.
                raw_target = tok.body or ""
                target = _strip_quotes(raw_target)
                if target and _SENTINEL_HD.fullmatch(target):
                    sentinel = target
                elif not target:
                    # Empty target (e.g. "cmd <<<" with nothing after) —
                    # create an empty-target RedirectSpec so validation
                    # produces "Here-string missing target".
                    return RedirectSpec(
                        fd=tok.fd, op="<<<",
                        target=Word(parts=()),
                        raw_operator=tok.value,
                    )

            if sentinel is not None:
                # Reuse — advance next_hd_id past this ID to avoid collisions
                m = _SENTINEL_HD.match(sentinel)
                assert m is not None
                hd_id = int(m.group(1))
                if hd_id >= next_hd_id:
                    next_hd_id = hd_id + 1
            else:
                sentinel = f"\x01H{next_hd_id}\x01"
                next_hd_id += 1

            wp = WordPart(text=sentinel, raw=sentinel, is_sentinel=True)
            target_word = Word(parts=(wp,))

            # ---------- populate mode: compute body ----------
            if capture_fn is not None:
                if kind in (TokenKind.R_HEREDOC, TokenKind.R_HEREDOC_STRIP):
                    body: Optional[str] = tok.body
                    if body and not tok.quoted_delim:
                        body = _expand_subst_in_text(body, capture_fn, env=env)
                    body = body[:MAX_HEREDOC_BODY] if body else ""
                    expansion._set_heredoc_for(sentinel, body)
                else:  # R_HERESTRING
                    body_word = _strip_quotes(tok.body) if tok.body else ""
                    if body_word and not tok.quoted_delim:
                        body_word = _expand_subst_in_text(body_word, capture_fn, env=env)
                    body = (body_word + "\n")[:MAX_HEREDOC_BODY]
                    expansion._set_heredoc_for(sentinel, body)

            op_map = {
                TokenKind.R_HEREDOC: "<<",
                TokenKind.R_HEREDOC_STRIP: "<<-",
                TokenKind.R_HERESTRING: "<<<",
            }
            op_str: Literal["<<", "<<-", "<<<"] = op_map[kind]  # type: ignore[assignment]
            return RedirectSpec(
                fd=tok.fd, op=op_str,
                target=target_word,
                strip_tabs=tok.strip_tabs,
                raw_operator=tok.value,
            )

        # file redirects: >, >>, <
        op_map = {
            TokenKind.R_OUT: ">",
            TokenKind.R_APPEND: ">>",
            TokenKind.R_IN: "<",
        }
        op_str: Literal[">", ">>", "<"] = op_map[kind]  # type: ignore[assignment]

        # Read the target word — detect glued target (no whitespace between
        # operator and target, e.g. "2>err").
        saved_pos = pos
        _skip_ws()
        glued = (pos == saved_pos)  # no WS tokens consumed → glued
        target_tok = _peek()
        if target_tok is None or target_tok.kind != TokenKind.WORD:
            return RedirectSpec(
                fd=tok.fd, op=op_str,
                target=Word(parts=()),
                raw_operator=tok.value,
            )

        _consume()
        target_parts = _split_word_parts(target_tok.value)
        target_word = Word(parts=tuple(target_parts))

        # Resolve $(...) sentinels in the target word
        resolved_parts: list[WordPart] = []
        for p in target_parts:
            # Check if this part contains an arg sentinel that we already assigned
            resolved_parts.append(p)
        target_word = Word(parts=tuple(resolved_parts))

        return RedirectSpec(
            fd=tok.fd, op=op_str,
            target=target_word,
            raw_operator=tok.value,
            glued_target=glued,
        )

    # Parse the program
    chains: list[AndOrNode] = []
    while True:
        chain = _parse_and_or()
        if chain is None:
            break
        chains.append(chain)

    return ProgramNode(chains=tuple(chains))


# ---------------------------------------------------------------------------
# serialize_program — reconstruct the cleaned command string from AST
# ---------------------------------------------------------------------------

def serialize_program(program: ProgramNode) -> str:
    """Reconstruct the cleaned command string from the AST.

    Produces a semantically equivalent string with sentinels in place of
    ``$(...)`` and heredoc/here-string bodies.  Words and redirects are
    separated by single spaces (the canonical form).
    """
    parts: list[str] = []

    for i, chain in enumerate(program.chains):
        if i > 0 and chain.operator is not None:
            parts.append(chain.operator)
            parts.append(" ")

        parts.append(_serialize_pipeline(chain.pipeline, sentinel=True))

        if chain.backgrounded:
            parts.append(" &")

    return "".join(parts)


def _serialize_pipeline(pipeline: PipelineNode, sentinel: bool = False) -> str:
    """Serialize a pipeline, joining its commands with `` | ``.

    Each command is serialized via :func:`_serialize_command`.  When
    *sentinel* is True (used by :func:`serialize_program`), commands use the
    sentinel form; otherwise the human-readable display form is used.
    Empty command strings are dropped (matching the empty-drop semantics).
    """
    cmd_strs = [_serialize_command(cmd, sentinel=sentinel) for cmd in pipeline.commands]
    return " | ".join(s for s in cmd_strs if s)


def _serialize_command(cmd: CommandNode, sentinel: bool = False) -> str:
    """Return a display string for *cmd*.

    When *sentinel* is True, words and redirect targets use the sentinel form
    (for :func:`serialize_program`); otherwise the human-readable display form
    is used (for :func:`cmd_to_display` and :func:`split_legacy`).
    """
    if sentinel:
        word_serialize = lambda w: w.serialized()  # noqa: E731
        target_serialize = lambda t: t.serialized()  # noqa: E731
    else:
        word_serialize = lambda w: w.display_serialized()  # noqa: E731
        target_serialize = lambda t: t.display_serialized()  # noqa: E731

    output: list[str] = []

    for w in cmd.words:
        s = word_serialize(w)
        if s:
            output.append(s)

    for rs in cmd.redirects:
        if rs.op == ">&":
            output.append(rs.raw_operator if rs.raw_operator else ">&")
        elif rs.op in ("<<", "<<-", "<<<"):
            op = rs.raw_operator if rs.raw_operator else rs.op
            output.append(op + " " + target_serialize(rs.target))
        else:
            op = rs.raw_operator if rs.raw_operator else rs.op
            sep = "" if rs.glued_target else " "
            output.append(op + sep + target_serialize(rs.target))

    return " ".join(output)


# ---------------------------------------------------------------------------
# split_legacy — reimplementation of legacy split
# ---------------------------------------------------------------------------

def split_legacy(command: str) -> list[tuple[Optional[str], list[str], bool]]:
    """Split a command string into a chain of pipe-connected pipelines.

    AST-projected: lexes *command*, builds an AST in replay mode, and
    projects each :class:`CommandNode` to its display form via
    :func:`_serialize_command`.  Preserves empty-drop semantics.

    Returns a list of ``(operator, pipeline, backgrounded)`` triples.
    """
    tokens = Lexer(command, replay_mode=True).tokenize()
    program = _build_ast(tokens, Expansion())  # replay mode
    chains = program_to_chain(program)
    result: list[tuple[Optional[str], list[str], bool]] = []
    for op, cmd_nodes, bg in chains:
        segs = [_serialize_command(cmd) for cmd in cmd_nodes]
        result.append((op, segs, bg))
    return result


# ---------------------------------------------------------------------------
# program_to_chain — project ProgramNode to legacy chain format
# ---------------------------------------------------------------------------


def program_to_chain(
    program: ProgramNode,
) -> list[tuple[Optional[str], list[CommandNode], bool]]:
    """Project a ProgramNode to the legacy chain format.

    Returns ``[(operator, [CommandNode...], backgrounded), ...]``.
    This is the AST-native equivalent of ``split_legacy`` for use by the
    live execution path so it can walk the AST directly.

    Empty pipelines are dropped (matching ``split_legacy``'s empty-drop
    semantics) so that ``_run_pipeline`` never receives an empty list.
    """
    result: list[tuple[Optional[str], list[CommandNode], bool]] = []
    for chain in program.chains:
        if not chain.pipeline.commands:
            continue  # drop empty pipeline (parity with split_legacy)
        result.append((
            chain.operator,
            list(chain.pipeline.commands),
            chain.backgrounded,
        ))
    return result


# ---------------------------------------------------------------------------
# cmd_to_display — create a human-readable display string from a CommandNode
# ---------------------------------------------------------------------------


def cmd_to_display(cmd: CommandNode) -> str:
    """Return a human-readable display string for *cmd*."""
    return _serialize_command(cmd)


# ---------------------------------------------------------------------------
# _expand_subst_in_text — used in heredoc body $() expansion
# ---------------------------------------------------------------------------

def _expand_subst_in_text(
    text: str,
    capture_fn,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Scan *text* for ``$( ... )`` and replace each with its raw output.

    Used for expanding ``$()`` inside unquoted heredoc bodies and unquoted
    here-string words.  No sentinel tokens — the output is spliced directly
    into the body text.

    Note that *env* is currently reserved and NOT used: ``$VAR`` / ``${VAR}``
    are NOT expanded in heredoc/here-string bodies (only ``$( ... )`` is).
    ``None`` or ``{}`` → every ``$VAR`` resolves to ``""`` (i.e. stays as the
    literal text).  The parameter exists to keep the call sites consistent
    with the surrounding expansion machinery, which does handle ``$VAR``.
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
            _rc, stdout_bytes = capture_fn(inner)
            expanded = stdout_bytes.decode("utf-8", errors="replace").rstrip("\n")
            result.append(expanded)
            i = j
        else:
            result.append(c)
            i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# _strip_quotes helper
# ---------------------------------------------------------------------------

def _strip_quotes(s: str) -> str:
    """Strip one level of single or double quotes from *s*."""
    if len(s) >= 2:
        if (s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"'):
            return s[1:-1]
    return s


# ---------------------------------------------------------------------------
# extract_redirects — AST-native + legacy string path
# ---------------------------------------------------------------------------

def extract_redirects(
    segment,
    expansion: Optional[Expansion] = None,
    work_dir: Optional[Path] = None,
) -> tuple[list[str], list[Redirect], Optional[str]]:
    """Tokenize a command segment, extracting redirect operators.

    Accepts either a string (legacy path — lexes inline) or a CommandNode
    (AST-native path).  Returns ``(args, redirects, None)`` on success, or
    ``([], [], error_msg)`` on error.

    *work_dir*, when given, enables glob/pathname expansion on unquoted
    metacharacters in command args (not redirect targets).

    All error strings match the original ``_extract_redirects`` exactly.
    """
    if isinstance(segment, str):
        return _extract_from_string(segment, expansion, work_dir)
    elif isinstance(segment, CommandNode):
        return _extract_from_node(segment, expansion, work_dir)
    else:
        return [], [], f"Unsupported segment type: {type(segment).__name__}"


def _extract_from_string(
    segment: str,
    expansion: Optional[Expansion] = None,
    work_dir: Optional[Path] = None,
) -> tuple[list[str], list[Redirect], Optional[str]]:
    """AST-projected path: lex *segment* and route through the AST.

    Builds an AST from the segment in replay mode (assigning sentinel IDs
    but resolving from *expansion*), then delegates to
    :func:`_extract_from_node`.  This replaces the hand-rolled char-by-char
    scanner but preserves every validation and error string exactly.
    """
    # Detect unbalanced quotes — the lexer is lenient, so pre-check.
    if _has_unbalanced_quotes(segment):
        return [], [], "Unbalanced quotes in command"

    try:
        tokens = Lexer(segment, replay_mode=True).tokenize()
    except (ParseError, ValueError) as e:
        return [], [], str(e)

    exp = expansion if expansion is not None else Expansion()
    try:
        program = _build_ast(tokens, exp)  # replay mode (no capture_fn)
    except (ParseError, ValueError) as e:
        return [], [], str(e)

    chain = program_to_chain(program)
    if not chain or not chain[0][1]:
        return [], [], None

    return _extract_from_node(chain[0][1][0], expansion, work_dir)


def _has_unbalanced_quotes(text: str) -> bool:
    """Return True if *text* has an unclosed single or double quote."""
    quote: Optional[str] = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote is not None:
            if quote == '"' and c == '\\' and i + 1 < n:
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
        elif c in ("'", '"'):
            quote = c
            i += 1
        else:
            i += 1
    return quote is not None


def _expand_tilde(s: str) -> str:
    """Shell-style tilde expansion, only for words beginning with ``~``."""
    return str(Path(s).expanduser()) if s.startswith("~") else s


def _expand_glob_arg(pattern: str, work_dir: Path) -> list[str]:
    """Expand a single command arg containing glob metacharacters.

    The pattern is resolved against ``work_dir`` (or used absolute), globbed,
    and each match is filtered to stay within allowed roots (work_dir + extra
    redirect roots). Returns the matched path strings (absolute), or [] if no
    matches survive containment. Caller falls back to the literal arg when this
    returns [].
    """
    from .containment import _contained_in_any
    from .config import EXTRA_REDIRECT_ROOTS
    if work_dir is None:
        return []
    p = _expand_tilde(pattern)   # tilde expansion precedes pathname expansion
    pat = os.path.join(str(work_dir), p) if not p.startswith("/") else p
    matches = _glob.glob(pat)
    out = []
    for m in matches:
        cand = _contained_in_any(m, [work_dir, *EXTRA_REDIRECT_ROOTS])
        if cand is not None:
            out.append(str(cand))
    out.sort()   # deterministic order (glob order is filesystem-dependent)
    return out


def _extract_from_node(
    cmd: CommandNode,
    expansion: Optional[Expansion] = None,
    work_dir: Optional[Path] = None,
) -> tuple[list[str], list[Redirect], Optional[str]]:
    """AST-native path: project a CommandNode to (args, redirects, err).

    Mirrors every validation and error string of ``_extract_from_string``
    so the two paths produce identical results.
    """
    args: list[str] = []
    redirects: list[Redirect] = []

    for w in cmd.words:
        resolved = ""
        pattern = ""
        glob_active = False
        for p in w.parts:
            if p.is_arg_sentinel and expansion is not None:
                val = expansion.arg_for(p)
                val = val if val is not None else p.text
                resolved += val
                if p.is_quoted:
                    pattern += _glob.escape(val)
                else:
                    pattern += val
                    if any(c in val for c in "*?["):
                        glob_active = True
            else:
                resolved += p.text
                if p.is_quoted:
                    pattern += _glob.escape(p.text)
                else:
                    pattern += p.text
                    if any(c in p.text for c in "*?["):
                        glob_active = True
        if resolved:
            if work_dir is not None and glob_active:
                matches = _expand_glob_arg(pattern, work_dir)
                if matches:
                    args.extend(matches)          # sorted/unique order from glob
                else:
                    args.append(_expand_tilde(resolved))   # POSIX: unmatched glob stays literal
            else:
                args.append(_expand_tilde(resolved))

    for rs in cmd.redirects:
        target_text = ""
        body: Optional[str] = None
        strip_tabs = rs.strip_tabs

        for p in rs.target.parts:
            if p.is_hd_sentinel and expansion is not None:
                target_text += p.text
                body = expansion.heredoc_for(p)
            elif p.is_arg_sentinel and expansion is not None:
                val = expansion.arg_for(p)
                target_text += val if val is not None else p.text
            else:
                target_text += p.text

        # ---- validation: missing target for file redirects ----
        if rs.op in (">", ">>"):
            target_text = _expand_tilde(target_text)
            if not target_text:
                return [], [], "Redirect operator missing target file"
        if rs.op == "<":
            target_text = _expand_tilde(target_text)
            if not target_text:
                return [], [], "Input redirect missing target file"

        # ---- validation: here-string ----
        if rs.op == "<<<":
            if not target_text:
                return [], [], "Here-string missing target"
            if body is None:
                return [], [], "Here-string body not found"

        # ---- validation: heredoc ----
        if rs.op in ("<<", "<<-"):
            if not target_text:
                return [], [], "Heredoc missing delimiter sentinel"
            if body is None:
                return [], [], "Heredoc body not found"

        # ---- validation: multiple stdin redirects ----
        if rs.op in ("<", "<<", "<<<", "<<-"):
            for r in redirects:
                if r.fd == 0:
                    return [], [], "Multiple stdin redirects in one segment"

        if rs.op in (">", ">>", "<"):
            redirects.append(Redirect(
                fd=rs.fd, op=rs.op,
                target_path=None, target_fd=None,
                raw_target=target_text,
            ))
        elif rs.op == ">&":
            redirects.append(Redirect(
                fd=rs.fd, op=">&",
                target_path=None,
                target_fd=int(target_text) if target_text.isdigit() else None,
                raw_target=target_text,
            ))
        elif rs.op in ("<<", "<<<", "<<-"):
            redirects.append(Redirect(
                fd=rs.fd, op=rs.op,
                body=body,
                strip_tabs=strip_tabs,
            ))

    return args, redirects, None


# ---------------------------------------------------------------------------
# parse_command — thin AST wrapper (scan → lex → build → serialize)
# ---------------------------------------------------------------------------

def parse_command(
    command: str,
    capture_fn,
    work_dir,
    timeout: int,
    depth: int,
    deadline: Optional[float] = None,
    subst_count: Optional[list[int]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[str, Expansion, Optional[ProgramNode]]:
    """Pre-pass: resolve ``$(...)``, heredocs, and here-strings.

    Lexes the command with :class:`Lexer`, builds the AST via
    :func:`_build_ast` in populate mode (which performs ``$()`` capture,
    ``$VAR``/``${VAR}`` expansion, and heredoc/here-string body
    resolution), and derives the cleaned command string from the AST
    via :func:`serialize_program`.

    *env* supplies ``$VAR`` values for expansion.  ``None`` or ``{}`` →
    every ``$VAR`` resolves to ``""``.  Expansion uses this env ONLY —
    NOT the per-command unveil_env (which is computed after parse time).

    Returns ``(cleaned_command, expansion, program_ast)``.
    """
    import time as _time

    if subst_count is None:
        subst_count = [0]
    if deadline is None:
        deadline = _time.time() + timeout

    # Reject unsupported constructs (quote-aware scan)
    _check_unsupported(command)

    expansion = Expansion()
    tokens = Lexer(command).tokenize()
    program = _build_ast(
        tokens, expansion,
        capture_fn=capture_fn,
        env=env,
        depth=depth,
        subst_count=subst_count,
        deadline=deadline,
    )
    cleaned = serialize_program(program)
    return cleaned, expansion, program


# ---------------------------------------------------------------------------
# _check_unsupported — kept for backward compatibility (tests import it)
# ---------------------------------------------------------------------------

def _check_unsupported(command: str) -> None:
    """Scan *command* for unsupported constructs and raise ParseError.

    Quote-aware scan for backticks, ``<(...)``, and ``>(...)``.
    Kept as a standalone function for backward compatibility with tests
    that import and call it directly.
    """
    i, n = 0, len(command)
    quote: Optional[str] = None

    while i < n:
        c = command[i]
        if quote is not None:
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            i += 1
            continue

        # Backtick
        if c == '`':
            raise ParseError("Backtick command substitution is not supported; use $(...)")

        # Process substitution <( ) or >( )
        if c == '<' and i + 1 < n and command[i + 1] == '(':
            raise ParseError("Process substitution <(...) is not supported")
        if c == '>' and i + 1 < n and command[i + 1] == '(':
            raise ParseError("Process substitution >(...) is not supported")

        i += 1
