"""Shell command parser — lexer, AST, and parse functions.

Replaces the hand-rolled char-by-char parsing passes in server.py with a
proper lexer + recursive-descent parser.  Exports the same Redirect,
Expansion, SENTINEL_ARG, and SENTINEL_HD symbols that server.py used to
define directly, so existing imports keep working.

Public API
----------
- ParseError, Redirect, Expansion, SENTINEL_ARG, SENTINEL_HD
- parse_command(text, capture_fn, ...) → (cleaned, expansion, program)
- split_legacy(text) → list[(op|None, [stages], bg)]
- extract_redirects(segment, expansion) → (args, redirects, err)
- serialize_program(program) → str
- program_to_chain(program) → list[(op|None, [CommandNode], bg)]
- Lexer, Token, TokenKind, CommandNode, PipelineNode, AndOrNode, ProgramNode (AST)
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


# ---------------------------------------------------------------------------
# Sentinel patterns (moved from server.py; same values)
# ---------------------------------------------------------------------------

SENTINEL_ARG = re.compile(r"\x01A(\d+)\x01")
SENTINEL_HD  = re.compile(r"\x01H(\d+)\x01")

MAX_SUBST_DEPTH = 8
MAX_SUBST_COUNT = 256
MAX_SUBST_OUTPUT = 64_000
MAX_HEREDOC_BODY = 256_000


# ---------------------------------------------------------------------------
# Expansion (side table)
# ---------------------------------------------------------------------------

@dataclass
class Expansion:
    """Side table holding resolved $() output words and heredoc/here-string bodies."""
    arg_values: dict[str, str] = field(default_factory=dict)
    heredoc_bodies: dict[str, str] = field(default_factory=dict)


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
        """Return the text for use in serialize_program."""
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


@dataclass(frozen=True)
class RedirectSpec:
    """A redirect operator plus its target word."""
    fd: int
    op: Literal[">", ">>", ">&", "<", "<<", "<<<", "<<-"]
    target: Word               # the target word (may be a sentinel)
    strip_tabs: bool = False
    raw_operator: str = ""     # e.g. "2>&1" or "1>&2" or ">" or ">>"


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

    def __init__(self, command: str):
        self._cmd = command
        self._n = len(command)
        self._pos = 0
        self._tokens: list[Token] = []

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
                    chars.append(self._cmd[i + 1])
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
                while i < self._n and self._cmd[i] != '"':
                    ch = self._cmd[i]
                    if ch == '\\' and i + 1 < self._n:
                        nxt = self._cmd[i + 1]
                        if nxt in ('"', '$', '\\'):
                            i += 2  # skip the backslash, keep the escaped char
                        elif nxt == '\n':
                            i += 2  # line continuation
                        else:
                            i += 2  # \X stays literal inside double quotes
                    else:
                        i += 1
                if i < self._n:
                    i += 1  # closing quote
                chars.append(self._cmd[dq_start:i])
                continue

            # word terminators (outside quotes)
            if c in (' ', '\t', '\n', '|', ';', '&'):
                break

            # $( outside quotes → let the main loop handle it (not mid-word)
            if c == '$' and i + 1 < self._n and self._cmd[i + 1] == '(':
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
        while self._pos < self._n:
            c = self._cmd[self._pos]
            if quote is not None:
                chars.append(c)
                if c == quote:
                    quote = None
                self._pos += 1
            elif c in ("'", '"'):
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
# AST builder — builds AST from tokens + pre-populated expansion
# ---------------------------------------------------------------------------

def _build_ast(
    tokens: list[Token],
    expansion: Expansion,
) -> ProgramNode:
    """Build an AST from the token stream using the already-populated expansion.

    The expansion table is pre-populated by the char-by-char scanner in
    ``parse_command``.  This function assigns matching sentinel IDs by
    walking tokens in the same left-to-right order as the scanner, so
    arg sentinel IDs and heredoc sentinel IDs match the expansion keys.
    """
    pos = 0
    n = len(tokens)
    next_arg_id = 0
    next_hd_id = 0

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

    def _split_word_parts(raw: str, next_arg_id: Optional[list[int]] = None) -> list[WordPart]:
        """Split a raw word token value into WordParts with quote stripping.

        When *next_arg_id* is provided (a single-element list), ``$(...)``
        inside double-quoted spans is detected and emitted as a SUBST
        ``WordPart`` with an assigned sentinel ID.  Single-quoted spans
        stay fully literal regardless.
        """
        parts: list[WordPart] = []
        i, n2 = 0, len(raw)

        # Quick path: no quotes
        if '"' not in raw and "'" not in raw:
            parts.append(WordPart(text=raw, raw=raw))
            return parts

        current_text: list[str] = []
        current_raw: list[str] = []

        def flush() -> None:
            if current_text or current_raw:
                t = "".join(current_text)
                r = "".join(current_raw)
                parts.append(WordPart(text=t, raw=r if r else t))
                current_text.clear()
                current_raw.clear()

        while i < n2:
            c = raw[i]
            if c == "'":
                # Single quote — fully literal, no $() expansion
                flush()
                current_raw.append(c)
                i += 1
                while i < n2 and raw[i] != "'":
                    current_text.append(raw[i])
                    current_raw.append(raw[i])
                    i += 1
                if i < n2:
                    current_raw.append("'")
                    i += 1
                continue
            if c == '"':
                flush()
                current_raw.append(c)
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
                    elif (ch == '$' and i + 1 < n2 and raw[i + 1] == '('
                          and next_arg_id is not None):
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
                        aid = next_arg_id[0]
                        next_arg_id[0] += 1
                        sentinel = f"\x01A{aid}\x01"
                        wp = WordPart(
                            text=sentinel, raw=raw_subst,
                            is_sentinel=True, is_quoted=True,
                        )
                        parts.append(wp)
                        i = j
                    else:
                        current_text.append(ch)
                        current_raw.append(ch)
                        i += 1
                if i < n2:
                    current_raw.append('"')
                    i += 1
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
        nonlocal pos, next_arg_id, next_hd_id
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
                _consume()
                sentinel = f"\x01A{next_arg_id}\x01"
                next_arg_id += 1
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
                _consume()
                nid = [next_arg_id]
                parts = _split_word_parts(t.value, nid)
                next_arg_id = nid[0]
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
        nonlocal pos, next_hd_id, next_arg_id

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
            sentinel = f"\x01H{next_hd_id}\x01"
            next_hd_id += 1
            wp = WordPart(text=sentinel, raw=sentinel, is_sentinel=True)
            target_word = Word(parts=(wp,))

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

        # Read the target word
        _skip_ws()
        target_tok = _peek()
        if target_tok is None or target_tok.kind != TokenKind.WORD:
            return RedirectSpec(
                fd=tok.fd, op=op_str,
                target=Word(parts=()),
                raw_operator=tok.value,
            )

        _consume()
        nid = [next_arg_id]
        target_parts = _split_word_parts(target_tok.value, nid)
        next_arg_id = nid[0]

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

        parts.append(_serialize_pipeline(chain.pipeline))

        if chain.backgrounded:
            parts.append(" &")

    return "".join(parts)


def _serialize_pipeline(pipeline: PipelineNode) -> str:
    cmd_strs = [_serialize_command(cmd) for cmd in pipeline.commands]
    return " | ".join(s for s in cmd_strs if s)


def _serialize_command(cmd: CommandNode) -> str:
    output: list[str] = []

    for w in cmd.words:
        s = w.serialized()
        if s:
            output.append(s)

    for rs in cmd.redirects:
        if rs.op == ">&":
            output.append(rs.raw_operator if rs.raw_operator else ">&")
        elif rs.op in ("<<", "<<-", "<<<"):
            op = rs.raw_operator if rs.raw_operator else rs.op
            output.append(op + " " + rs.target.serialized())
        else:
            op = rs.raw_operator if rs.raw_operator else rs.op
            output.append(op + " " + rs.target.serialized())

    return " ".join(output)


# ---------------------------------------------------------------------------
# split_legacy — reimplementation of _split_command
# ---------------------------------------------------------------------------

def split_legacy(command: str) -> list[tuple[Optional[str], list[str], bool]]:
    """Split a command string into a chain of pipe-connected pipelines.

    Re-implements the exact behaviour of the original ``_split_command``
    function.  Lenient: ``;;``, ``|||``, leading ``|``, trailing ``|``,
    lone ``;`` produce empty segments that are dropped.

    Returns a list of ``(operator, pipeline, backgrounded)`` triples.
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
            # Handle backslash escapes inside double quotes
            if quote == '"' and c == '\\' and i + 1 < n:
                nxt = command[i + 1]
                if nxt in ('"', '$', '\\', '\n'):
                    current_seg.append(c)
                    current_seg.append(nxt)
                    i += 2
                    continue
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
            # redirect operator, not a backgrounding operator.
            current_seg.append(c)
            i += 1
            continue
        if c == "&":
            # Bare '&' backgrounding
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
    """Return a display-friendly string for *cmd* using the serialized form."""
    return _serialize_command(cmd)


# ---------------------------------------------------------------------------
# _expand_subst_in_text — used in heredoc body $() expansion
# ---------------------------------------------------------------------------

def _expand_subst_in_text(
    text: str,
    capture_fn,
) -> str:
    """Scan *text* for ``$( ... )`` and replace each with its raw output.

    Used for expanding ``$()`` inside unquoted heredoc bodies and unquoted
    here-string words.  No sentinel tokens — the output is spliced directly
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
) -> tuple[list[str], list[Redirect], Optional[str]]:
    """Tokenize a command segment, extracting redirect operators.

    Accepts either a string (legacy path — lexes inline) or a CommandNode
    (AST-native path).  Returns ``(args, redirects, None)`` on success, or
    ``([], [], error_msg)`` on error.

    All error strings match the original ``_extract_redirects`` exactly.
    """
    if isinstance(segment, str):
        return _extract_from_string(segment, expansion)
    elif isinstance(segment, CommandNode):
        return _extract_from_node(segment, expansion)
    else:
        return [], [], f"Unsupported segment type: {type(segment).__name__}"


def _extract_from_string(
    segment: str,
    expansion: Optional[Expansion] = None,
) -> tuple[list[str], list[Redirect], Optional[str]]:
    """Legacy path: lex a string segment inline.  Reproduces the exact
    behaviour of the original ``_extract_redirects``."""
    args: list[str] = []
    redirects: list[Redirect] = []
    i = 0
    n = len(segment)

    def _resolve_word(w: str) -> str:
        if not expansion:
            return w
        def _replace(m: re.Match) -> str:
            key = f"\x01A{m.group(1)}\x01"
            val = expansion.arg_values.get(key)
            return val if val is not None else m.group(0)
        return SENTINEL_ARG.sub(_replace, w)

    def _resolve_heredoc_body(token: str, kind: str = "heredoc") -> Optional[str]:
        if not expansion:
            return None
        m = SENTINEL_HD.fullmatch(token)
        if not m:
            return None
        key = f"\x01H{m.group(1)}\x01"
        return expansion.heredoc_bodies.get(key)

    def _read_word() -> Optional[str]:
        nonlocal i
        chars: list[str] = []
        quote: Optional[str] = None
        while i < n:
            c = segment[i]
            if quote is not None:
                # Handle backslash escapes inside double quotes
                if quote == '"' and c == '\\' and i + 1 < n:
                    nxt = segment[i + 1]
                    if nxt in ('"', '$', '\\'):
                        # escaped char: skip backslash, keep the escaped char
                        chars.append(nxt)
                        i += 2
                        continue
                    elif nxt == '\n':
                        # line continuation: skip backslash and newline
                        i += 2
                        continue
                if c == quote:
                    quote = None
                    i += 1
                    continue
                chars.append(c)
                i += 1
            # Backslash outside quotes: escape next character
            elif c == '\\' and i + 1 < n:
                chars.append(segment[i + 1])
                i += 2
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

        # -- 4-char fd-dup: 2>&1 --
        if rem.startswith('2>&1') and not (len(rem) > 4 and (rem[4].isalnum() or rem[4] == '_')):
            i += 4
            redirects.append(Redirect(fd=2, op='>&', target_path=None, target_fd=1, raw_target='1'))
            continue
        if rem.startswith('1>&2') and not (len(rem) > 4 and (rem[4].isalnum() or rem[4] == '_')):
            i += 4
            redirects.append(Redirect(fd=1, op='>&', target_path=None, target_fd=2, raw_target='2'))
            continue

        # -- 4-char fd-dup for disallowed target fds --
        if (len(rem) >= 4 and rem[0].isdigit() and rem[1:3] == '>&'
                and rem[3].isdigit()
                and not (len(rem) > 4 and (rem[4].isalnum() or rem[4] == '_'))):
            return [], [], "Redirect dup target fd must be 1 or 2"

        # -- 3-char: 2>>, 1>> --
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

        # -- here-string: <<< --
        if rem.startswith('<<<'):
            i += 3
            t = _read_redirect_target()
            if t is None:
                return [], [], "Here-string missing target"
            body = _resolve_heredoc_body(t, "here-string")
            if body is None:
                return [], [], "Here-string body not found"
            for r in redirects:
                if r.fd == 0:
                    return [], [], "Multiple stdin redirects in one segment"
            redirects.append(Redirect(fd=0, op='<<<', body=body))
            continue

        # -- heredoc with tab strip: <<- --
        if rem.startswith('<<-'):
            i += 3
            t = _read_redirect_target()
            if t is None:
                return [], [], "Heredoc missing delimiter sentinel"
            body = _resolve_heredoc_body(t, "heredoc")
            if body is None:
                return [], [], "Heredoc body not found"
            for r in redirects:
                if r.fd == 0:
                    return [], [], "Multiple stdin redirects in one segment"
            redirects.append(Redirect(fd=0, op='<<-', body=body, strip_tabs=True))
            continue

        # -- 2-char: >>, 2>, 1> --
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

        # -- heredoc: << --
        if rem.startswith('<<'):
            i += 2
            t = _read_redirect_target()
            if t is None:
                return [], [], "Heredoc missing delimiter sentinel"
            body = _resolve_heredoc_body(t, "heredoc")
            if body is None:
                return [], [], "Heredoc body not found"
            for r in redirects:
                if r.fd == 0:
                    return [], [], "Multiple stdin redirects in one segment"
            redirects.append(Redirect(fd=0, op='<<', body=body))
            continue

        # -- 1-char: >, < --
        if rem.startswith('>'):
            i += 1
            t = _read_redirect_target()
            if t is None:
                return [], [], "Redirect operator missing target file"
            redirects.append(Redirect(fd=1, op='>', target_path=None, target_fd=None, raw_target=t))
            continue
        if rem.startswith('<'):
            i += 1
            t = _read_redirect_target()
            if t is None:
                return [], [], "Input redirect missing target file"
            for r in redirects:
                if r.fd == 0:
                    return [], [], "Multiple stdin redirects in one segment"
            redirects.append(Redirect(fd=0, op='<', target_path=None, target_fd=None, raw_target=t))
            continue

        # -- digit + > that aren't 1> / 2> --
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


def _extract_from_node(
    cmd: CommandNode,
    expansion: Optional[Expansion] = None,
) -> tuple[list[str], list[Redirect], Optional[str]]:
    """AST-native path: project a CommandNode to (args, redirects, err).

    Mirrors every validation and error string of ``_extract_from_string``
    so the two paths produce identical results.
    """
    args: list[str] = []
    redirects: list[Redirect] = []

    for w in cmd.words:
        resolved = ""
        for p in w.parts:
            if p.is_arg_sentinel and expansion is not None:
                val = expansion.arg_values.get(p.text)
                resolved += val if val is not None else p.text
            else:
                resolved += p.text
        if resolved:
            args.append(resolved)

    for rs in cmd.redirects:
        target_text = ""
        body: Optional[str] = None
        strip_tabs = rs.strip_tabs

        for p in rs.target.parts:
            if p.is_hd_sentinel and expansion is not None:
                target_text += p.text
                body = expansion.heredoc_bodies.get(p.text)
            elif p.is_arg_sentinel and expansion is not None:
                val = expansion.arg_values.get(p.text)
                target_text += val if val is not None else p.text
            else:
                target_text += p.text

        # ---- validation: missing target for file redirects ----
        if rs.op in (">", ">>"):
            if not target_text:
                return [], [], "Redirect operator missing target file"
        if rs.op == "<":
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
# parse_command — full expansion + AST parse
# ---------------------------------------------------------------------------

def parse_command(
    command: str,
    capture_fn,
    work_dir,
    timeout: int,
    depth: int,
    deadline: Optional[float] = None,
    subst_count: Optional[list[int]] = None,
) -> tuple[str, Expansion, Optional[ProgramNode]]:
    """Pre-pass: resolve ``$(...)``, heredocs, and here-strings.

    Uses the proven char-by-char scanner to produce the cleaned command
    string and populate the expansion table.  Also tokenizes with the new
    :class:`Lexer` and builds the full AST via :func:`_build_ast`, so
    consumers get a real ``ProgramNode``.

    Returns ``(cleaned_command, expansion, program_ast)``.
    """
    import time as _time

    if subst_count is None:
        subst_count = [0]
    if deadline is None:
        deadline = _time.time() + timeout

    expansion = Expansion(arg_values={}, heredoc_bodies={})
    output: list[str] = []
    i, n = 0, len(command)
    quote: Optional[str] = None
    next_arg_id = 0
    next_hd_id = 0

    # ---- char-by-char scanner (proven, tested) ----
    # This populates `expansion` and produces `output` (→ cleaned string).
    # It rejects unsupported constructs and handles escapes for quotes.
    # This is the EXACT code from the old parse_command, preserved verbatim.

    # Reject unsupported constructs during the scan
    _check_unsupported(command)

    while i < n:
        c = command[i]

        # ---- inside a quote ----
        if quote is not None:
            # Double-quote: handle backslash escapes
            if quote == '"' and c == '\\' and i + 1 < n:
                nxt = command[i + 1]
                if nxt in ('"', '$', '\\', '\n'):
                    output.append(c)
                    output.append(nxt)
                    i += 2
                    continue

            # Double-quote: $( ... ) command substitution
            if quote == '"' and c == '$' and i + 1 < n and command[i + 1] == '(':
                # Check for $(( arithmetic
                if i + 2 < n and command[i + 2] == '(':
                    raise ParseError("Arithmetic expansion $((...)) is not supported")
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

                # Check depth/count limits before recursing
                if depth + 1 > MAX_SUBST_DEPTH:
                    raise ValueError(
                        f"Command substitution depth limit ({MAX_SUBST_DEPTH}) exceeded"
                    )
                subst_count[0] += 1
                if subst_count[0] > MAX_SUBST_COUNT:
                    raise ValueError(
                        f"Command substitution count limit ({MAX_SUBST_COUNT}) exceeded"
                    )

                # Recursively capture inner command output
                rc, stdout_bytes = capture_fn(inner)
                result = stdout_bytes.decode("utf-8", errors="replace").rstrip("\n")
                result = result[:MAX_SUBST_OUTPUT]
                sentinel = f"\x01A{next_arg_id}\x01"
                next_arg_id += 1
                expansion.arg_values[sentinel] = result
                output.append(sentinel)
                i = j
                continue

            # Default: copy char verbatim (both single and double quotes)
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

        # ---- backslash outside quotes: escape next character ----
        if c == '\\' and i + 1 < n:
            # Copy both the backslash and the escaped character
            output.append(c)
            output.append(command[i + 1])
            i += 2
            continue

        # ---- $( ... ) command substitution ----
        if c == '$' and i + 1 < n and command[i + 1] == '(':
            # Check for $(( arithmetic
            if i + 2 < n and command[i + 2] == '(':
                raise ParseError("Arithmetic expansion $((...)) is not supported")
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

            # Check depth/count limits before recursing
            if depth + 1 > MAX_SUBST_DEPTH:
                raise ValueError(
                    f"Command substitution depth limit ({MAX_SUBST_DEPTH}) exceeded"
                )
            subst_count[0] += 1
            if subst_count[0] > MAX_SUBST_COUNT:
                raise ValueError(
                    f"Command substitution count limit ({MAX_SUBST_COUNT}) exceeded"
                )

            # Recursively capture inner command output
            rc, stdout_bytes = capture_fn(inner)
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
                body_word = _expand_subst_in_text(body_word, capture_fn)

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
            # Handle <<-\EOF (backslash-escaped delimiter → literal body)
            if i < n and command[i] == '\\':
                delim_quoted = True
                i += 1
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
                body = _expand_subst_in_text(body, capture_fn)

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
            # Handle <<\EOF (backslash-escaped delimiter → literal body)
            if i < n and command[i] == '\\':
                delim_quoted = True
                i += 1
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
                body = _expand_subst_in_text(body, capture_fn)

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

    cleaned = "".join(output)

    # ---- build AST from the ORIGINAL command (not the cleaned string) ----
    # The token stream preserves the original structure.  We assign sentinel
    # IDs in the same order as the scanner above, so they match the expansion
    # keys.
    try:
        lexer = Lexer(command)
        tokens = lexer.tokenize()
        program: Optional[ProgramNode] = _build_ast(tokens, expansion)
    except (ParseError, ValueError):
        # If lexing fails for some reason, return program=None rather than
        # crashing — the cleaned string + expansion are still valid.
        program = None

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
