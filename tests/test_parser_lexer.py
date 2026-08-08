"""Unit tests for the shell parser lexer — token kinds, positions,
quote state, redirect-at-boundary, heredoc body collection, $(...) spans."""

import unittest

from shell_sandbox_mcp.parser import (
    Expansion,
    ParseError,
    Redirect,
    TokenKind,
    Token,
    _check_unsupported,
    _expand_subst_in_text,
    _strip_quotes,
    extract_redirects,
    parse_command,
    split_legacy,
)


# ---------------------------------------------------------------------------
# AST navigation helpers — used to locate sentinel WordParts without
# rebuilding sentinel keys.
# ---------------------------------------------------------------------------

def _find_hd_sentinel(prog):
    """Return the first heredoc-sentinel WordPart in the first command."""
    cmd = prog.chains[0].pipeline.commands[0]
    for rs in cmd.redirects:
        for p in rs.target.parts:
            if p.is_hd_sentinel:
                return p
    return None


def _find_arg_sentinel(prog):
    """Return the first arg-sentinel WordPart in the first command."""
    cmd = prog.chains[0].pipeline.commands[0]
    for w in cmd.words:
        for p in w.parts:
            if p.is_arg_sentinel:
                return p
    return None


class LexerUnsupportedRejectionTest(unittest.TestCase):
    """Test that _check_unsupported rejects forbidden constructs."""

    def test_backtick_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            _check_unsupported("echo `ls`")
        self.assertIn("Backtick", str(ctx.exception))

    def test_process_substitution_lt_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            _check_unsupported("diff <(echo a) <(echo b)")
        self.assertIn("Process substitution <(...)", str(ctx.exception))

    def test_process_substitution_gt_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            _check_unsupported("echo >(cat)")
        self.assertIn("Process substitution >(...)", str(ctx.exception))

    def test_backtick_inside_single_quotes_ignored(self) -> None:
        # Backticks inside quotes are literal — no error.
        _check_unsupported("echo '`ls`'")  # should not raise

    def test_backtick_inside_double_quotes_ignored(self) -> None:
        _check_unsupported('echo "`ls`"')  # should not raise

    def test_lt_paren_inside_quotes_ignored(self) -> None:
        _check_unsupported("echo '<(foo)'")  # should not raise

    def test_arithmetic_rejected_in_parse(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse_command(
                "echo $((1+1))",
                lambda inner: (0, b""),
                None, 30, 0,
            )
        self.assertIn("Arithmetic", str(ctx.exception))


class LexerHeredocBodyTest(unittest.TestCase):
    """Test heredoc body collection via parse_command."""

    def _parse(self, cmd: str):
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b""
        cleaned, expansion, prog = parse_command(
            cmd, fake_capture, None, 30, 0,
        )
        return cleaned, expansion, prog

    def test_basic_heredoc(self) -> None:
        cleaned, exp, prog = self._parse("cat <<EOF\nhello\nworld\nEOF")
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        body = exp.heredoc_for(part)
        self.assertIsNotNone(body)
        self.assertEqual(body, "hello\nworld\n")

    def test_heredoc_tab_strip(self) -> None:
        cleaned, exp, prog = self._parse("cat <<-EOF\n\t\thello\n\tEOF")
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertIn("<<-", cleaned)
        self.assertEqual(exp.heredoc_for(part), "hello\n")

    def test_heredoc_single_quoted_delim(self) -> None:
        cleaned, exp, prog = self._parse("cat <<'EOF'\n$(echo hi)\nEOF")
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        # Body should be literal — $() NOT expanded
        self.assertEqual(exp.heredoc_for(part), "$(echo hi)\n")

    def test_heredoc_backslash_escaped_delim(self) -> None:
        """<<\\EOF should act like <<'EOF' — literal body."""
        cleaned, exp, prog = self._parse("cat <<\\EOF\n$(echo hi)\nEOF")
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "$(echo hi)\n")

    def test_heredoc_dash_backslash_escaped_delim(self) -> None:
        """<<-\\EOF should act like <<-'EOF' — literal body with tab strip."""
        cleaned, exp, prog = self._parse("cat <<-\\EOF\n\t$(echo hi)\n\tEOF")
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "$(echo hi)\n")

    def test_heredoc_missing_terminator(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._parse("cat <<EOF\nhello\nworld\n")
        self.assertIn("not found", str(ctx.exception))

    def test_herestring_basic(self) -> None:
        cleaned, exp, prog = self._parse("cat <<<hello world")
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "hello world\n")

    def test_herestring_single_quoted(self) -> None:
        cleaned, exp, prog = self._parse("cat <<<'hello world'")
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "hello world\n")

    def test_herestring_expands_dollar_paren(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"expanded"
        cleaned, exp, prog = parse_command(
            "cat <<<$(echo hi)", fake_capture, None, 30, 0,
        )
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "expanded\n")

    def test_heredoc_body_expands_dollar_paren_in_unquoted(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"hello"
        cleaned, exp, prog = parse_command(
            "cat <<EOF\n$(echo hello)\nEOF", fake_capture, None, 30, 0,
        )
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "hello\n")


class LexerSubstSpanTest(unittest.TestCase):
    """Test $(...) span extraction."""

    def test_basic_subst(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            self.assertEqual(inner, "echo hello")
            return 0, b"hello"
        cleaned, exp, prog = parse_command(
            "echo $(echo hello)", fake_capture, None, 30, 0,
        )
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.arg_for(part), "hello")

    def test_nested_subst(self) -> None:
        captured: list[str] = []

        def fake_capture(inner: str) -> tuple[int, bytes]:
            captured.append(inner)
            return 0, inner.encode()

        cleaned, exp, prog = parse_command(
            "echo $(echo $(echo inner))", fake_capture, None, 30, 0,
        )
        # The outer capture is called with the whole inner text
        self.assertIn("echo $(echo inner)", captured)

    def test_unbalanced_dollar_paren(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_command(
                "echo $(unclosed", lambda inner: (0, b""), None, 30, 0,
            )
        self.assertIn("Unbalanced", str(ctx.exception))

    def test_escaped_dollar_paren_not_expanded(self) -> None:
        """\\$(echo hi) in heredoc body stays literal."""
        # Use _expand_subst_in_text directly
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"EXPANDED"

        result = _expand_subst_in_text("\\$(echo hi)", fake_capture)
        self.assertEqual(result, "\\$(echo hi)")

    def test_dollar_paren_expanded_in_unquoted_body(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"replaced"

        result = _expand_subst_in_text("prefix $(echo hi) suffix", fake_capture)
        self.assertEqual(result, "prefix replaced suffix")


class LexerQuoteStateTest(unittest.TestCase):
    """Test quote-awareness in lexer and parser."""

    def test_operators_inside_quotes_preserved(self) -> None:
        # split_legacy should treat operators inside quotes as literal
        result = split_legacy('echo "a|b" | wc')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], ['echo "a|b"', "wc"])

    def test_semicolon_inside_quotes_preserved(self) -> None:
        result = split_legacy('echo "a;b"')
        self.assertEqual(result, [(None, ['echo "a;b"'], False)])

    def test_ampersand_inside_quotes_preserved(self) -> None:
        result = split_legacy('echo "a & b"')
        self.assertEqual(result, [(None, ['echo "a & b"'], False)])

    def test_redirect_inside_quotes_literal(self) -> None:
        args, redirs, err = extract_redirects('echo ">" hello')
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ">", "hello"])
        self.assertEqual(len(redirs), 0)

    def test_redirect_inside_single_quotes_literal(self) -> None:
        args, redirs, err = extract_redirects("echo '>' hello")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ">", "hello"])
        self.assertEqual(len(redirs), 0)

    def test_unbalanced_quotes_error(self) -> None:
        args, redirs, err = extract_redirects('echo "hi')
        self.assertEqual(err, "Unbalanced quotes in command")

    def test_glued_redirect_not_recognized(self) -> None:
        # foo>bar — > is not at word boundary, treated as literal
        args, redirs, err = extract_redirects("echo foo>bar")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "foo>bar"])
        self.assertEqual(len(redirs), 0)


class PositionalParameterLexingTest(unittest.TestCase):
    """Test that $1, $#, $@, $* lex as VARREF tokens."""

    def _tokenize(self, command: str):
        from shell_sandbox_mcp.parser import Lexer
        return Lexer(command).tokenize()

    def test_dollar1_varref(self) -> None:
        tokens = self._tokenize("echo $1")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF and t.value == "1"]
        self.assertEqual(len(varrefs), 1)

    def test_dollar_hash_varref(self) -> None:
        tokens = self._tokenize("echo $#")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF and t.value == "#"]
        self.assertEqual(len(varrefs), 1)

    def test_dollar_at_varref(self) -> None:
        tokens = self._tokenize("echo $@")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF and t.value == "@"]
        self.assertEqual(len(varrefs), 1)

    def test_dollar_star_varref(self) -> None:
        tokens = self._tokenize("echo $*")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF and t.value == "*"]
        self.assertEqual(len(varrefs), 1)

    def test_dollar0_varref(self) -> None:
        tokens = self._tokenize("echo $0")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF and t.value == "0"]
        self.assertEqual(len(varrefs), 1)


class FuncParensLexingTest(unittest.TestCase):
    """Test that FUNC_PARENS tokens are emitted for function definitions."""

    def _tokenize(self, command: str):
        from shell_sandbox_mcp.parser import Lexer
        return Lexer(command).tokenize()

    def test_posix_func_parens_emitted(self) -> None:
        tokens = self._tokenize("f() echo hi")
        kinds = [t.kind for t in tokens]
        self.assertIn(TokenKind.FUNC_PARENS, kinds)

    def test_func_parens_not_emitted_for_keyword_form(self) -> None:
        """function f echo — no FUNC_PARENS."""
        tokens = self._tokenize("function f echo hi")
        kinds = [t.kind for t in tokens]
        self.assertNotIn(TokenKind.FUNC_PARENS, kinds)

    def test_func_parens_emitted_for_keyword_with_parens(self) -> None:
        """function f() echo — has FUNC_PARENS."""
        tokens = self._tokenize("function f() echo hi")
        kinds = [t.kind for t in tokens]
        self.assertIn(TokenKind.FUNC_PARENS, kinds)


if __name__ == "__main__":
    unittest.main()
