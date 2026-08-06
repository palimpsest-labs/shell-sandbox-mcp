"""Unit tests for the shell parser lexer — token kinds, positions,
quote state, redirect-at-boundary, heredoc body collection, $(...) spans."""

import unittest

from shell_sandbox_mcp.parser import (
    Expansion,
    ParseError,
    Redirect,
    SENTINEL_ARG,
    SENTINEL_HD,
    TokenKind,
    Token,
    _check_unsupported,
    _expand_subst_in_text,
    _strip_quotes,
    extract_redirects,
    parse_command,
    split_legacy,
)


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

    def _parse(self, cmd: str) -> tuple[str, Expansion]:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b""
        cleaned, expansion, _prog = parse_command(
            cmd, fake_capture, None, 30, 0,
        )
        return cleaned, expansion

    def test_basic_heredoc(self) -> None:
        cleaned, exp = self._parse("cat <<EOF\nhello\nworld\nEOF")
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn(sentinel, exp.heredoc_bodies)
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\nworld\n")

    def test_heredoc_tab_strip(self) -> None:
        cleaned, exp = self._parse("cat <<-EOF\n\t\thello\n\tEOF")
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn("<<-", cleaned)
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")

    def test_heredoc_single_quoted_delim(self) -> None:
        cleaned, exp = self._parse("cat <<'EOF'\n$(echo hi)\nEOF")
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        # Body should be literal — $() NOT expanded
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")

    def test_heredoc_backslash_escaped_delim(self) -> None:
        """<<\EOF should act like <<'EOF' — literal body."""
        cleaned, exp = self._parse("cat <<\\EOF\n$(echo hi)\nEOF")
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")

    def test_heredoc_dash_backslash_escaped_delim(self) -> None:
        """<<-\EOF should act like <<-'EOF' — literal body with tab strip."""
        cleaned, exp = self._parse("cat <<-\\EOF\n\t$(echo hi)\n\tEOF")
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")

    def test_heredoc_missing_terminator(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._parse("cat <<EOF\nhello\nworld\n")
        self.assertIn("not found", str(ctx.exception))

    def test_herestring_basic(self) -> None:
        cleaned, exp = self._parse("cat <<<hello world")
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello world\n")

    def test_herestring_single_quoted(self) -> None:
        cleaned, exp = self._parse("cat <<<'hello world'")
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello world\n")

    def test_herestring_expands_dollar_paren(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"expanded"
        cleaned, exp, _prog = parse_command(
            "cat <<<$(echo hi)", fake_capture, None, 30, 0,
        )
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "expanded\n")

    def test_heredoc_body_expands_dollar_paren_in_unquoted(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"hello"
        cleaned, exp, _prog = parse_command(
            "cat <<EOF\n$(echo hello)\nEOF", fake_capture, None, 30, 0,
        )
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")


class LexerSubstSpanTest(unittest.TestCase):
    """Test $(...) span extraction."""

    def test_basic_subst(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            self.assertEqual(inner, "echo hello")
            return 0, b"hello"
        cleaned, exp, _prog = parse_command(
            "echo $(echo hello)", fake_capture, None, 30, 0,
        )
        m = SENTINEL_ARG.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01A{m.group(1)}\x01"
        self.assertEqual(exp.arg_values[sentinel], "hello")

    def test_nested_subst(self) -> None:
        captured: list[str] = []

        def fake_capture(inner: str) -> tuple[int, bytes]:
            captured.append(inner)
            return 0, inner.encode()

        cleaned, exp, _prog = parse_command(
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


if __name__ == "__main__":
    unittest.main()
