import unittest
from shell_sandbox_mcp.parser import (
    Expansion, ParseError, SENTINEL_HD,
    extract_redirects, parse_command,
)


class HeredocBackslashDelimTest(unittest.TestCase):
    """Test <<\\EOF (backslash-escaped delimiter)."""

    def _parse(self, cmd):
        def fake(inner):
            return 0, b''
        return parse_command(cmd, fake, None, 30, 0)

    def test_backslash_delim_literal_body(self):
        cmd = "cat <<\\EOF\n$(echo hi)\nEOF"
        cleaned, exp, _prog = self._parse(cmd)
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")


class UnsupportedRejectionTest(unittest.TestCase):
    def test_arithmetic_rejected(self):
        with self.assertRaises(ParseError) as ctx:
            parse_command("echo $((1+1))", lambda i: (0, b""), None, 30, 0)
        self.assertIn("Arithmetic", str(ctx.exception))

    def test_backtick_rejected(self):
        with self.assertRaises(ParseError):
            parse_command("echo `ls`", lambda i: (0, b""), None, 30, 0)

    def test_lt_paren_rejected(self):
        with self.assertRaises(ParseError):
            parse_command("diff <(echo a)", lambda i: (0, b""), None, 30, 0)

    def test_gt_paren_rejected(self):
        with self.assertRaises(ParseError):
            parse_command("echo >(cat)", lambda i: (0, b""), None, 30, 0)

    def test_backtick_inside_quotes_passes(self):
        cleaned, exp, _prog = parse_command(
            "echo '`ls`'", lambda i: (0, b""), None, 30, 0,
        )
        self.assertIn("`ls`", cleaned)

    def test_lt_paren_inside_quotes_passes(self):
        cleaned, exp, _prog = parse_command(
            'echo "<(foo)"', lambda i: (0, b""), None, 30, 0,
        )
        self.assertIn("<(foo)", cleaned)


class LegacyBehaviorTest(unittest.TestCase):
    def test_glued_not_redirect(self):
        args, redirs, err = extract_redirects("echo foo>bar")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "foo>bar"])

    def test_empty_quotes_dropped(self):
        args, redirs, err = extract_redirects('echo ""')
        self.assertIsNone(err)
        self.assertEqual(args, ["echo"])

    def test_empty_single_quotes_dropped(self):
        args, redirs, err = extract_redirects("echo ''")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo"])


class ParseErrorSubclassTest(unittest.TestCase):
    def test_is_value_error(self):
        self.assertTrue(issubclass(ParseError, ValueError))

    def test_caught_as_value_error(self):
        try:
            raise ParseError("x")
        except ValueError:
            pass
        else:
            self.fail("ParseError should be caught as ValueError")


# ---------------------------------------------------------------------------
# Backslash escape tests (§6 of the refactor spec)
# ---------------------------------------------------------------------------

class BackslashEscapeTest(unittest.TestCase):
    """Test real backslash escape handling outside and inside quotes."""

    def _parse(self, cmd, outputs=None):
        outputs = outputs or {}
        def fake_capture(inner):
            val = outputs.get(inner, "")
            return 0, val.encode()
        return parse_command(cmd, fake_capture, None, 30, 0)

    def test_backslash_space_escapes_space(self) -> None:
        """echo a\\ b → args ['echo', 'a b'] — backslash escapes the space."""
        args, redirs, err = extract_redirects("echo a\\ b")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a b"])

    def test_backslash_semicolon_literal(self) -> None:
        """echo \\; → the ; is NOT an operator."""
        args, redirs, err = extract_redirects("echo \\;")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ";"])

    def test_backslash_dollar_literal(self) -> None:
        """echo \\$ → literal $, not substitution start."""
        args, redirs, err = extract_redirects("echo \\$")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "$"])

    def test_backslash_escaped_dollar_paren_not_subst(self) -> None:
        """\\$(cmd) → the backslash prevents $() expansion."""
        cleaned, exp, prog = self._parse("echo \\$(whoami)", {"whoami": "SHOULD_NOT_MATCH"})
        # $(...) inside should NOT be expanded because of backslash
        self.assertNotIn("\x01A", cleaned)
        self.assertIn("\\$(whoami)", cleaned)

    def test_backslash_quote_inside_double_quotes(self) -> None:
        """\\\" inside \"...\" → literal quote, does NOT close the string."""
        # echo "hello\\"world" → the \\" is an escaped quote, not a closing quote
        # So the word is: hello"world (with embedded quote)
        args, redirs, err = extract_redirects('echo "hello\\"world"')
        self.assertIsNone(err)
        # After quote stripping: hello"world
        self.assertEqual(args, ["echo", 'hello"world'])

    def test_double_quoted_backslash_dollar(self) -> None:
        """\\$ inside \"...\" → literal $."""
        args, redirs, err = extract_redirects('echo "\\$HOME"')
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "$HOME"])

    def test_double_quoted_backslash_backslash(self) -> None:
        """\\\\ inside \"...\" → literal \\."""
        args, redirs, err = extract_redirects('echo "\\\\"')
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "\\"])

    def test_single_quoted_backslash_literal(self) -> None:
        """\\ inside '...' is always literal."""
        args, redirs, err = extract_redirects("echo '\\n\\t'")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "\\n\\t"])

    def test_python_c_with_escaped_quote_and_heredoc(self) -> None:
        """python -c \"...\\\"... << ...\" — the << inside quotes is NOT a heredoc."""
        # This tests that \" does NOT prematurely close the quote and mis-parse <<
        args, redirs, err = extract_redirects(
            'python -c "print(\\"hello\\") << end"'
        )
        self.assertIsNone(err)
        # << end should be part of the quoted string, not a heredoc redirect
        self.assertEqual(len(redirs), 0)

    def test_backslash_newline_line_continuation_in_double_quotes(self) -> None:
        """\\ + newline inside \"...\" is line continuation."""
        # The backslash-newline is stripped
        args, redirs, err = extract_redirects('echo "hello\\\nworld"')
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "helloworld"])


# ---------------------------------------------------------------------------
# Arithmetic inside quotes stays literal
# ---------------------------------------------------------------------------

class ArithmeticInQuotesTest(unittest.TestCase):
    """$(( inside double quotes is literal, rejected only when unquoted."""

    def test_arithmetic_inside_double_quotes_passes(self) -> None:
        cleaned, exp, _prog = parse_command(
            'echo "$((1+1))"', lambda i: (0, b""), None, 30, 0,
        )
        self.assertIn("$((1+1))", cleaned)

    def test_arithmetic_inside_single_quotes_passes(self) -> None:
        cleaned, exp, _prog = parse_command(
            "echo '$((1+1))'", lambda i: (0, b""), None, 30, 0,
        )
        self.assertIn("$((1+1))", cleaned)


# ---------------------------------------------------------------------------
# Nested $(...) and heredoc body edge cases
# ---------------------------------------------------------------------------

class NestedSubstAndHeredocTest(unittest.TestCase):
    """Test nested $(...), heredoc body with heredoc-like line."""

    def _parse(self, cmd, outputs=None):
        outputs = outputs or {}
        def fake_capture(inner):
            val = outputs.get(inner, "")
            return 0, val.encode()
        return parse_command(cmd, fake_capture, None, 30, 0)

    def test_nested_dollar_paren(self) -> None:
        captured = []
        def fake_capture(inner):
            captured.append(inner)
            return 0, inner.encode()
        cleaned, exp, prog = parse_command(
            "echo $(echo $(echo inner))", fake_capture, None, 30, 0,
        )
        self.assertIn("echo $(echo inner)", captured)

    def test_heredoc_body_containing_heredoc_like_line(self) -> None:
        """Heredoc body containing a line that looks like another heredoc."""
        cmd = "cat <<EOF\nline1\n<<IGNORED\nline3\nEOF"
        cleaned, exp, prog = self._parse(cmd)
        m = SENTINEL_HD.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        body = exp.heredoc_bodies.get(sentinel)
        self.assertIsNotNone(body)
        # The <<IGNORED line should be in the body, not treated as a new heredoc
        self.assertIn("<<IGNORED", body)


if __name__ == "__main__":
    unittest.main()
