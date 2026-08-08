import tempfile
import unittest
from pathlib import Path
from typing import Mapping

from shell_sandbox_mcp.parser import (
    Expansion, ParseError,
    extract_redirects, parse_command, program_to_chain,
)


def _extract(cmd: str, env: Mapping[str, str]) -> list[str]:
    """Parse *cmd* with *env* and return the extracted args (AST path)."""
    with tempfile.TemporaryDirectory() as td:
        _cleaned, exp, prog = parse_command(
            cmd, lambda i: (0, b""), Path(td), 30, 0, env=env,
        )
        chain = program_to_chain(prog)
        cmd_node = chain[0][1][0]
        args, _redirs, err = extract_redirects(cmd_node, exp)
        if err is not None:
            raise AssertionError(f"extract error for {cmd!r}: {err}")
        return args


class HeredocBackslashDelimTest(unittest.TestCase):
    """Test <<\\EOF (backslash-escaped delimiter)."""

    def _parse(self, cmd):
        def fake(inner):
            return 0, b''
        return parse_command(cmd, fake, None, 30, 0)

    def test_backslash_delim_literal_body(self):
        cmd = "cat <<\\EOF\n$(echo hi)\nEOF"
        cleaned, exp, prog = self._parse(cmd)
        cmd_node = prog.chains[0].pipeline.commands[0]
        part = None
        for rs in cmd_node.redirects:
            for p in rs.target.parts:
                if p.is_hd_sentinel:
                    part = p
                    break
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "$(echo hi)\n")


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

    def test_empty_quotes_kept(self):
        # POSIX: a quoted empty string is still one (empty) argv entry.
        args, redirs, err = extract_redirects('echo ""')
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ""])

    def test_empty_single_quotes_kept(self):
        args, redirs, err = extract_redirects("echo ''")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ""])

    def test_empty_quotes_before_empty_var_kept(self):
        # ""$X with X="" → the "" literal is a genuine empty arg (POSIX),
        # even though it's adjacent to an expansion sentinel.
        self.assertEqual(_extract('echo ""$X', {"X": ""}), ["echo", ""])

    def test_empty_var_then_quotes_kept(self):
        # $X"" with X="" → same: the trailing "" literal is preserved.
        self.assertEqual(_extract('echo $X""', {"X": ""}), ["echo", ""])

    def test_empty_quotes_around_empty_var_kept(self):
        # "$X" with X="" → the quoted empty expansion is preserved.
        self.assertEqual(_extract('echo "$X"', {"X": ""}), ["echo", ""])

    def test_empty_var_adjacent_no_spurious_arg(self):
        # "a$X"b with X="" → no extra empty arg; text just concatenates.
        self.assertEqual(_extract('echo "a$X"b', {"X": ""}), ["echo", "ab"])


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
        cmd = prog.chains[0].pipeline.commands[0]
        has_arg_sentinel = any(
            p.is_arg_sentinel for w in cmd.words for p in w.parts
        )
        self.assertFalse(has_arg_sentinel)
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
    """$(( inside double quotes is now rejected (since $( is active inside "...")."""

    def test_arithmetic_inside_double_quotes_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse_command(
                'echo "$((1+1))"', lambda i: (0, b""), None, 30, 0,
            )
        self.assertIn("Arithmetic", str(ctx.exception))

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
        cmd_node = prog.chains[0].pipeline.commands[0]
        part = None
        for rs in cmd_node.redirects:
            for p in rs.target.parts:
                if p.is_hd_sentinel:
                    part = p
                    break
        self.assertIsNotNone(part)
        body = exp.heredoc_for(part)
        self.assertIsNotNone(body)
        # The <<IGNORED line should be in the body, not treated as a new heredoc
        self.assertIn("<<IGNORED", body)


if __name__ == "__main__":
    unittest.main()
