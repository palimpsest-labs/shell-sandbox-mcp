"""Tests for AST-native case/esac and subshell parsing.

Run with::

    PYTHONPATH=src python3 -m pytest tests/test_parser_case.py -q
"""

import unittest

from shell_sandbox_mcp.parser import (
    CaseClause,
    CaseNode,
    CommandNode,
    ParseError,
    SubshellNode,
    _build_ast,
    Expansion,
)
from shell_sandbox_mcp.parser import Lexer


# Helpers to parse a command string and extract the first chain's first pipeline's first command.
def _parse_first_cmd(command: str):
    """Parse *command* and return the first CommandLike, or None."""
    tokens = Lexer(command).tokenize()
    program = _build_ast(tokens, Expansion(), command=command)
    if not program.chains:
        return None
    pipeline = program.chains[0].pipeline
    if not pipeline.commands:
        return None
    return pipeline.commands[0]


def _parse_program(command: str):
    """Parse *command* and return the full AST."""
    tokens = Lexer(command).tokenize()
    return _build_ast(tokens, Expansion(), command=command)


# ---------------------------------------------------------------------------
# CaseNode parsing tests
# ---------------------------------------------------------------------------


class CaseParsingTest(unittest.TestCase):
    """Test that the parser correctly produces CaseNode ASTs."""

    def test_simple_case(self) -> None:
        cmd = _parse_first_cmd("case $x in a) echo yes;; esac")
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertEqual(node.subject.strip(), "$x")
        self.assertEqual(len(node.clauses), 1)
        self.assertEqual(node.clauses[0].pattern, "a")
        self.assertIn("echo yes", node.clauses[0].body)

    def test_case_multiple_clauses(self) -> None:
        cmd = _parse_first_cmd(
            "case $x in a) echo a;; b) echo b;; esac"
        )
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertEqual(len(node.clauses), 2)
        self.assertEqual(node.clauses[0].pattern, "a")
        self.assertEqual(node.clauses[1].pattern, "b")

    def test_case_default_star(self) -> None:
        cmd = _parse_first_cmd(
            "case $x in a) echo a;; *) echo default;; esac"
        )
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertEqual(len(node.clauses), 2)
        self.assertEqual(node.clauses[1].pattern, "*")

    def test_case_pipe_alternation(self) -> None:
        cmd = _parse_first_cmd(
            "case $x in a|b) echo ab;; esac"
        )
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertEqual(node.clauses[0].pattern, "a|b")

    def test_case_no_dsemi_last_clause(self) -> None:
        """Last clause before esac does not need ;;."""
        cmd = _parse_first_cmd(
            "case $x in a) echo a;; b) echo b esac"
        )
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertEqual(len(node.clauses), 2)

    def test_case_optional_leading_lparen(self) -> None:
        cmd = _parse_first_cmd(
            "case $x in (a) echo yes;; esac"
        )
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertEqual(node.clauses[0].pattern, "a")

    def test_case_glob_pattern(self) -> None:
        cmd = _parse_first_cmd(
            "case $x in *.txt) echo text;; esac"
        )
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertIn("*", node.clauses[0].pattern)

    def test_case_quoted_pattern(self) -> None:
        cmd = _parse_first_cmd(
            'case $x in "a b") echo yes;; esac'
        )
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertIn('"a b"', node.clauses[0].pattern)

    def test_case_with_varref_subject(self) -> None:
        cmd = _parse_first_cmd(
            "case $HOME in /home/*) echo home;; esac"
        )
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertIn("$HOME", node.subject)

    def test_case_missing_esac(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("case $x in a) echo yes;;")

    def test_case_missing_in(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("case $x a) echo yes;; esac")

    def test_case_empty_subject(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("case in a) echo yes;; esac")
        self.assertIn("missing subject", str(ctx.exception))

    def test_case_in_pipe_rejected(self) -> None:
        """'echo hi | case $x in a) echo ok;; esac' → ParseError."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("echo hi | case $x in a) echo ok;; esac")
        self.assertIn("compound command cannot appear in a pipe", str(ctx.exception))

    def test_echo_case_is_not_compound(self) -> None:
        """'echo case' — case is an argument, not a keyword."""
        cmd = _parse_first_cmd("echo case $x in a")
        self.assertIsInstance(cmd, CommandNode)

    def test_quoted_case_is_not_keyword(self) -> None:
        """'\"case\"' — quoted case is not a keyword, is a regular command."""
        # "case" is a quoted word, not the keyword.  The rest ($x in a) echo yes;; esac)
        # is parsed as additional pipeline stages; esac appears at command position
        # after ;; and is rejected.
        with self.assertRaises(ParseError):
            _parse_first_cmd('"case" $x in a) echo yes;; esac')

    def test_nested_case_in_if(self) -> None:
        """If containing a case in its body."""
        from shell_sandbox_mcp.parser import IfNode
        cmd = _parse_first_cmd(
            "if true; then case $x in a) echo hi;; esac; fi"
        )
        self.assertIsInstance(cmd, IfNode)

    def test_nested_if_in_case_body(self) -> None:
        """Case body containing an if."""
        from shell_sandbox_mcp.parser import IfNode
        # The if inside the case body text is opaque (stored as raw text),
        # but the parser should correctly identify the outer case.
        cmd = _parse_first_cmd(
            "case $x in a) if true; then echo hi; fi;; esac"
        )
        self.assertIsInstance(cmd, CaseNode)

    def test_bare_esac_is_error(self) -> None:
        """'esac' at command position → ParseError."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("esac")

    def test_case_with_heredoc_in_body(self) -> None:
        """Case body can contain heredocs."""
        cmd = _parse_first_cmd(
            "case $x in a) cat <<EOF\nhello\nEOF\n;; esac"
        )
        self.assertIsInstance(cmd, CaseNode)


# ---------------------------------------------------------------------------
# SubshellNode parsing tests
# ---------------------------------------------------------------------------


class SubshellParsingTest(unittest.TestCase):
    """Test that the parser correctly produces SubshellNode ASTs."""

    def test_simple_subshell(self) -> None:
        cmd = _parse_first_cmd("(echo hi)")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("echo hi", node.body)

    def test_subshell_with_semicolon(self) -> None:
        cmd = _parse_first_cmd("(echo hi; echo there)")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("echo hi", node.body)
        self.assertIn("echo there", node.body)

    def test_nested_subshell(self) -> None:
        cmd = _parse_first_cmd("( ( echo inner ) )")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("( echo inner )", node.body)

    def test_if_inside_subshell(self) -> None:
        cmd = _parse_first_cmd("( if true; then echo hi; fi )")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("if true", node.body)
        self.assertIn("fi", node.body)

    def test_subshell_unbalanced_error(self) -> None:
        """'(echo hi' — missing closing paren → ParseError."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("(echo hi")

    def test_double_paren_arithmetic_error(self) -> None:
        """'(( 1+1 ))' — arithmetic command → ParseError."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("(( 1+1 ))")
        self.assertIn("Arithmetic command", str(ctx.exception))

    def test_subshell_in_pipe_rejected(self) -> None:
        """'echo hi | (echo ok)' → ParseError."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("echo hi | (echo ok)")
        self.assertIn("compound command cannot appear in a pipe", str(ctx.exception))

    def test_echo_paren_is_not_subshell(self) -> None:
        """'echo (hi)' — paren is an argument, not a subshell."""
        cmd = _parse_first_cmd("echo (hi)")
        self.assertIsInstance(cmd, CommandNode)

    def test_echo_open_paren_is_argument(self) -> None:
        """'echo (' — starts a subshell at command position but not mid-word."""
        # Actually, '(' at non-command position should be treated as a word
        # character. Let's test that echo '(' is an argument.
        cmd = _parse_first_cmd("echo (")
        self.assertIsInstance(cmd, CommandNode)

    def test_dollar_subst_still_works(self) -> None:
        """$(...) still parses correctly."""
        cmd = _parse_first_cmd("echo $(whoami)")
        self.assertIsInstance(cmd, CommandNode)

    def test_dollar_dollar_paren_still_error(self) -> None:
        """$(( )) still raises ParseError."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("echo $(( 1+1 ))")

    def test_process_subst_still_error(self) -> None:
        """<( ) still raises ParseError."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("echo <(cat)")
        with self.assertRaises(ParseError):
            _parse_first_cmd("echo >(cat)")

    # -- BLOCKER 1 regression tests: parens inside subshell bodies ----------

    def test_subshell_with_unquoted_parens_in_args(self) -> None:
        """( echo (hi) ) — unquoted paren pairs stay inside the word."""
        cmd = _parse_first_cmd("( echo (hi) )")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("echo (hi)", node.body)

    def test_case_inside_subshell(self) -> None:
        """( case $x in a) echo ok;; esac ) — case inside subshell."""
        cmd = _parse_first_cmd("( case $x in a) echo ok;; esac )")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("case $x in", node.body)
        self.assertIn("echo ok", node.body)
        self.assertIn("esac", node.body)

    def test_assignment_with_paren_in_value_inside_subshell(self) -> None:
        """( x=(a) ) — assignment with paren in value."""
        cmd = _parse_first_cmd("( x=(a) )")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("x=(a)", node.body)

    def test_arg_ending_in_rparen_inside_subshell(self) -> None:
        """( echo a) ) — arg ending in ')'."""
        cmd = _parse_first_cmd("( echo a) )")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("echo a", node.body)

    def test_backslash_escaped_parens_inside_subshell(self) -> None:
        r"""( echo \(hi\) ) — backslash-escaped parens."""
        cmd = _parse_first_cmd(r"( echo \(hi\) )")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("echo \\(hi\\)", node.body)

    def test_triple_nested_subshell_with_paren_args(self) -> None:
        """( ( echo (a) ) ) — triple-nested subshell."""
        cmd = _parse_first_cmd("( ( echo (a) ) )")
        self.assertIsInstance(cmd, SubshellNode)
        node: SubshellNode = cmd
        self.assertIn("( echo (a) )", node.body)

    def test_case_pattern_with_nested_parens(self) -> None:
        r"""case $x in @(a|b)) echo ok;; esac — nested parens in pattern."""
        cmd = _parse_first_cmd("case $x in @(a|b)) echo ok;; esac")
        self.assertIsInstance(cmd, CaseNode)
        node: CaseNode = cmd
        self.assertEqual(node.clauses[0].pattern, "@(a|b)")


if __name__ == "__main__":
    unittest.main()
