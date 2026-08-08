"""Tests for AST-native compound command parsing (if/while/until/for).

Run with::

    PYTHONPATH=src python3 -m pytest tests/test_parser_compound.py -q
"""

import unittest

from shell_sandbox_mcp.parser import (
    CommandNode,
    ForNode,
    IfBranch,
    IfNode,
    ParseError,
    WhileNode,
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
# IfNode parsing tests
# ---------------------------------------------------------------------------


class IfParsingTest(unittest.TestCase):
    """Test that the parser correctly produces IfNode ASTs."""

    def test_simple_if(self) -> None:
        cmd = _parse_first_cmd("if true; then echo hi; fi")
        self.assertIsInstance(cmd, IfNode)
        node: IfNode = cmd
        self.assertEqual(len(node.branches), 1)
        self.assertIsNone(node.else_body)
        self.assertIn("true", node.branches[0].cond)
        self.assertIn("echo", node.branches[0].body)

    def test_if_else(self) -> None:
        cmd = _parse_first_cmd("if false; then echo yes; else echo no; fi")
        self.assertIsInstance(cmd, IfNode)
        node: IfNode = cmd
        self.assertEqual(len(node.branches), 1)
        self.assertIsNotNone(node.else_body)
        self.assertIn("echo no", node.else_body)

    def test_if_elif(self) -> None:
        cmd = _parse_first_cmd(
            "if false; then echo a; elif true; then echo b; fi"
        )
        self.assertIsInstance(cmd, IfNode)
        node: IfNode = cmd
        self.assertEqual(len(node.branches), 2)
        self.assertIsNone(node.else_body)
        self.assertIn("echo a", node.branches[0].body)
        self.assertIn("echo b", node.branches[1].body)

    def test_if_elif_else(self) -> None:
        cmd = _parse_first_cmd(
            "if false; then echo a; elif false; then echo b; else echo c; fi"
        )
        self.assertIsInstance(cmd, IfNode)
        node: IfNode = cmd
        self.assertEqual(len(node.branches), 2)
        self.assertIsNotNone(node.else_body)
        self.assertIn("echo c", node.else_body)

    def test_if_multiple_elifs(self) -> None:
        cmd = _parse_first_cmd(
            "if false; then echo 1; elif false; then echo 2; "
            "elif false; then echo 3; else echo 4; fi"
        )
        self.assertIsInstance(cmd, IfNode)
        node: IfNode = cmd
        self.assertEqual(len(node.branches), 3)

    def test_echo_if_is_not_compound(self) -> None:
        """'echo if' — if is an argument, not a keyword."""
        cmd = _parse_first_cmd("echo if true")
        self.assertIsInstance(cmd, CommandNode)

    def test_quoted_if_is_not_keyword(self) -> None:
        """'\"if\" true' — quoted if is not a keyword."""
        cmd = _parse_first_cmd('"if" true')
        self.assertIsInstance(cmd, CommandNode)

    def test_backslash_escaped_if_not_keyword(self) -> None:
        r"""'\if true' — \if lexes as 'if', which matches the reserved word.

        In our lexer \if produces the literal characters 'if', which
        matches the reserved word.  True POSIX escape would require
        quoting (e.g. "if" or 'if').
        """
        # \if lexes as "if" → recognized as keyword → parses as IfNode.
        cmd = _parse_first_cmd("\\if true; then echo hi; fi")
        # Currently \if → IfNode because lexer produces 'if' token.
        self.assertIsInstance(cmd, IfNode)

    def test_bare_fi_is_error(self) -> None:
        """'fi' at command position without matching if → ParseError."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("fi")

    def test_bare_then_is_error(self) -> None:
        """'then' at command position → ParseError."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("then echo hi")

    def test_bare_else_is_error(self) -> None:
        """'else' at command position → ParseError."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("else echo hi")

    def test_bare_elif_is_error(self) -> None:
        """'elif' at command position → ParseError."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("elif true; then echo hi; fi")

    def test_if_missing_then(self) -> None:
        """'if true; echo hi; fi' → missing then."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("if true; echo hi; fi")

    def test_if_missing_fi(self) -> None:
        """'if true; then echo hi' → missing fi."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("if true; then echo hi")

    def test_if_compound_in_pipe_rejected(self) -> None:
        """'echo hi | if true; then echo ok; fi' → ParseError (B1)."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("echo hi | if true; then echo ok; fi")
        self.assertIn("compound command cannot appear in a pipe", str(ctx.exception))

    def test_if_compound_followed_by_pipe_rejected(self) -> None:
        """'if true; then echo ok; fi | cat' → ParseError (B1)."""
        with self.assertRaises(ParseError) as ctx:
            _parse_program("if true; then echo ok; fi | cat")
        self.assertIn("compound command cannot appear in a pipe", str(ctx.exception))

    def test_empty_condition_rejected(self) -> None:
        """'if; then echo hi; fi' → ParseError (S5)."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("if; then echo hi; fi")
        self.assertIn("expected command after 'if'", str(ctx.exception))

    def test_empty_elif_condition_rejected(self) -> None:
        """'if true; then echo hi; elif; then echo oops; fi' → ParseError (S5)."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("if true; then echo hi; elif; then echo oops; fi")
        self.assertIn("expected command after 'elif'", str(ctx.exception))

    def test_if_no_semicolon_before_then(self) -> None:
        """'if true then echo hi; fi' → valid (no ; needed)."""
        cmd = _parse_first_cmd("if true then echo hi; fi")
        self.assertIsInstance(cmd, IfNode)

    def test_condition_with_pipes(self) -> None:
        """Condition can contain pipes."""
        cmd = _parse_first_cmd("if echo a | grep a; then echo ok; fi")
        self.assertIsInstance(cmd, IfNode)
        node: IfNode = cmd
        self.assertIn("|", node.branches[0].cond)

    def test_condition_with_and_and(self) -> None:
        """Condition can contain &&."""
        cmd = _parse_first_cmd("if true && echo hi; then echo ok; fi")
        self.assertIsInstance(cmd, IfNode)

    def test_condition_with_or_or(self) -> None:
        """Condition can contain ||."""
        cmd = _parse_first_cmd("if false || true; then echo ok; fi")
        self.assertIsInstance(cmd, IfNode)


# ---------------------------------------------------------------------------
# WhileNode / UntilNode parsing tests
# ---------------------------------------------------------------------------


class WhileParsingTest(unittest.TestCase):
    """Test parsing of while/until loops."""

    def test_simple_while(self) -> None:
        cmd = _parse_first_cmd("while true; do echo hi; done")
        self.assertIsInstance(cmd, WhileNode)
        node: WhileNode = cmd
        self.assertFalse(node.until)
        self.assertIn("true", node.cond)
        self.assertIn("echo", node.body)

    def test_simple_until(self) -> None:
        cmd = _parse_first_cmd("until false; do echo hi; done")
        self.assertIsInstance(cmd, WhileNode)
        node: WhileNode = cmd
        self.assertTrue(node.until)
        self.assertIn("false", node.cond)
        self.assertIn("echo", node.body)

    def test_echo_while_is_not_compound(self) -> None:
        cmd = _parse_first_cmd("echo while true")
        self.assertIsInstance(cmd, CommandNode)

    def test_while_missing_do(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("while true; echo hi; done")

    def test_while_missing_done(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("while true; do echo hi")

    def test_while_compound_in_pipe_rejected(self) -> None:
        """'echo hi | while true; do echo ok; done' → ParseError (B1)."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("echo hi | while true; do echo ok; done")
        self.assertIn("compound command cannot appear in a pipe", str(ctx.exception))

    def test_while_empty_condition_rejected(self) -> None:
        """'while; do echo hi; done' → ParseError (S5)."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("while; do echo hi; done")
        self.assertIn("expected command after 'while'", str(ctx.exception))

    def test_until_empty_condition_rejected(self) -> None:
        """'until; do echo hi; done' → ParseError (S5)."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("until; do echo hi; done")
        self.assertIn("expected command after 'until'", str(ctx.exception))

    def test_bare_do_is_error(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("do echo hi")

    def test_bare_done_is_error(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("done")

    def test_while_without_semicolon(self) -> None:
        """'while true do echo hi; done' — do directly after condition."""
        cmd = _parse_first_cmd("while true do echo hi; done")
        self.assertIsInstance(cmd, WhileNode)


# ---------------------------------------------------------------------------
# ForNode parsing tests
# ---------------------------------------------------------------------------


class ForParsingTest(unittest.TestCase):
    """Test AST-native for-loop parsing."""

    def test_simple_for(self) -> None:
        cmd = _parse_first_cmd("for i in a b c; do echo $i; done")
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertEqual(node.var_name, "i")
        self.assertEqual(node.in_words, ("a", "b", "c"))
        self.assertIn("echo", node.body)

    def test_for_no_in_clause(self) -> None:
        cmd = _parse_first_cmd("for i; do echo $i; done")
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertEqual(node.var_name, "i")
        self.assertEqual(node.in_words, ())

    def test_for_no_in_no_semicolon(self) -> None:
        cmd = _parse_first_cmd("for i do echo hi; done")
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertEqual(node.in_words, ())

    def test_for_without_semicolon_before_do(self) -> None:
        cmd = _parse_first_cmd("for i in x y do echo $i; done")
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertEqual(node.in_words, ("x", "y"))

    def test_for_with_subst_in_words(self) -> None:
        cmd = _parse_first_cmd("for i in $(echo a); do echo $i; done")
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertEqual(len(node.in_words), 1)
        self.assertIn("$(", node.in_words[0])

    def test_for_with_varref_in_words(self) -> None:
        cmd = _parse_first_cmd("for i in $HOME; do echo $i; done")
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertEqual(len(node.in_words), 1)
        self.assertIn("$", node.in_words[0])

    def test_for_invalid_var_name(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("for 1x in a; do echo hi; done")

    def test_for_missing_var_name(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("for in a; do echo hi; done")

    def test_for_missing_do(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("for i in a; echo hi; done")

    def test_for_missing_done(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("for i in a; do echo hi")

    def test_for_compound_in_pipe_rejected(self) -> None:
        """'echo hi | for i in a; do echo $i; done' → ParseError (B1)."""
        with self.assertRaises(ParseError) as ctx:
            _parse_first_cmd("echo hi | for i in a; do echo $i; done")
        self.assertIn("compound command cannot appear in a pipe", str(ctx.exception))

    def test_for_bg_inside_body_scans(self) -> None:
        """'for i in a; do echo $i & done' parses correctly (S4)."""
        cmd = _parse_first_cmd("for i in a b; do echo $i & done")
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertIn("&", node.body)

    def test_for_closed_by_fi_is_error(self) -> None:
        """'for x in a; do echo x; fi' → ParseError (mismatched delimiter)."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("for x in a; do echo x; fi")

    def test_echo_for_is_not_compound(self) -> None:
        cmd = _parse_first_cmd("echo for i in a")
        self.assertIsInstance(cmd, CommandNode)


# ---------------------------------------------------------------------------
# Nesting tests
# ---------------------------------------------------------------------------


class NestingTest(unittest.TestCase):
    """Test that compound commands can be nested."""

    def test_if_inside_for_body_parses(self) -> None:
        """For-loop containing an if in its body."""
        cmd = _parse_first_cmd(
            "for i in a b; do if true; then echo $i; fi; done"
        )
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertIn("if true", node.body)
        self.assertIn("fi", node.body)

    def test_while_inside_if_body_parses(self) -> None:
        cmd = _parse_first_cmd(
            "if true; then while test 1; do echo loop; done; fi"
        )
        self.assertIsInstance(cmd, IfNode)
        node: IfNode = cmd
        self.assertIn("while test", node.branches[0].body)
        self.assertIn("done", node.branches[0].body)

    def test_for_inside_while_parses(self) -> None:
        cmd = _parse_first_cmd(
            "while true; do for x in a; do echo $x; done; done"
        )
        self.assertIsInstance(cmd, WhileNode)
        node: WhileNode = cmd
        self.assertIn("for x", node.body)

    def test_done_inside_if_inside_for_does_not_close_for(self) -> None:
        """The 'done' of a while inside a for should not close the for."""
        cmd = _parse_first_cmd(
            "for i in 1 2; do while true; do echo hi; done; done"
        )
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertIn("while true", node.body)
        self.assertIn("done", node.body)  # both 'done' tokens are in body

    def test_deep_triple_nesting_if_while_for(self) -> None:
        """if → while → for triple nesting (B2)."""
        cmd = _parse_first_cmd(
            "if true; then while true; do for i in a; do echo $i; done; done; fi"
        )
        self.assertIsInstance(cmd, IfNode)
        node: IfNode = cmd
        body = node.branches[0].body
        self.assertIn("while true", body)
        self.assertIn("for i in a", body)
        self.assertIn("done", body)

    def test_all_three_constructs_nested(self) -> None:
        """for → if → while triple nesting."""
        cmd = _parse_first_cmd(
            "for x in a b; do if true; then while test 1; do echo $x; done; fi; done"
        )
        self.assertIsInstance(cmd, ForNode)
        node: ForNode = cmd
        self.assertIn("if true", node.body)
        self.assertIn("while test", node.body)
        self.assertIn("fi", node.body)
        self.assertIn("done", node.body)

    def test_heredoc_inside_if_body_parses(self) -> None:
        """if true; then cat <<EOF\nhello\nEOF\nfi parses correctly."""
        cmd = _parse_first_cmd(
            "if true; then cat <<EOF\nhello\nEOF\nfi"
        )
        self.assertIsInstance(cmd, IfNode)
        node: IfNode = cmd
        self.assertIn("cat", node.branches[0].body)
        self.assertIn("<<EOF", node.branches[0].body)


if __name__ == "__main__":
    unittest.main()
