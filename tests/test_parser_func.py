"""Tests for function definition and brace group parsing (Phase C).

Run with::

    PYTHONPATH=src python3 -m pytest tests/test_parser_func.py -q
"""

import unittest

from shell_sandbox_mcp.parser import (
    CaseClause,
    CaseNode,
    CommandNode,
    ForNode,
    FuncNode,
    GroupNode,
    IfBranch,
    IfNode,
    ParseError,
    SubshellNode,
    WhileNode,
    _build_ast,
    Expansion,
    TokenKind,
)
from shell_sandbox_mcp.parser import Lexer


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


def _tokenize(command: str):
    """Return the list of tokens for *command*."""
    return Lexer(command).tokenize()


# ---------------------------------------------------------------------------
# Function definition parsing
# ---------------------------------------------------------------------------


class FuncParsingTest(unittest.TestCase):
    """Test that the parser correctly produces FuncNode ASTs."""

    def test_posix_func_simple(self) -> None:
        cmd = _parse_first_cmd("f() echo hi")
        self.assertIsInstance(cmd, FuncNode)
        node: FuncNode = cmd
        self.assertEqual(node.name, "f")
        self.assertIn("echo", node.body)

    def test_posix_func_with_semicolon(self) -> None:
        cmd = _parse_first_cmd("f() echo hi; echo there")
        self.assertIsInstance(cmd, FuncNode)
        node: FuncNode = cmd
        self.assertEqual(node.name, "f")
        self.assertIn("echo hi", node.body)
        self.assertNotIn("echo there", node.body)  # body stops at ;

    def test_keyword_func_simple(self) -> None:
        cmd = _parse_first_cmd("function f echo hi")
        self.assertIsInstance(cmd, FuncNode)
        node: FuncNode = cmd
        self.assertEqual(node.name, "f")
        self.assertIn("echo", node.body)

    def test_keyword_func_with_parens(self) -> None:
        cmd = _parse_first_cmd("function f() echo hi")
        self.assertIsInstance(cmd, FuncNode)
        node: FuncNode = cmd
        self.assertEqual(node.name, "f")
        self.assertIn("echo", node.body)

    def test_func_group_body(self) -> None:
        cmd = _parse_first_cmd("f() { echo hi; }")
        self.assertIsInstance(cmd, FuncNode)
        node: FuncNode = cmd
        self.assertEqual(node.name, "f")
        self.assertIn("echo hi", node.body)

    def test_func_compound_body_if(self) -> None:
        cmd = _parse_first_cmd("f() if true; then echo x; fi")
        self.assertIsInstance(cmd, FuncNode)
        node: FuncNode = cmd
        self.assertEqual(node.name, "f")
        self.assertIn("if true", node.body)
        self.assertIn("echo x", node.body)

    def test_func_body_terminated_by_newline(self) -> None:
        cmd = _parse_first_cmd("f() echo hi\necho there")
        self.assertIsInstance(cmd, FuncNode)
        node: FuncNode = cmd
        self.assertIn("echo hi", node.body)
        self.assertNotIn("echo there", node.body)

    def test_func_body_terminated_by_and_and(self) -> None:
        cmd = _parse_first_cmd("f() echo hi && echo there")
        self.assertIsInstance(cmd, FuncNode)
        node: FuncNode = cmd
        self.assertIn("echo hi", node.body)
        self.assertNotIn("echo there", node.body)

    # ---- invalid names ----

    def test_digit_leading_name_is_not_func(self) -> None:
        """1f() — digit-leading names are not valid identifiers."""
        cmd = _parse_first_cmd("1f() echo hi")
        # Should parse as a regular command, not a function definition
        self.assertIsInstance(cmd, CommandNode)

    def test_reserved_name_is_error(self) -> None:
        """if() is a reserved word — should raise ParseError."""
        with self.assertRaises(ParseError):
            _parse_first_cmd("if() echo hi")

    def test_reserved_while_name(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("while() echo hi")

    # ---- missing body ----

    def test_func_missing_body(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("f()")

    def test_func_missing_body_only_ws(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("f()   ")

    # ---- not a function def ----

    def test_f_x_is_not_func_def(self) -> None:
        """f x is NOT a function definition (no parens)."""
        cmd = _parse_first_cmd("f x")
        self.assertIsInstance(cmd, CommandNode)

    def test_echo_f_parens_is_command(self) -> None:
        """echo f() — f() is an argument, not a function def."""
        cmd = _parse_first_cmd("echo f() hi")
        self.assertIsInstance(cmd, CommandNode)


# ---------------------------------------------------------------------------
# Positional parameter lexing
# ---------------------------------------------------------------------------


class PositionalLexingTest(unittest.TestCase):
    """Test that $1/$#/$@/$* lex as VARREF."""

    def _find_kinds(self, tokens, kind):
        return [t for t in tokens if t.kind == kind]

    def test_dollar1_is_varref(self) -> None:
        tokens = _tokenize("echo $1")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF]
        self.assertTrue(any(t.value == "1" for t in varrefs))

    def test_dollar_hash_is_varref(self) -> None:
        tokens = _tokenize("echo $#")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF]
        self.assertTrue(any(t.value == "#" for t in varrefs))

    def test_dollar_at_is_varref(self) -> None:
        tokens = _tokenize("echo $@")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF]
        self.assertTrue(any(t.value == "@" for t in varrefs))

    def test_dollar_star_is_varref(self) -> None:
        tokens = _tokenize("echo $*")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF]
        self.assertTrue(any(t.value == "*" for t in varrefs))

    def test_dollar0_is_varref(self) -> None:
        tokens = _tokenize("echo $0")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF]
        self.assertTrue(any(t.value == "0" for t in varrefs))

    def test_braced_dollar10(self) -> None:
        tokens = _tokenize("echo ${10}")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF]
        self.assertTrue(any(t.value == "10" for t in varrefs))

    def test_braced_dollar1_default(self) -> None:
        tokens = _tokenize("echo ${1:-default}")
        varrefs = [t for t in tokens if t.kind == TokenKind.VARREF]
        self.assertTrue(any("1:-default" in t.value for t in varrefs))


# ---------------------------------------------------------------------------
# Brace group parsing
# ---------------------------------------------------------------------------


class GroupParsingTest(unittest.TestCase):
    """Test that ``{ command; }`` parses as GroupNode."""

    def test_simple_group(self) -> None:
        cmd = _parse_first_cmd("{ echo hi; }")
        self.assertIsInstance(cmd, GroupNode)
        node: GroupNode = cmd
        self.assertIn("echo hi", node.body)

    def test_group_multiple_commands(self) -> None:
        cmd = _parse_first_cmd("{ echo a; echo b; }")
        self.assertIsInstance(cmd, GroupNode)
        node: GroupNode = cmd
        self.assertIn("echo a", node.body)
        self.assertIn("echo b", node.body)

    def test_group_nested(self) -> None:
        cmd = _parse_first_cmd("{ { echo hi; }; }")
        self.assertIsInstance(cmd, GroupNode)

    def test_group_missing_close(self) -> None:
        with self.assertRaises(ParseError):
            _parse_first_cmd("{ echo hi")


# ---------------------------------------------------------------------------
# FUNC_PARENS token emission
# ---------------------------------------------------------------------------


class FuncParensTokenTest(unittest.TestCase):
    """Test that FUNC_PARENS tokens are emitted correctly."""

    def test_func_parens_emitted(self) -> None:
        tokens = _tokenize("f() echo hi")
        kinds = [t.kind for t in tokens]
        self.assertIn(TokenKind.FUNC_PARENS, kinds)

    def test_func_parens_position(self) -> None:
        tokens = _tokenize("f() echo hi")
        func_parens_tokens = [t for t in tokens if t.kind == TokenKind.FUNC_PARENS]
        self.assertEqual(len(func_parens_tokens), 1)
        self.assertEqual(func_parens_tokens[0].value, "()")

    def test_keyword_func_no_parens(self) -> None:
        """function f echo — no FUNC_PARENS."""
        tokens = _tokenize("function f echo hi")
        kinds = [t.kind for t in tokens]
        self.assertNotIn(TokenKind.FUNC_PARENS, kinds)

    def test_keyword_func_with_parens(self) -> None:
        """function f() echo — has FUNC_PARENS."""
        tokens = _tokenize("function f() echo hi")
        kinds = [t.kind for t in tokens]
        self.assertIn(TokenKind.FUNC_PARENS, kinds)
