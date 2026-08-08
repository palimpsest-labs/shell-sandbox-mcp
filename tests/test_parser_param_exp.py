"""Tests for POSIX parameter-expansion operators in ${...}.

Operators covered:
- ``${VAR:-default}``, ``${VAR:=default}`` (same as ``:-``; env is read-only),
  ``${VAR:?msg}`` (error), ``${VAR:+alt}`` (alternate).
- ``${#VAR}`` string length.
- ``${VAR#pat}`` / ``${VAR##pat}`` / ``${VAR%pat}`` / ``${VAR%%pat}``
  prefix/suffix glob removal (fnmatch-style, shortest/longest).
- ``${VAR:offset[:len]}`` substring (non-negative offsets, clamped).
- ``${VAR,,}`` / ``${VAR,}`` / ``${VAR^^}`` / ``${VAR^}`` case modification.
- Nested expansion ($VAR / ${...} / $(...) inside operands).
- Quote interplay (dq expands, sq literal, ``#``/``}`` inside quoted operand).
- Unknown-op literal fallback; special params stay literal; unbalanced ``${``.

Both STRING and AST paths must agree (via ``_assert_both_equal``).
"""

import tempfile
import unittest
from pathlib import Path
from typing import Optional, Mapping

from shell_sandbox_mcp.parser import (
    Expansion,
    ParseError,
    Redirect,
    extract_redirects,
    parse_command,
    program_to_chain,
)


# ---------------------------------------------------------------------------
# helpers (mirrors test_parser_varexp.py)
# ---------------------------------------------------------------------------

ENV = {
    "HOME": "/root",
    "X": "a b",
    "PATH": "/bin:/usr/bin",
    "EMPTY": "",
    "S": "Hello World",
    "F": "fooXfooY",
    "FILE": "archive.tar.gz",
}
# UNSET_* variables are simply not present in ENV.


def _stub_capture(outputs: dict[str, str] | None = None):
    """Return a capture_fn that maps inner command text -> (rc, stdout_bytes)."""
    outputs = outputs or {}
    def fake_capture(inner: str):
        val = outputs.get(inner, "")
        return 0, val.encode("utf-8")
    return fake_capture


def _parse(cmd: str, env: Mapping[str, str] | None = ENV,
           outputs: dict[str, str] | None = None):
    """Shortcut for parse_command."""
    cap = _stub_capture(outputs)
    with tempfile.TemporaryDirectory() as td:
        return parse_command(cmd, cap, Path(td), 30, 0, env=env)


def _both_extract(
    cmd: str,
    env: Mapping[str, str] | None = ENV,
    outputs: dict[str, str] | None = None,
) -> tuple[
    tuple[list[str], list[Redirect], Optional[str]],
    tuple[list[str], list[Redirect], Optional[str]],
]:
    """Run extract_redirects via both STRING and AST paths."""
    cap = _stub_capture(outputs)
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        try:
            cleaned, exp, prog = parse_command(cmd, cap, wd, 30, 0, env=env)
        except (ValueError, ParseError) as exc:
            str_result = extract_redirects(cmd, None)
            return str_result, ([], [], str(exc))

        if prog is None:
            str_result = extract_redirects(cleaned, exp)
            return str_result, ([], [], None)

        str_result = extract_redirects(cleaned, exp)

        chain = program_to_chain(prog)
        if chain and chain[0][1]:
            cmd_node = chain[0][1][0]
            ast_result = extract_redirects(cmd_node, exp)
        else:
            ast_result = ([], [], None)

    return str_result, ast_result


def _assert_both_equal(
    test: unittest.TestCase,
    cmd: str,
    expected: list[str],
    env: Mapping[str, str] | None = ENV,
    outputs: dict[str, str] | None = None,
) -> None:
    """Assert both paths produce *expected* and agree."""
    str_r, ast_r = _both_extract(cmd, env=env, outputs=outputs)
    str_args, _, str_err = str_r
    ast_args, _, ast_err = ast_r

    test.assertIsNone(str_err, f"String path error for {cmd!r}: {str_err}")
    test.assertIsNone(ast_err, f"AST path error for {cmd!r}: {ast_err}")
    test.assertEqual(str_args, expected, f"String path mismatch for {cmd!r}")
    test.assertEqual(ast_args, expected, f"AST path mismatch for {cmd!r}")
    test.assertEqual(str_args, ast_args,
                     f"Path divergence for {cmd!r}: STRING={str_args}, AST={ast_args}")


def _find_arg_sentinel(prog):
    """Return the first arg-sentinel WordPart in the first command, or None."""
    cmd = prog.chains[0].pipeline.commands[0]
    for w in cmd.words:
        for p in w.parts:
            if p.is_arg_sentinel:
                return p
    return None


# ---------------------------------------------------------------------------
# Default / assign / alternate / error
# ---------------------------------------------------------------------------

class DefaultOperatorTest(unittest.TestCase):
    def test_default_unset(self) -> None:
        _assert_both_equal(self, "echo ${UNSET_VAR:-x}", ["echo", "x"])

    def test_default_set(self) -> None:
        _assert_both_equal(self, "echo ${HOME:-x}", ["echo", "/root"])

    def test_default_empty(self) -> None:
        _assert_both_equal(self, "echo ${EMPTY:-x}", ["echo", "x"])

    def test_assign_same_as_default(self) -> None:
        # ${VAR:=x} is identical to ${VAR:-x} here (env is read-only per call).
        _assert_both_equal(self, "echo ${UNSET_VAR:=x}", ["echo", "x"])
        _assert_both_equal(self, "echo ${HOME:=x}", ["echo", "/root"])

    def test_alternate_set(self) -> None:
        _assert_both_equal(self, "echo ${HOME:+alt}", ["echo", "alt"])

    def test_alternate_empty(self) -> None:
        _assert_both_equal(self, "echo ${EMPTY:+alt}", ["echo"])

    def test_alternate_unset_in_word(self) -> None:
        _assert_both_equal(self, "echo a${UNSET_VAR:+alt}b", ["echo", "ab"])

    def test_error_set_no_error(self) -> None:
        _assert_both_equal(self, "echo ${HOME:?msg}", ["echo", "/root"])

    def test_error_unset_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _parse("echo ${UNSET_VAR:?my message}")
        self.assertIn("my message", str(ctx.exception))

    def test_error_unset_default_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _parse("echo ${UNSET_VAR:?}")
        self.assertIn("parameter not set or null", str(ctx.exception))


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------

class LengthOperatorTest(unittest.TestCase):
    def test_length(self) -> None:
        _assert_both_equal(self, "echo ${#HOME}", ["echo", "5"])

    def test_length_unset(self) -> None:
        _assert_both_equal(self, "echo ${#UNSET_VAR}", ["echo", "0"])


# ---------------------------------------------------------------------------
# Prefix / suffix removal
# ---------------------------------------------------------------------------

class PrefixRemovalTest(unittest.TestCase):
    def test_shortest_prefix(self) -> None:
        # #*/ removes up to the first '/' → "bin:/usr/bin"
        _assert_both_equal(self, "echo ${PATH#*/}", ["echo", "bin:/usr/bin"])

    def test_longest_prefix(self) -> None:
        # ##*/ removes up to the last '/' → "bin"
        _assert_both_equal(self, "echo ${PATH##*/}", ["echo", "bin"])

    def test_shortest_prefix_glob(self) -> None:
        # #*foo removes the shortest prefix matching *foo
        _assert_both_equal(self, "echo ${F#*foo}", ["echo", "XfooY"])

    def test_longest_prefix_glob(self) -> None:
        # ##*foo removes the longest prefix matching *foo
        _assert_both_equal(self, "echo ${F##*foo}", ["echo", "Y"])

    def test_no_match_unchanged(self) -> None:
        _assert_both_equal(self, "echo ${HOME#zzz}", ["echo", "/root"])


class SuffixRemovalTest(unittest.TestCase):
    def test_shortest_suffix(self) -> None:
        # %.% removes the shortest suffix (last dot group) → "archive.tar"
        _assert_both_equal(self, "echo ${FILE%.*}", ["echo", "archive.tar"])

    def test_longest_suffix(self) -> None:
        # %%.% removes the longest suffix → "archive"
        _assert_both_equal(self, "echo ${FILE%%.*}", ["echo", "archive"])

    def test_no_match_unchanged(self) -> None:
        _assert_both_equal(self, "echo ${HOME%zzz}", ["echo", "/root"])

    def test_empty_pattern(self) -> None:
        _assert_both_equal(self, "echo ${HOME#}", ["echo", "/root"])
        _assert_both_equal(self, "echo ${HOME%}", ["echo", "/root"])


# ---------------------------------------------------------------------------
# Substring
# ---------------------------------------------------------------------------

class SubstringTest(unittest.TestCase):
    def test_offset_len(self) -> None:
        _assert_both_equal(self, "echo ${PATH:0:4}", ["echo", "/bin"])

    def test_offset_only(self) -> None:
        _assert_both_equal(self, "echo ${PATH:5}", ["echo", "/usr/bin"])

    def test_len_clamped(self) -> None:
        _assert_both_equal(self, "echo ${PATH:0:100}", ["echo", "/bin:/usr/bin"])

    def test_offset_beyond_length(self) -> None:
        # offset beyond length → empty string → word dropped
        _assert_both_equal(self, "echo ${PATH:100}", ["echo"])

    def test_offset_len_in_word(self) -> None:
        _assert_both_equal(self, "echo a${PATH:0:4}b", ["echo", "a/binb"])

    def test_len_zero(self) -> None:
        _assert_both_equal(self, "echo ${PATH:2:0}", ["echo"])


# ---------------------------------------------------------------------------
# Case modification
# ---------------------------------------------------------------------------

class CaseModTest(unittest.TestCase):
    def test_lower_all(self) -> None:
        _assert_both_equal(self, "echo ${S,,}", ["echo", "hello", "world"])

    def test_lower_first(self) -> None:
        _assert_both_equal(self, "echo ${S,}", ["echo", "hello", "World"])

    def test_upper_all(self) -> None:
        _assert_both_equal(self, "echo ${X^^}", ["echo", "A", "B"])

    def test_upper_first(self) -> None:
        _assert_both_equal(self, "echo ${X^}", ["echo", "A", "b"])

    def test_case_empty(self) -> None:
        _assert_both_equal(self, "echo ${UNSET_VAR^^}", ["echo"])


# ---------------------------------------------------------------------------
# Nested expansion inside operands
# ---------------------------------------------------------------------------

class NestedExpansionTest(unittest.TestCase):
    def test_nested_braced_default(self) -> None:
        _assert_both_equal(self, "echo ${UNSET_VAR:-${HOME}}", ["echo", "/root"])

    def test_nested_bare_var_default(self) -> None:
        _assert_both_equal(self, "echo ${UNSET_VAR:-$HOME}", ["echo", "/root"])

    def test_nested_subst_default(self) -> None:
        _assert_both_equal(
            self, "echo ${UNSET_VAR:-$(echo hi)}",
            ["echo", "hi"], outputs={"echo hi": "hi"})

    def test_nested_alternate(self) -> None:
        # Unquoted nested ${X} → field-split by IFS.
        _assert_both_equal(self, "echo ${HOME:+${X}}", ["echo", "a", "b"])

    def test_nested_subst_alternate(self) -> None:
        _assert_both_equal(
            self, "echo ${HOME:+$(echo alt)}",
            ["echo", "alt"], outputs={"echo alt": "alt"})

    def test_nested_in_pattern(self) -> None:
        # pattern itself expands $F → "fooXfooY"; ##*FOO (uppercase) won't match
        _assert_both_equal(self, "echo ${UNSET_VAR:-$F}", ["echo", "fooXfooY"])

    def test_escaped_dollar_in_operand(self) -> None:
        # \$-escaped $ in an operand is literal (backslash stripped)
        _assert_both_equal(self, r"echo ${UNSET_VAR:-\$HOME}", ["echo", "$HOME"])

    def test_deep_nested_raises_valueerror_not_recursion(self) -> None:
        # A deep nested ${...} chain (beyond MAX_SUBST_DEPTH) must raise a
        # clean ValueError via the depth guard, NOT Python's RecursionError.
        # UNSET_VAR is unset so the :- operand (the nested chain) is evaluated.
        depth = 1000
        cmd = "echo ${UNSET_VAR:-" + "${Y:-" * depth + "z" + "}" * depth + "}"
        with self.assertRaises(ValueError) as ctx:
            _parse(cmd)
        self.assertIn("Parameter expansion depth limit", str(ctx.exception))


# ---------------------------------------------------------------------------
# Quote interplay
# ---------------------------------------------------------------------------

class QuoteInterplayTest(unittest.TestCase):
    def test_dq_expands(self) -> None:
        _assert_both_equal(self, 'echo "${UNSET_VAR:-x}"', ["echo", "x"])

    def test_sq_literal(self) -> None:
        _assert_both_equal(self, "echo '${UNSET_VAR:-x}'", ["echo", "${UNSET_VAR:-x}"])

    def test_dq_plain_braced(self) -> None:
        _assert_both_equal(self, 'echo "${HOME}"', ["echo", "/root"])

    def test_hash_in_quoted_operand(self) -> None:
        _assert_both_equal(self, 'echo ${UNSET_VAR:-"a#b"}', ["echo", "a#b"])

    def test_brace_in_quoted_operand(self) -> None:
        _assert_both_equal(self, 'echo ${UNSET_VAR:-"a}b"}', ["echo", "a}b"])

    def test_brace_in_sq_operand(self) -> None:
        _assert_both_equal(self, "echo ${UNSET_VAR:-a'b}c'}", ["echo", "ab}c"])


# ---------------------------------------------------------------------------
# Unknown-operator literal fallback + special params
# ---------------------------------------------------------------------------

class LiteralFallbackTest(unittest.TestCase):
    def test_unknown_colon_op_literal(self) -> None:
        # Non-numeric offset → not a substring → unknown operator → literal
        _assert_both_equal(self, "echo ${HOME:abc}", ["echo", "${HOME:abc}"])

    def test_unknown_case_pattern_literal(self) -> None:
        _assert_both_equal(self, "echo ${HOME^foo}", ["echo", "${HOME^foo}"])

    def test_dollar_dollar_resolves(self) -> None:
        _assert_both_equal(self, "echo $$", ["echo", "11111"], env={**ENV, "$": "11111", "?": "0"})

    def test_dollar_question_resolves(self) -> None:
        _assert_both_equal(self, "echo $?", ["echo", "0"], env={**ENV, "$": "11111", "?": "0"})

    def test_dollar_zero_literal(self) -> None:
        # Phase C: $0 is now a positional parameter (not literal).
        # With no positional params provided, it resolves to empty → word dropped.
        _assert_both_equal(self, "echo $0", ["echo"])

    def test_braced_number_literal(self) -> None:
        # Phase C: ${10} is now a positional parameter (braced multi-digit).
        # With no positional params provided, it resolves to empty → word dropped.
        _assert_both_equal(self, "echo ${10}", ["echo"])

    def test_empty_braces_literal(self) -> None:
        # ${} has no valid parameter name — stays literal.
        _assert_both_equal(self, "echo ${}", ["echo", "${}"])

    def test_unknown_literal_in_word(self) -> None:
        _assert_both_equal(self, "echo a${HOME:abc}b", ["echo", "a${HOME:abc}b"])


# ---------------------------------------------------------------------------
# Unbalanced braces → ParseError
# ---------------------------------------------------------------------------

class UnbalancedBracesTest(unittest.TestCase):
    def test_unbalanced_raises(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            _parse("echo ${HOME")
        self.assertIn("Unbalanced", str(ctx.exception))


# ---------------------------------------------------------------------------
# Scanner-level checks (expansion table populated)
# ---------------------------------------------------------------------------

class ScannerExpansionTest(unittest.TestCase):
    def test_default_in_table(self) -> None:
        cleaned, exp, prog = _parse("echo ${UNSET_VAR:-x}")
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.arg_for(part), "x")

    def test_length_in_table(self) -> None:
        cleaned, exp, prog = _parse("echo ${#HOME}")
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.arg_for(part), "5")

    def test_sq_no_sentinel(self) -> None:
        cleaned, exp, prog = _parse("echo '${UNSET_VAR:-x}'")
        self.assertFalse(any(p.is_arg_sentinel
                             for w in prog.chains[0].pipeline.commands[0].words
                             for p in w.parts))
        self.assertIn("${UNSET_VAR:-x}", cleaned)


if __name__ == "__main__":
    unittest.main()
