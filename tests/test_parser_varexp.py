"""Tests for $VAR / ${VAR} environment-variable expansion.

Validates that:
- Unquoted $VAR expands (one word, no field-split).
- Double-quoted "$VAR" expands.
- Braced "${VAR}" expands.
- Single-quoted '$VAR' is literal.
- Escaped \$VAR (unquoted and dq) is literal.
- Special params: $$ and $? are now special-variable lookups (PID, last exit code);
  $0, $1, $@ are positional parameters
  (Phase C) that resolve to empty when no positional params are supplied.
- Braced-default ${VAR:-x} expands (new parameter-expansion operators live in
  test_parser_param_exp.py; plain ${VAR} stays a straight env lookup).
- $$ resolves via special-variable lookup, $(( is still rejected.
- $ at EOL is literal.
- var-then-subst ordering: $A$(echo b).
- Adjacent text: a$HOMEb → a (POSIX: $HOMEb is var HOMEb, unset).
- Both STRING and AST paths agree on all cases where they can.
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
# helpers
# ---------------------------------------------------------------------------

ENV = {"HOME": "/root", "X": "a b"}


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

        # STRING path
        str_result = extract_redirects(cleaned, exp)

        # AST path
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


def _assert_str_path(
    test: unittest.TestCase,
    cmd: str,
    expected: list[str],
    env: Mapping[str, str] | None = ENV,
    outputs: dict[str, str] | None = None,
) -> None:
    """Assert only the STRING path (for cases where AST may diverge)."""
    str_r, _ast_r = _both_extract(cmd, env=env, outputs=outputs)
    str_args, _, str_err = str_r
    test.assertIsNone(str_err, f"String path error for {cmd!r}: {str_err}")
    test.assertEqual(str_args, expected, f"String path mismatch for {cmd!r}")


# ---------------------------------------------------------------------------
# AST helpers for opaque sentinel lookups
# ---------------------------------------------------------------------------

def _find_arg_sentinel(prog):
    """Return the first arg-sentinel WordPart in the first command, or None."""
    cmd = prog.chains[0].pipeline.commands[0]
    for w in cmd.words:
        for p in w.parts:
            if p.is_arg_sentinel:
                return p
    return None


def _any_arg_sentinel(prog) -> bool:
    """Return True if any arg-sentinel exists in the first command."""
    return _find_arg_sentinel(prog) is not None


# ---------------------------------------------------------------------------
# Basic expansion
# ---------------------------------------------------------------------------

class BasicVarExpansionTest(unittest.TestCase):
    """$VAR expands when env provides a value."""

    def test_unquoted_expands(self) -> None:
        _assert_both_equal(self, "echo $HOME", ["echo", "/root"])

    def test_dq_expands(self) -> None:
        _assert_both_equal(self, 'echo "$HOME"', ["echo", "/root"])

    def test_braced_expands(self) -> None:
        _assert_both_equal(self, 'echo "${HOME}"', ["echo", "/root"])

    def test_unquoted_braced_expands(self) -> None:
        _assert_both_equal(self, "echo ${HOME}", ["echo", "/root"])

    def test_sq_literal(self) -> None:
        _assert_both_equal(self, "echo '$HOME'", ["echo", "$HOME"])

    def test_escaped_dollar_literal(self) -> None:
        """echo \$HOME → literal $HOME (backslash stripped)."""
        _assert_both_equal(self, r"echo \$HOME", ["echo", "$HOME"])

    def test_dq_escaped_dollar_literal(self) -> None:
        r"""echo "\$HOME" → literal $HOME."""
        _assert_both_equal(self, r'echo "\$HOME"', ["echo", "$HOME"])


# ---------------------------------------------------------------------------
# Space value — with IFS field splitting (Phase E)
# ---------------------------------------------------------------------------

class FieldSplitTest(unittest.TestCase):
    """Unquoted $VAR with space-containing value is field-split by IFS."""

    def test_unquoted_space_value(self) -> None:
        """echo $X with X='a b' → ['echo','a','b'] (IFS field splitting)."""
        _assert_both_equal(self, "echo $X", ["echo", "a", "b"])

    def test_dq_space_value(self) -> None:
        _assert_both_equal(self, 'echo "$X"', ["echo", "a b"])

    def test_braced_space_value(self) -> None:
        _assert_both_equal(self, 'echo "${X}"', ["echo", "a b"])


# ---------------------------------------------------------------------------
# Unset variables
# ---------------------------------------------------------------------------

class UnsetVarTest(unittest.TestCase):
    """Vars not in env resolve to empty string."""

    def test_unset_unquoted(self) -> None:
        """Empty var → whole-word dropped (unified on AST path)."""
        _assert_both_equal(self, "echo $UNSET_VAR", ["echo"])

    def test_unset_in_word(self) -> None:
        """Empty var in the middle of a word."""
        # NO env passed → unset → resolves to ""
        str_r, ast_r = _both_extract("echo pre${UNSET_VAR}post", env={})
        str_args, _, str_err = str_r
        ast_args, _, ast_err = ast_r
        self.assertIsNone(str_err)
        self.assertIsNone(ast_err)
        # Both paths agree: the word is 'prepost' (empty var contributes nothing)
        self.assertEqual(str_args, ["echo", "prepost"])
        self.assertEqual(ast_args, ["echo", "prepost"])

    def test_two_unset_vars(self) -> None:
        """$A$B both unset → whole-word dropped (unified on AST path)."""
        _assert_both_equal(self, "echo $A$B", ["echo"])


# ---------------------------------------------------------------------------
# Adjacent text
# ---------------------------------------------------------------------------

class AdjacentTextTest(unittest.TestCase):
    """$VAR adjacent to other text in a word."""

    def test_var_prefix_text(self) -> None:
        """echo a$HOMEb → a (POSIX: $HOMEb is var HOMEb, unset→empty)."""
        _assert_both_equal(self, "echo a$HOMEb", ["echo", "a"])

    def test_var_suffix_text(self) -> None:
        """echo $HOME/x → /root/x."""
        _assert_both_equal(self, "echo $HOME/x", ["echo", "/root/x"])

    def test_braced_with_adjacent(self) -> None:
        """echo ${HOME}/x → /root/x."""
        _assert_both_equal(self, 'echo "${HOME}/x"', ["echo", "/root/x"])

    def test_braced_unquoted_with_adjacent(self) -> None:
        """echo ${HOME}/x → /root/x (unquoted)."""
        _assert_both_equal(self, "echo ${HOME}/x", ["echo", "/root/x"])


# ---------------------------------------------------------------------------
# Special parameters — always literal
# ---------------------------------------------------------------------------

class SpecialParamTest(unittest.TestCase):
    """$$ and $? stay literal.  $0, $1, $@ are now positional parameters
    (Phase C), resolving to empty when no positional params are provided."""

    def test_dollar_dollar_resolves(self) -> None:
        _assert_both_equal(self, "echo $$", ["echo", "11111"], env={"$": "11111", "?": "0"})

    def test_dollar_question_resolves(self) -> None:
        _assert_both_equal(self, "echo $?", ["echo", "0"], env={"$": "11111", "?": "0"})

    def test_dollar_zero_literal(self) -> None:
        # Phase C: $0 is a positional parameter, resolves to empty → dropped.
        _assert_both_equal(self, "echo $0", ["echo"])

    def test_dollar_one_literal(self) -> None:
        # Phase C: $1 is a positional parameter, resolves to empty → dropped.
        _assert_both_equal(self, "echo $1", ["echo"])

    def test_dollar_at_literal(self) -> None:
        # Phase C: $@ is a positional parameter, resolves to empty → dropped.
        _assert_both_equal(self, "echo $@", ["echo"])


# ---------------------------------------------------------------------------
# Braced-default forms — expanded (operators implemented; see param_exp tests)
# ---------------------------------------------------------------------------

class BracedDefaultExpansionTest(unittest.TestCase):
    """${VAR:-x} default operator is now expanded, not literal."""

    def test_braced_default_expands(self) -> None:
        _assert_both_equal(self, "echo ${VAR:-x}", ["echo", "x"])

    def test_braced_default_in_dq_expands(self) -> None:
        _assert_both_equal(self, 'echo "${VAR:-x}"', ["echo", "x"])


# ---------------------------------------------------------------------------
# Edge cases: $ at EOL, $((, var-then-subst
# ---------------------------------------------------------------------------

class EdgeCaseTest(unittest.TestCase):
    """Miscellaneous edge cases for $ handling."""

    def test_dollar_eol_literal(self) -> None:
        _assert_both_equal(self, "echo $", ["echo", "$"])

    def test_dollar_dollar_paren_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            _parse("echo $((1+1))")
        self.assertIn("Arithmetic", str(ctx.exception))

    def test_var_then_subst(self) -> None:
        """echo $A$(echo b) — var expands (empty if unset), sub expands (b)."""
        str_r, ast_r = _both_extract("echo $A$(echo b)", outputs={"echo b": "b"}, env={})
        str_args, _, str_err = str_r
        ast_args, _, ast_err = ast_r
        self.assertIsNone(str_err)
        self.assertIsNone(ast_err)
        # $A is unset → empty; $(echo b) → b; as one word → "b"
        self.assertEqual(str_args, ["echo", "b"])
        self.assertEqual(ast_args, ["echo", "b"])

    def test_var_then_subst_with_value(self) -> None:
        """echo $HOME$(echo b) → /rootb."""
        _assert_both_equal(self, "echo $HOME$(echo b)",
                           ["echo", "/rootb"],
                           outputs={"echo b": "b"})

    def test_subst_then_var(self) -> None:
        """echo $(echo a)$HOME → a/root."""
        _assert_both_equal(self, "echo $(echo a)$HOME",
                           ["echo", "a/root"],
                           outputs={"echo a": "a"})

    def test_dq_var_then_subst_single_arg(self) -> None:
        r"""echo "$HOME$(echo b)" → /rootb (single arg)."""
        _assert_both_equal(self, 'echo "$HOME$(echo b)"',
                           ["echo", "/rootb"],
                           outputs={"echo b": "b"})

    def test_multi_var_one_word(self) -> None:
        """echo $HOME$USER — two vars concatenated into one word."""
        env2 = {"HOME": "/root", "USER": "arch"}
        _assert_both_equal(self, "echo $HOME$USER", ["echo", "/rootarch"], env=env2)


# ---------------------------------------------------------------------------
# Double-quoted specific cases
# ---------------------------------------------------------------------------

class DoubleQuotedVarTest(unittest.TestCase):
    """Variable expansion inside double quotes — parity checks."""

    def test_dq_literal_text_around_var(self) -> None:
        _assert_both_equal(self, 'echo "home=$HOME"', ["echo", "home=/root"])

    def test_dq_multiple_vars(self) -> None:
        env2 = {"A": "hello", "B": "world"}
        _assert_both_equal(self, 'echo "$A $B"', ["echo", "hello world"], env=env2)

    def test_dq_braced_var_adjacent(self) -> None:
        _assert_both_equal(self, 'echo "${HOME}dir"', ["echo", "/rootdir"])


# ---------------------------------------------------------------------------
# Scanner-level checks (expansion table populated)
# ---------------------------------------------------------------------------

class ScannerExpansionTest(unittest.TestCase):
    """Check that the expansion table is correctly populated."""

    def test_expansion_table_has_value(self) -> None:
        cleaned, exp, prog = _parse("echo $HOME")
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part, "Should have a sentinel")
        self.assertEqual(exp.arg_for(part), "/root")

    def test_braced_expansion_table(self) -> None:
        cleaned, exp, prog = _parse("echo ${HOME}")
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part, "Should have a sentinel for ${HOME}")
        self.assertEqual(exp.arg_for(part), "/root")

    def test_unset_not_in_table(self) -> None:
        """Unset var should produce sentinel with empty string value."""
        cleaned, exp, prog = _parse("echo $UNSET_VAR", env={})
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.arg_for(part), "")

    def test_escaped_dollar_no_sentinel(self) -> None:
        """\$VAR should NOT produce a sentinel."""
        cleaned, exp, prog = _parse(r"echo \$HOME")
        self.assertFalse(_any_arg_sentinel(prog))
        self.assertIn("$HOME", cleaned)

    def test_sq_no_sentinel(self) -> None:
        """'$HOME' should NOT produce a sentinel."""
        cleaned, exp, prog = _parse("echo '$HOME'")
        self.assertFalse(_any_arg_sentinel(prog))
        self.assertIn("$HOME", cleaned)

    def test_special_param_no_sentinel(self) -> None:
        """$? / $$ / $! / $- now produce sentinels (special-variable lookup)."""
        cleaned, exp, prog = _parse("echo $$")
        # $$ is now a VARREF — it DOES produce a sentinel
        self.assertTrue(_any_arg_sentinel(prog))


# ---------------------------------------------------------------------------
# Env allowlist integration — expansion uses allowlisted env, not host
# ---------------------------------------------------------------------------

class EnvAllowlistIntegrationTest(unittest.TestCase):
    """$VAR expansion uses only the allowlist env, not os.environ."""

    def test_expansion_ignores_os_environ(self) -> None:
        """Pass a custom env; assert os.environ values are NOT used."""
        import os
        cleaned, exp, prog = _parse("echo $HOME", env={"HOME": "/custom"})
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.arg_for(part), "/custom")
        # If os.environ leaked, the value would be the real HOME, not /custom
        if "HOME" in os.environ:
            self.assertNotEqual(exp.arg_for(part) or "", os.environ["HOME"])

    def test_no_env_all_vars_empty(self) -> None:
        """With empty env, all $VAR become sentinels with '' values."""
        cleaned, exp, prog = _parse("echo $HOME", env={})
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part, "Even unset vars produce a sentinel")
        self.assertEqual(exp.arg_for(part), "")


if __name__ == "__main__":
    unittest.main()
