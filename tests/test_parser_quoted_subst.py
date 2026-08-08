"""Tests for POSIX-style $() expansion inside double quotes.

Validates that:
- ``$(...)`` expands inside ``"..."`` (new behaviour).
- ``'...'`` stays fully literal (no expansion).
- Backslash-escaped ``$`` prevents expansion inside double quotes.
- ``$((`` is rejected inside double quotes (like unquoted).
- ``$VAR`` / ``${VAR}`` stay literal everywhere.
- Scanner path and AST path produce identical results (differential parity).
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

def _stub_capture(outputs: dict[str, str]):
    """Return a capture_fn that maps inner command text -> (rc, stdout_bytes)."""
    def fake_capture(inner: str):
        val = outputs.get(inner, "")
        return 0, val.encode("utf-8")
    return fake_capture


def _parse(cmd: str, outputs: dict[str, str] | None = None, env: Mapping[str, str] | None = None):
    """Shortcut for parse_command with a stub capture and optional *env*."""
    outputs = outputs or {}
    capture_fn = _stub_capture(outputs)
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        return parse_command(cmd, capture_fn, wd, 30, 0, env=env)


# AST helpers
def _find_arg_sentinel(prog):
    cmd = prog.chains[0].pipeline.commands[0]
    for w in cmd.words:
        for p in w.parts:
            if p.is_arg_sentinel:
                return p
    return None

def _find_hd_sentinel(prog):
    cmd = prog.chains[0].pipeline.commands[0]
    for rs in cmd.redirects:
        for p in rs.target.parts:
            if p.is_hd_sentinel:
                return p
    return None

def _all_arg_sentinels(prog):
    cmd = prog.chains[0].pipeline.commands[0]
    result = []
    for w in cmd.words:
        for p in w.parts:
            if p.is_arg_sentinel:
                result.append(p)
    return result

def _any_arg_sentinel(prog) -> bool:
    return _find_arg_sentinel(prog) is not None


# ---------------------------------------------------------------------------
# Basic double-quoted $() expansion (scanner path)
# ---------------------------------------------------------------------------

class QuotedSubstScannerTest(unittest.TestCase):
    """Test the char-by-char scanner path (parse_command) for double-quoted $()."""

    def test_dq_subst_simple(self) -> None:
        """echo "$(echo inner)" -> inner (expands inside double quotes)."""
        cleaned, exp, prog = _parse('echo "$(echo inner)"', {"echo inner": "inner"})
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part, "Should have a sentinel in the cleaned string")
        self.assertEqual(exp.arg_for(part), "inner")

    def test_dq_subst_multi_word_single_arg(self) -> None:
        """echo "$(echo a b c)" -> single arg 'a b c', NO field splitting."""
        cleaned, exp, prog = _parse(
            'echo "$(echo a b c)"', {"echo a b c": "a b c"}
        )
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part)
        # The value must be stored as ONE string with spaces -- not split into args
        self.assertEqual(exp.arg_for(part), "a b c")

    def test_sq_stays_literal(self) -> None:
        """echo '$(echo inner)' -> literal $(echo inner) (single quotes literal)."""
        cleaned, exp, prog = _parse(
            "echo '$(echo inner)'", {"echo inner": "SHOULD_NOT_APPEAR"}
        )
        # No sentinel should be in the AST
        self.assertFalse(_any_arg_sentinel(prog))
        # The literal $(echo inner) must be present in cleaned
        self.assertIn("$(echo inner)", cleaned)

    def test_escaped_dollar_paren_literal(self) -> None:
        r"""echo "\$(echo x)" -> literal $(echo x), NOT expanded."""
        cleaned, exp, prog = _parse(
            r'echo "\$(echo x)"', {"echo x": "SHOULD_NOT_APPEAR"}
        )
        self.assertFalse(_any_arg_sentinel(prog))
        self.assertIn("$(echo x)", cleaned)

    def test_compound_dq_subst(self) -> None:
        r"""echo "pre$(echo mid)post" -> premidpost as single arg."""
        cleaned, exp, prog = _parse(
            r'echo "pre$(echo mid)post"', {"echo mid": "mid"}
        )
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.arg_for(part), "mid")
        # The cleaned string should have the sentinel between "pre and post"
        self.assertIn('"pre', cleaned)
        self.assertIn('post"', cleaned)

    def test_nested_dq_subst(self) -> None:
        r"""echo "$(echo "$(echo deep)")" -- nested expansion inside double quotes."""
        captured: list[str] = []

        def fake_capture(inner: str):
            captured.append(inner)
            return 0, inner.encode()

        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            cleaned, exp, prog = parse_command(
                'echo "$(echo "$(echo deep)")"', fake_capture, wd, 30, 0,
            )
        # The outer capture should get: echo "$(echo deep)"
        # The inner capture should get: echo deep
        self.assertIn('echo "$(echo deep)"', captured)

    def test_dq_multiple_subst(self) -> None:
        r"""echo "$(echo a)$(echo b)" -> two sentinels, both resolved."""
        cleaned, exp, prog = _parse(
            r'echo "$(echo a)$(echo b)"',
            {"echo a": "alpha", "echo b": "beta"},
        )
        parts = _all_arg_sentinels(prog)
        self.assertEqual(len(parts), 2, f"Expected 2 sentinels, got {parts}")
        self.assertEqual(exp.arg_for(parts[0]), "alpha")
        self.assertEqual(exp.arg_for(parts[1]), "beta")


# ---------------------------------------------------------------------------
# $(( arithmetic rejection inside double quotes
# ---------------------------------------------------------------------------

class QuotedArithmeticRejectionTest(unittest.TestCase):
    """$(( inside double quotes must be rejected; single-quoted stays literal."""

    def test_dq_double_paren_rejected(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            _parse('echo "$((1+1))"')
        self.assertIn("Arithmetic", str(ctx.exception))

    def test_sq_double_paren_literal(self) -> None:
        cleaned, exp, prog = _parse("echo '$((1+1))'")
        self.assertIn("$((1+1))", cleaned)
        self.assertFalse(_any_arg_sentinel(prog))


# ---------------------------------------------------------------------------
# $VAR / ${VAR} expansion (with and without env)
# ---------------------------------------------------------------------------

class VariableExpansionTest(unittest.TestCase):
    """$VAR and ${VAR} expand when env is provided; empty-string when not."""

    def test_unquoted_var_expands(self) -> None:
        """echo $HOME with env expands to /root (sentinel, not literal)."""
        cleaned, exp, prog = _parse("echo $HOME", env={"HOME": "/root"})
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part, "Should have a sentinel for $HOME")
        self.assertEqual(exp.arg_for(part), "/root")

    def test_dq_var_expands(self) -> None:
        """echo "$HOME" with env expands to /root."""
        cleaned, exp, prog = _parse('echo "$HOME"', env={"HOME": "/root"})
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part, "Should have a sentinel for $HOME in dq")
        self.assertEqual(exp.arg_for(part), "/root")

    def test_dq_braced_var_expands(self) -> None:
        """echo "${HOME}" with env expands to /root."""
        cleaned, exp, prog = _parse('echo "${HOME}"', env={"HOME": "/root"})
        part = _find_arg_sentinel(prog)
        self.assertIsNotNone(part, "Should have a sentinel for ${HOME} in dq")
        self.assertEqual(exp.arg_for(part), "/root")

    def test_sq_var_literal(self) -> None:
        cleaned, exp, prog = _parse("echo '$HOME'")
        self.assertIn("$HOME", cleaned)


# ---------------------------------------------------------------------------
# Heredoc with quoted delimiter -- body stays literal (unchanged)
# ---------------------------------------------------------------------------

class HeredocQuotedDelimLiteralTest(unittest.TestCase):
    """Heredocs with quoted delimiters must keep $(...) literal in body."""

    def test_dq_delim_body_literal(self) -> None:
        cmd = 'cat <<"EOF"\n$(echo hi)\nEOF'
        cleaned, exp, prog = _parse(cmd, {"echo hi": "SHOULD_NOT_APPEAR"})
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "$(echo hi)\n")

    def test_sq_delim_body_literal(self) -> None:
        cmd = "cat <<'EOF'\n$(echo hi)\nEOF"
        cleaned, exp, prog = _parse(cmd, {"echo hi": "SHOULD_NOT_APPEAR"})
        part = _find_hd_sentinel(prog)
        self.assertIsNotNone(part)
        self.assertEqual(exp.heredoc_for(part), "$(echo hi)\n")


# ---------------------------------------------------------------------------
# Differential AST parity -- scanner path vs AST path
# ---------------------------------------------------------------------------

class QuotedSubstASTParityTest(unittest.TestCase):
    """For each quoted-subst case, assert the scanner path and AST path agree."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _both_extract_with_capture(
        cmd: str,
        capture_outputs: dict[str, str],
        env: Mapping[str, str] | None = None,
    ) -> tuple[
        tuple[list[str], list[Redirect], Optional[str]],
        tuple[list[str], list[Redirect], Optional[str]],
    ]:
        """Same as DifferentialASTParityTest._both_extract but with a stub capture_fn.

        Returns ((str_args, str_redirs, str_err), (ast_args, ast_redirs, ast_err)).
        """
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            cap = _stub_capture(capture_outputs)

            try:
                cleaned, exp, prog = parse_command(cmd, cap, wd, 30, 0, env=env)
            except (ValueError, ParseError) as exc:
                # parse_command rejected; string path can't run either
                str_result = extract_redirects(cmd, None)
                return str_result, ([], [], str(exc))

            if prog is None:
                str_result = extract_redirects(cleaned, exp)
                return str_result, ([], [], None)

            # String path: extract from cleaned string
            str_result = extract_redirects(cleaned, exp)

            # AST path: extract from CommandNode
            chain = program_to_chain(prog)
            if chain and chain[0][1]:
                cmd_node = chain[0][1][0]
                ast_result = extract_redirects(cmd_node, exp)
            else:
                ast_result = ([], [], None)

        return str_result, ast_result

    def _assert_parity(
        self, cmd: str, expected_args: list[str], capture_outputs: dict[str, str],
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Assert both paths produce the same args and match expected."""
        str_r, ast_r = self._both_extract_with_capture(cmd, capture_outputs, env=env)
        str_args, _, str_err = str_r
        ast_args, _, ast_err = ast_r

        # Both paths should have same error state
        self.assertEqual(
            str_err, ast_err,
            f"Error mismatch for {cmd!r}: string={str_err!r}, AST={ast_err!r}"
        )

        if str_err is not None:
            return  # both errored -- parity holds

        # Both should produce same args
        self.assertEqual(
            str_args, ast_args,
            f"Arg mismatch for {cmd!r}: string={str_args}, AST={ast_args}"
        )

        # And match expected
        self.assertEqual(
            str_args, expected_args,
            f"Expected {expected_args!r}, got {str_args!r}"
        )

    # ------------------------------------------------------------------
    # parity test cases
    # ------------------------------------------------------------------

    def test_dq_simple_parity(self) -> None:
        self._assert_parity(
            'echo "$(echo inner)"',
            ["echo", "inner"],
            {"echo inner": "inner"},
        )

    def test_dq_multi_word_parity(self) -> None:
        """Single arg with spaces -- no field splitting in either path."""
        self._assert_parity(
            'echo "$(echo a b c)"',
            ["echo", "a b c"],
            {"echo a b c": "a b c"},
        )

    def test_sq_literal_parity(self) -> None:
        self._assert_parity(
            "echo '$(echo inner)'",
            ["echo", "$(echo inner)"],
            {"echo inner": "SHOULD_NOT"},
        )

    def test_escaped_dollar_paren_parity(self) -> None:
        self._assert_parity(
            r'echo "\$(echo x)"',
            ["echo", r"$(echo x)"],
            {"echo x": "SHOULD_NOT"},
        )

    def test_compound_dq_parity(self) -> None:
        self._assert_parity(
            r'echo "pre$(echo mid)post"',
            ["echo", "premidpost"],
            {"echo mid": "mid"},
        )

    def test_dq_multiple_parity(self) -> None:
        self._assert_parity(
            r'echo "$(echo a)$(echo b)"',
            ["echo", "alphabeta"],
            {"echo a": "alpha", "echo b": "beta"},
        )

    def test_var_expansion_parity(self) -> None:
        """$HOME expands in both paths when env is provided."""
        self._assert_parity(
            'echo "$HOME"',
            ["echo", "/root"],
            {},
            env={"HOME": "/root"},
        )

    def test_dq_empty_subst_parity(self) -> None:
        """Quoted empty $() output -- one empty arg (POSIX), not dropped."""
        self._assert_parity(
            'echo "$(echo -n)"',
            ["echo", ""],
            {"echo -n": ""},
        )

    def test_unquoted_subst_still_works(self) -> None:
        """Unquoted $() must still work (regression check)."""
        self._assert_parity(
            "echo $(echo outer)",
            ["echo", "outer"],
            {"echo outer": "outer"},
        )

    def test_unquoted_subst_with_dq_nested(self) -> None:
        """Unquoted $() with double-quoted content inside."""
        self._assert_parity(
            'echo $(echo "$(echo inner)")',
            ["echo", "inner"],
            {"echo inner": "inner",
             'echo "$(echo inner)"': "inner"},
        )


if __name__ == "__main__":
    unittest.main()
