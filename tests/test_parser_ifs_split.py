"""Tests for IFS field splitting (_field_split pure function + end-to-end).

Covers:
- Pure _field_split with default IFS, custom IFS (ws-only, nws-only, mixed).
- End-to-end via extract_redirects with IFS threaded through Expansion.
- Quoted vs unquoted expansions.
- Edge cases: empty value, whitespace-only, adjacent text, glob.
"""

import unittest

from shell_sandbox_mcp.parser import _IFS_DEFAULT, _IFS_WS, _field_split


# ---------------------------------------------------------------------------
# Pure _field_split tests
# ---------------------------------------------------------------------------

class FieldSplitPureTest(unittest.TestCase):
    """Test _field_split directly with value + IFS inputs."""

    # -- empty / whitespace-only -------------------------------------------

    def test_empty_value_default_ifs(self):
        self.assertEqual(_field_split("", _IFS_DEFAULT), [])

    def test_empty_value_custom_ifs(self):
        self.assertEqual(_field_split("", ","), [])

    def test_empty_value_empty_ifs(self):
        self.assertEqual(_field_split("", ""), [])

    def test_whitespace_only_default_ifs(self):
        self.assertEqual(_field_split("   ", _IFS_DEFAULT), [])

    def test_whitespace_only_tab_newline(self):
        self.assertEqual(_field_split(" \t \n ", _IFS_DEFAULT), [])

    # -- default IFS (space/tab/newline) -----------------------------------

    def test_single_word(self):
        self.assertEqual(_field_split("hello", _IFS_DEFAULT), ["hello"])

    def test_two_words(self):
        self.assertEqual(_field_split("a b", _IFS_DEFAULT), ["a", "b"])

    def test_leading_ws_trimmed(self):
        self.assertEqual(_field_split("  a b", _IFS_DEFAULT), ["a", "b"])

    def test_trailing_ws_trimmed(self):
        self.assertEqual(_field_split("a b  ", _IFS_DEFAULT), ["a", "b"])

    def test_ws_runs_collapse(self):
        self.assertEqual(_field_split("a   b", _IFS_DEFAULT), ["a", "b"])

    def test_tabs_and_newlines(self):
        self.assertEqual(_field_split("a\tb\nc", _IFS_DEFAULT), ["a", "b", "c"])

    # -- unset IFS (should behave like default) ----------------------------

    def test_unset_ifs_like_default(self):
        # Unset IFS uses default — tested via _effective_ifs
        from shell_sandbox_mcp.parser import _effective_ifs
        self.assertEqual(_effective_ifs(None), _IFS_DEFAULT)
        self.assertEqual(_effective_ifs(""), "")
        self.assertEqual(_effective_ifs(","), ",")

    # -- non-ws-only IFS (e.g. IFS=",") — each char is hard delimiter ------

    def test_nws_single_word(self):
        self.assertEqual(_field_split("hello", ","), ["hello"])

    def test_nws_two_fields(self):
        self.assertEqual(_field_split("a,b", ","), ["a", "b"])

    def test_nws_consecutive_delimiters(self):
        # ",," → each comma is a separate delimiter → two empty fields
        self.assertEqual(_field_split("a,,b", ","), ["a", "", "b"])

    def test_nws_leading_delimiter(self):
        self.assertEqual(_field_split(",a", ","), ["", "a"])

    def test_nws_trailing_delimiter(self):
        self.assertEqual(_field_split("a,", ","), ["a", ""])

    def test_nws_leading_and_trailing(self):
        self.assertEqual(_field_split(",a,", ","), ["", "a", ""])

    def test_nws_all_delimiters(self):
        self.assertEqual(_field_split(",,", ","), ["", "", ""])

    def test_nws_empty_value(self):
        self.assertEqual(_field_split("", ","), [])

    # -- ws-only IFS (e.g. IFS=" ") ---------------------------------------

    def test_ws_only_space(self):
        self.assertEqual(_field_split("a b", " "), ["a", "b"])

    def test_ws_only_tab(self):
        self.assertEqual(_field_split("a\tb", "\t"), ["a", "b"])

    def test_ws_only_mixed_ws(self):
        self.assertEqual(_field_split("a \t b", " \t"), ["a", "b"])

    # -- mixed IFS (ws + non-ws) — maximal-run delimiter sequences ---------

    def test_mixed_simple(self):
        # "," in IFS → non-ws delimiter; space is absorbed
        self.assertEqual(_field_split("a,b", " ,"), ["a", "b"])

    def test_mixed_space_between_commas(self):
        # "a, ,b" → ", ," is one delimiter sequence
        self.assertEqual(_field_split("a, ,b", " ,"), ["a", "b"])

    def test_mixed_space_around_commas(self):
        # "a , b , c" → ", " and " , " each one delimiter
        self.assertEqual(_field_split("a , b , c", " ,"), ["a", "b", "c"])

    def test_mixed_ws_after_nws_absorbed(self):
        self.assertEqual(_field_split("a,  b", " ,"), ["a", "b"])

    def test_mixed_ws_before_nws_absorbed(self):
        self.assertEqual(_field_split("a  ,b", " ,"), ["a", "b"])

    def test_mixed_consecutive_nws_no_ws_between(self):
        # ",," with IFS=" ," → one delimiter sequence (no ws between)
        self.assertEqual(_field_split("a,,b", " ,"), ["a", "b"])

    def test_mixed_leading_nws_delim(self):
        self.assertEqual(_field_split(",a", " ,"), ["", "a"])

    def test_mixed_trailing_nws_delim(self):
        self.assertEqual(_field_split("a,", " ,"), ["a", ""])

    def test_mixed_leading_ws_trimmed(self):
        self.assertEqual(_field_split("  a", " ,"), ["a"])

    def test_mixed_trailing_ws_trimmed(self):
        self.assertEqual(_field_split("a  ", " ,"), ["a"])

    def test_mixed_plan_example(self):
        # Plan §4: IFS=" ,"; X="a, b , c" → ["a","b","c"]
        self.assertEqual(_field_split("a, b , c", " ,"), ["a", "b", "c"])

    def test_mixed_colon_space(self):
        # IFS=": " (colon + space)
        self.assertEqual(_field_split("a:b", ": "), ["a", "b"])
        self.assertEqual(_field_split("a::b", ": "), ["a", "b"])  # :: one sequence
        self.assertEqual(_field_split("a: b", ": "), ["a", "b"])
        self.assertEqual(_field_split("a :b", ": "), ["a", "b"])
        self.assertEqual(_field_split("a : b", ": "), ["a", "b"])

    # -- empty IFS → no splitting ------------------------------------------

    def test_empty_ifs_no_split(self):
        self.assertEqual(_field_split("a b c", ""), ["a b c"])

    def test_empty_ifs_empty_value(self):
        self.assertEqual(_field_split("", ""), [])


# ---------------------------------------------------------------------------
# End-to-end IFS field splitting via extract_redirects
# ---------------------------------------------------------------------------

import tempfile
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


ENV = {"HOME": "/root", "X": "a b", "Y": "  a  b  ", "Z": ""}


def _stub_capture(outputs=None):
    outputs = outputs or {}
    def fake_capture(inner):
        val = outputs.get(inner, "")
        return 0, val.encode("utf-8")
    return fake_capture


def _extract_ast(cmd, env=None, outputs=None, ifs=None):
    """AST-path extract with IFS threaded through expansion."""
    cap = _stub_capture(outputs)
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        try:
            cleaned, exp, prog = parse_command(cmd, cap, wd, 30, 0, env=env)
        except (ValueError, ParseError) as exc:
            return None, str(exc)

        if prog is None:
            return None, "no program"

        exp.ifs = ifs
        chain = program_to_chain(prog)
        if chain and chain[0][1]:
            cmd_node = chain[0][1][0]
            args, redirs, err = extract_redirects(cmd_node, exp)
            return args, err
        return None, "no chain"


class FieldSplitEndToEndTest(unittest.TestCase):
    """End-to-end tests: $VAR expansion with IFS field splitting."""

    # -- default IFS (X="a b") → splits to two args -----------------------

    def test_default_ifs_splits_space(self):
        args, err = _extract_ast("echo $X", env=ENV)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b"])

    def test_quoted_no_split(self):
        args, err = _extract_ast('echo "$X"', env=ENV)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a b"])

    def test_braced_quoted_no_split(self):
        args, err = _extract_ast('echo "${X}"', env=ENV)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a b"])

    # -- empty value (X="") ------------------------------------------------

    def test_empty_value_dropped(self):
        # X="" → _field_split("") → [] → dropped
        args, err = _extract_ast("echo $Z", env=ENV)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo"])

    # -- whitespace-only value (X="   ") -----------------------------------

    def test_whitespace_only_dropped(self):
        args, err = _extract_ast("echo $Y", env=ENV)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b"])

    # -- custom IFS="," (non-ws only) --------------------------------------

    def test_nws_ifs_comma_single(self):
        env = {"X": "a"}
        args, err = _extract_ast("echo $X", env=env, ifs=",")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a"])

    def test_nws_ifs_comma_consecutive(self):
        env = {"X": "a,,b"}
        args, err = _extract_ast("echo $X", env=env, ifs=",")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "", "b"])

    def test_nws_ifs_leading_trailing_empty(self):
        env = {"X": ",a,"}
        args, err = _extract_ast("echo $X", env=env, ifs=",")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "", "a", ""])

    def test_nws_ifs_empty_value(self):
        env = {"X": ""}
        args, err = _extract_ast("echo $X", env=env, ifs=",")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo"])

    # -- custom IFS=" " (ws-only) ------------------------------------------

    def test_ws_only_ifs_splits(self):
        env = {"X": "a b"}
        args, err = _extract_ast("echo $X", env=env, ifs=" ")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b"])

    def test_ws_only_ifs_trim(self):
        env = {"X": "  a  b  "}
        args, err = _extract_ast("echo $X", env=env, ifs=" ")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b"])

    # -- empty IFS → no splitting ------------------------------------------

    def test_empty_ifs_no_split_end_to_end(self):
        env = {"X": "a b"}
        args, err = _extract_ast("echo $X", env=env, ifs="")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a b"])

    # -- unset IFS (None) → default IFS -----------------------------------

    def test_unset_ifs_uses_default(self):
        env = {"X": "a b"}
        args, err = _extract_ast("echo $X", env=env)  # ifs=None
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b"])

    # -- adjacent text (pre/post) ------------------------------------------

    def test_pre_text_split(self):
        env = {"X": "a b"}
        args, err = _extract_ast("echo pre$X", env=env)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "prea", "b"])

    def test_post_text_split(self):
        env = {"X": "a b"}
        args, err = _extract_ast("echo ${X}post", env=env)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "bpost"])

    def test_pre_post_text_split(self):
        env = {"X": "a b"}
        args, err = _extract_ast("echo pre${X}post", env=env)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "prea", "bpost"])

    # -- glob interaction --------------------------------------------------

    def test_glob_after_split(self):
        # X="*.py *.md" — after split, each field undergoes glob expansion
        env = {"X": "*.py *.md"}
        args, err = _extract_ast("echo $X", env=env)
        self.assertIsNone(err)
        # No actual .py/.md files in temp dir → unmatched globs stay literal
        self.assertEqual(args, ["echo", "*.py", "*.md"])

    # -- mixed IFS (ws + non-ws) -------------------------------------------

    def test_mixed_ifs_end_to_end(self):
        env = {"X": "a, b , c"}
        args, err = _extract_ast("echo $X", env=env, ifs=" ,")
        self.assertIsNone(err)
        # ", " and " , " each one delimiter sequence → "a", "b", "c"
        self.assertEqual(args, ["echo", "a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
