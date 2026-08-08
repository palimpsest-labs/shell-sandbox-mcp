"""Golden + parity tests for the direct for-in-word expanders.

Covers the refactor of ``_expand_for_word`` into
:func:`_expand_for_word_direct` (single-word) and
:func:`_expand_for_words` (batched), replacing the synthetic
``__for_expand`` wrapper with a direct parse.

Each test exercises both the direct and batched expanders and asserts they
agree with each other AND with the retained synthetic path
(:func:`_expand_for_word_synthetic`), so a behavioural regression in the
refactor surfaces as a parity failure.

Run with::

    python3 -m pytest tests/test_expand_for_word.py -q
"""

import tempfile
import unittest
from pathlib import Path

from shell_sandbox_mcp import server
from shell_sandbox_mcp.runner import (
    _expand_for_word_direct,
    _expand_for_words,
    _expand_for_word_synthetic,
)

BASE = {"HOME": "/root", "X": "a b", "Y": "  a  b  ", "Z": "", "COL": "a:b::c"}


def _direct(word, wd, *, positional=(), ifs=None, field_split=True):
    return _expand_for_word_direct(
        word, wd, 30, 0, BASE, server,
        positional=positional, field_split=field_split, ifs=ifs,
    )


def _synthetic(word, wd, *, positional=(), ifs=None, field_split=True):
    return _expand_for_word_synthetic(
        word, wd, 30, 0, BASE, server,
        positional=positional, field_split=field_split, ifs=ifs,
    )


class ExpandForWordTest(unittest.TestCase):
    """Golden + parity tests for the direct/batched for-word expanders."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wd = Path(self._tmp.name)
        # Files for glob expansion.
        (self.wd / "a.txt").write_text("x")
        (self.wd / "b.txt").write_text("y")
        (self.wd / "sub").mkdir()
        (self.wd / "sub" / "c.log").write_text("z")
        self._orig_capture = server._capture_stdout

    def tearDown(self) -> None:
        server._capture_stdout = self._orig_capture
        self._tmp.cleanup()

    def _stub_capture(self, out: bytes = b"CAP") -> None:
        def fake(command, work_dir, timeout, depth,
                 deadline=None, subst_count=None, env=None):
            return 0, out
        server._capture_stdout = fake

    # ------------------------------------------------------------------
    # single-word golden tests
    # ------------------------------------------------------------------

    def test_literal(self) -> None:
        self.assertEqual(_direct("hello", self.wd), ["hello"])

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(_direct("", self.wd), [""])

    def test_quoted_empty_word(self) -> None:
        # The in-word is the two-char literal "" (quote chars preserved).
        self.assertEqual(_direct('""', self.wd), [""])

    def test_var_expansion(self) -> None:
        self.assertEqual(_direct("$X", self.wd), ["a", "b"])

    def test_braced_default(self) -> None:
        self.assertEqual(_direct("${UNSET:-fallback}", self.wd), ["fallback"])

    def test_empty_var_dropped(self) -> None:
        self.assertEqual(_direct("$Z", self.wd), [""])

    def test_quoted_empty_var_kept(self) -> None:
        self.assertEqual(_direct('"$Z"', self.wd), [""])

    # -- "$@" / "$*" fan-out -------------------------------------------

    def test_at_quoted_fanout(self) -> None:
        self.assertEqual(
            _direct('"$@"', self.wd, positional=("a", "b", "c")),
            ["a", "b", "c"],
        )

    def test_at_quoted_zero_positionals_empty_iteration(self) -> None:
        # Verified against the synthetic path + test_parser_at_split: zero
        # positionals → one empty iteration ([""]).
        self.assertEqual(_direct('"$@"', self.wd, positional=()), [""])

    def test_at_unquoted_join_then_split(self) -> None:
        # Unquoted $@ resolves via env["@"], absent from BASE → empty (dropped).
        # Both the direct and synthetic paths agree on this (parity-preserved).
        self.assertEqual(_direct("$@", self.wd, positional=("a", "b")), [""])

    def test_at_one_empty_positional(self) -> None:
        self.assertEqual(_direct('"$@"', self.wd, positional=("",)), [""])

    def test_at_mixed_empty_positionals(self) -> None:
        self.assertEqual(
            _direct('"$@"', self.wd, positional=("a", "", "b")),
            ["a", "", "b"],
        )

    def test_star_join(self) -> None:
        self.assertEqual(
            _direct('"$*"', self.wd, positional=("a", "b", "c")),
            ["a b c"],
        )

    def test_star_join_zero_positionals(self) -> None:
        self.assertEqual(_direct('"$*"', self.wd, positional=()), [""])

    # -- quoted / escaped spaces ---------------------------------------

    def test_quoted_space_stays_one_arg(self) -> None:
        # The lexer preserves the quotes in the in-word token, so "a b"
        # re-parses as ONE arg (matching the synthetic path).
        self.assertEqual(_direct('"a b"', self.wd), ["a b"])

    def test_single_quoted_space(self) -> None:
        self.assertEqual(_direct("'a b'", self.wd), ["a b"])

    # -- IFS splitting -------------------------------------------------

    def test_ifs_default_ws_split(self) -> None:
        self.assertEqual(_direct("$X", self.wd, ifs=None), ["a", "b"])

    def test_ifs_nws_empty_preservation(self) -> None:
        # COL="a:b::c" with IFS=":" → empty preserved between consecutive colons.
        self.assertEqual(
            _direct("$COL", self.wd, ifs=":"),
            ["a", "b", "", "c"],
        )

    def test_ifs_mixed(self) -> None:
        self.assertEqual(_direct("$X", self.wd, ifs=" "), ["a", "b"])

    def test_ifs_empty_no_split(self) -> None:
        self.assertEqual(_direct("$X", self.wd, ifs=""), ["a b"])

    # -- glob ----------------------------------------------------------

    def test_glob_matched(self) -> None:
        matches = _direct("*.txt", self.wd)
        self.assertEqual(matches, [str(self.wd / "a.txt"), str(self.wd / "b.txt")])

    def test_glob_subdir(self) -> None:
        matches = _direct("sub/*.log", self.wd)
        self.assertEqual(matches, [str(self.wd / "sub" / "c.log")])

    def test_glob_unmatched_stays_literal(self) -> None:
        self.assertEqual(_direct("nope*.xyz", self.wd), ["nope*.xyz"])

    # -- metachar / literal edge cases ---------------------------------

    def test_leading_dash(self) -> None:
        self.assertEqual(_direct("-leading", self.wd), ["-leading"])

    def test_assignment_like_word(self) -> None:
        # x=1 as a single word stays one arg (matches synthetic).
        self.assertEqual(_direct("x=1", self.wd), ["x=1"])

    def test_quoted_metachars_literal(self) -> None:
        self.assertEqual(_direct("'a|b'", self.wd), ["a|b"])
        self.assertEqual(_direct("'a;b'", self.wd), ["a;b"])
        self.assertEqual(_direct("'a&&b'", self.wd), ["a&&b"])

    # -- case-subject (field_split=False) ------------------------------

    def test_case_subject_no_field_split(self) -> None:
        self.assertEqual(
            _direct("$X", self.wd, field_split=False), ["a b"],
        )
        self.assertEqual(
            _direct('"$@"', self.wd, positional=("p", "q"), field_split=False),
            ["p", "q"],
        )

    # ------------------------------------------------------------------
    # batched vs direct parity
    # ------------------------------------------------------------------

    def test_batch_equals_direct_flat(self) -> None:
        words = [
            "hello", "a.txt", "*.txt", "sub/*.log", "nope*.xyz",
            "$X", "$Z", '"$Z"', "${X:-fallback}", 'pre$X',
            '"$@"', '$@', '"$*"', '"a b"', '""', '-leading', 'x=1',
            "'a|b'", "'a;b'", "'a&&b'",
        ]
        flat_direct = [a for w in words for a in _direct(w, self.wd)]
        batch = _expand_for_words(words, self.wd, 30, 0, BASE, server)
        self.assertEqual(batch, flat_direct)

    def test_batch_equals_direct_with_positionals(self) -> None:
        words = ['"$@"', "$@", '"$*"', "$X", 'pre$X', "hello"]
        flat_direct = [
            a for w in words
            for a in _direct(w, self.wd, positional=("a", "b", "c"))
        ]
        batch = _expand_for_words(
            words, self.wd, 30, 0, BASE, server,
            positional=("a", "b", "c"),
        )
        self.assertEqual(batch, flat_direct)

    def test_batch_equals_direct_with_ifs(self) -> None:
        words = ["$COL", '"$@"', "a:b", '"a:b"', "$X"]
        flat_direct = [
            a for w in words for a in _direct(w, self.wd, ifs=":")
        ]
        batch = _expand_for_words(words, self.wd, 30, 0, BASE, server, ifs=":")
        self.assertEqual(batch, flat_direct)

    def test_batch_empty_list(self) -> None:
        self.assertEqual(_expand_for_words([], self.wd, 30, 0, BASE, server), [])

    def test_batch_with_empty_word(self) -> None:
        self.assertEqual(
            _expand_for_words(["a", "", "b"], self.wd, 30, 0, BASE, server),
            ["a", "", "b"],
        )

    # ------------------------------------------------------------------
    # direct vs synthetic parity (A/B)
    # ------------------------------------------------------------------

    def test_direct_matches_synthetic_corpus(self) -> None:
        words = [
            "hello", "a.txt", "*.txt", "sub/*.log", "nope*.xyz",
            "$X", "$Z", '"$Z"', "${X:-fallback}", 'pre$X',
            '"$@"', '$@', '"$*"', '"a b"', "'a b'", '""',
            "-leading", "x=1", "'a|b'", "'a;b'", "'a&&b'", "'single'",
        ]
        for pos in [(), ("a",), ("a", "b", "c")]:
            for w in words:
                self.assertEqual(
                    _direct(w, self.wd, positional=pos),
                    _synthetic(w, self.wd, positional=pos),
                    msg=f"word={w!r} positional={pos!r}",
                )

    def test_direct_matches_synthetic_ifs(self) -> None:
        for w in ["$COL", '"$@"', "a:b", '"a:b"', "$X", "$Z", '""']:
            self.assertEqual(
                _direct(w, self.wd, positional=("p", "q"), ifs=":"),
                _synthetic(w, self.wd, positional=("p", "q"), ifs=":"),
                msg=f"word={w!r}",
            )

    def test_direct_matches_synthetic_case_subject(self) -> None:
        for w in ["$X", '"$X"', "$@", '"$@"', "$Z", '"$Z"', "a b", '"a b"', "$*", '"$*"']:
            self.assertEqual(
                _direct(w, self.wd, positional=("p", "q"), field_split=False),
                _synthetic(w, self.wd, positional=("p", "q"), field_split=False),
                msg=f"word={w!r}",
            )

    def test_direct_matches_synthetic_subst(self) -> None:
        self._stub_capture(b"CAP")
        for w in ["$(echo hi)", '"$(echo hi)"', "$(echo)"]:
            self.assertEqual(
                _direct(w, self.wd),
                _synthetic(w, self.wd),
                msg=f"word={w!r}",
            )

    # ------------------------------------------------------------------
    # live execution: for-loop with batched expansion
    # ------------------------------------------------------------------

    def test_for_loop_through_shell(self) -> None:
        from shell_sandbox_mcp.parser import CommandNode
        from shell_sandbox_mcp.server import _extract_redirects
        calls: list[dict] = []
        orig_segment, orig_pipeline = server._run_segment, server._run_pipeline

        def fake_segment(command, work_dir, timeout, expansion=None, **kwargs):
            if isinstance(command, CommandNode):
                args, _, _ = _extract_redirects(command, expansion, work_dir)
                calls.append(" ".join(args))
            return 0, ""

        server._run_segment = fake_segment
        server._run_pipeline = lambda segs, wd, to, **kw: (0, "")
        try:
            server.shell_run(
                'for x in a "b c" *.txt; do echo $x; done',
                cwd=str(self.wd), timeout=30,
            )
        finally:
            server._run_segment = orig_segment
            server._run_pipeline = orig_pipeline
        # a, "b c", then glob matches a.txt / b.txt
        self.assertEqual(
            calls,
            ["echo a", "echo b c",
             f"echo {self.wd}/a.txt", f"echo {self.wd}/b.txt"],
        )


if __name__ == "__main__":
    unittest.main()
