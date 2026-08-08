"""Golden regression corpus for the shell parser.

Every ``(input, expected_args, expected_redirects, expected_expansion)``
triple below pins the EXACT parser output — any change to the lexer, AST
projection, or expansion behaviour that alters these strings/values fails
the suite.  The historical ``DifferentialASTParityTest`` cross-validation
was deleted: both string and AST paths now route through ``_build_ast``,
making that parity check tautological.  Instead, this suite locks the
observable output so drift cannot slip through silently.
"""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from shell_sandbox_mcp.parser import (
    Expansion,
    Lexer,
    Redirect,
    _build_ast,
    _serialize_command,
    extract_redirects,
    parse_command,
    program_to_chain,
)


# ---------------------------------------------------------------------------
# Table-driven case dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitCase:
    """A single chain-projection golden case."""
    name: str
    command: str
    # expected_chain: (op|None, stages: tuple[str,...], backgrounded: bool)
    expected_chain: Tuple[Tuple[Optional[str], Tuple[str, ...], bool], ...]


@dataclass(frozen=True)
class RedirectCase:
    """A single extract_redirects golden case."""
    name: str
    command: str
    expected_args: Tuple[str, ...] = ()
    expected_redirects: Tuple[Redirect, ...] = ()
    expected_err: Optional[str] = None
    # capture_outputs: mapping of inner "$(...)" text -> captured stdout
    capture_outputs: Mapping[str, str] = ()


@dataclass(frozen=True)
class ExpansionCase:
    """A single parse_command expansion golden case."""
    name: str
    command: str
    capture_outputs: Mapping[str, str] = ()
    expected_heredoc_body: Optional[str] = None
    expected_arg_value: Optional[str] = None
    expected_cleaned_starts_with: Optional[str] = None
    raises_with_substring: Optional[str] = None


# ---------------------------------------------------------------------------
# SplitCommandGoldenTest — pins program_to_chain + _serialize_command projections
# ---------------------------------------------------------------------------
# The live path uses Lexer → _build_ast → program_to_chain →
# _serialize_command to project a command string into the chain format.
# These goldens pin that projection exactly.

SPLIT_CASES = (
    SplitCase("no_operator_single_segment", "ls -la",
              ((None, ("ls -la",), False),)),
    SplitCase("semicolon_splits", "echo hi; echo bye",
              ((None, ("echo hi",), False), (";", ("echo bye",), False))),
    SplitCase("and_and_splits", "make && make test",
              ((None, ("make",), False), ("&&", ("make test",), False))),
    SplitCase("or_or_splits", "false || echo fallback",
              ((None, ("false",), False), ("||", ("echo fallback",), False))),
    SplitCase("mixed_operators", "a && b; c || d",
              ((None, ("a",), False), ("&&", ("b",), False),
               (";", ("c",), False), ("||", ("d",), False))),
    SplitCase("operator_inside_quotes_preserved", 'echo "a; b"',
              ((None, ('echo "a; b"',), False),)),
    SplitCase("operator_inside_quotes_single", "printf 'a && b'; ls",
              ((None, ("printf 'a && b'",), False), (";", ("ls",), False))),
    SplitCase("whitespace_and_empty_segments_dropped", "  a   ;;  b  ",
              ((None, ("a",), False), (";", ("b",), False))),
    SplitCase("empty_command", "", ()),
    SplitCase("whitespace_only_command", "   ", ()),
    SplitCase("only_operator_is_empty", ";", ()),
    SplitCase("single_pipe_splits_into_stages", "ls | wc",
              ((None, ("ls", "wc"), False),)),
    SplitCase("multi_stage_pipeline", "a | b | c",
              ((None, ("a", "b", "c"), False),)),
    SplitCase("pipe_inside_quotes_preserved", 'echo "a|b" | wc',
              ((None, ('echo "a|b"', "wc"), False),)),
    SplitCase("pipe_inside_quotes_single", "printf 'a | b'",
              ((None, ("printf 'a | b'",), False),)),
    SplitCase("pipe_distinguished_from_or_or", "false || echo fallback | wc",
              ((None, ("false",), False),
               ("||", ("echo fallback", "wc"), False))),
    SplitCase("pipe_and_chain_mix", "a | b && c | d ; e",
              ((None, ("a", "b"), False), ("&&", ("c", "d"), False),
               (";", ("e",), False))),
    SplitCase("pipe_at_start_drops_empty_lead", "| ls",
              ((None, ("ls",), False),)),
    SplitCase("pipe_at_end_drops_empty_tail", "ls |",
              ((None, ("ls",), False),)),
    SplitCase("triple_pipe_treated_as_or_or_plus_empty_stage", "a ||| b",
              ((None, ("a",), False), ("||", ("b",), False))),
    SplitCase("bare_ampersand_backgrounds_pipeline", "a & b",
              ((None, ("a",), True), (None, ("b",), False))),
    SplitCase("bare_ampersand_backgrounds_two", "echo hi & ls",
              ((None, ("echo hi",), True), (None, ("ls",), False))),
    SplitCase("ampersand_with_and_operator", "a && b & c",
              ((None, ("a",), False), ("&&", ("b",), True),
               (None, ("c",), False))),
    SplitCase("double_ampersand_stays_and_operator", "a && b",
              ((None, ("a",), False), ("&&", ("b",), False))),
    SplitCase("fd_dup_ampersand_is_not_backgrounding", "echo hi 2>&1",
              ((None, ("echo hi 2>&1",), False),)),
    SplitCase("fd_dup_1gt2_not_backgrounding", "cmd 1>&2",
              ((None, ("cmd 1>&2",), False),)),
    SplitCase("fd_dup_with_redir_and_background", "grep x 2>err &",
              ((None, ("grep x 2>err",), True),)),
    SplitCase("ampersand_inside_quotes_preserved", 'echo "a & b"',
              ((None, ('echo "a & b"',), False),)),
    SplitCase("ampersand_inside_quotes_single", "printf 'x & y'",
              ((None, ("printf 'x & y'",), False),)),
)


class SplitCommandGoldenTest(unittest.TestCase):
    """Golden: exact string projections from the live chain projection path."""

    def test_split_golden(self) -> None:
        for case in SPLIT_CASES:
            with self.subTest(name=case.name, command=case.command):
                tokens = Lexer(case.command, replay_mode=True).tokenize()
                program = _build_ast(tokens, Expansion())
                chains = program_to_chain(program)
                projected = [
                    (op, [_serialize_command(cmd) for cmd in cmd_nodes], bg)
                    for op, cmd_nodes, bg in chains
                ]
                self.assertEqual(
                    projected,
                    [(op, list(stages), bg)
                     for op, stages, bg in case.expected_chain],
                )


# ---------------------------------------------------------------------------
# RedirectsGoldenTest — pins exact (args, redirects, err) output
# ---------------------------------------------------------------------------
# All rows are driven through a stubbed capture_fn + parse_command, then
# extract_redirects on the cleaned command + expansion.  Plain assertEqual
# on Redirect lists works field-by-field (frozen dataclass, eq=True).

REDIRECT_CASES = (
    RedirectCase("simple_stdout_redirect", "echo hi > out.txt",
                 ("echo", "hi"),
                 (Redirect(fd=1, op=">", raw_target="out.txt"),)),
    RedirectCase("stdout_append", "echo hi >> log.txt",
                 ("echo", "hi"),
                 (Redirect(fd=1, op=">>", raw_target="log.txt"),)),
    RedirectCase("stderr_redirect", "cmd 2> err.txt",
                 ("cmd",),
                 (Redirect(fd=2, op=">", raw_target="err.txt"),)),
    RedirectCase("stderr_append", "cmd 2>> err.txt",
                 ("cmd",),
                 (Redirect(fd=2, op=">>", raw_target="err.txt"),)),
    RedirectCase("2gt1_fd_dup", "cmd 2>&1",
                 ("cmd",),
                 (Redirect(fd=2, op=">&", target_fd=1, raw_target="1"),)),
    RedirectCase("1gt2_fd_dup", "cmd 1>&2",
                 ("cmd",),
                 (Redirect(fd=1, op=">&", target_fd=2, raw_target="2"),)),
    RedirectCase("2gt1x_not_fd_dup", "cmd 2>&1x",
                 ("cmd",),
                 (Redirect(fd=2, op=">", raw_target="&1x"),)),
    RedirectCase("1gt2y_not_fd_dup", "cmd 1>&2y",
                 ("cmd",),
                 (Redirect(fd=1, op=">", raw_target="&2y"),)),
    RedirectCase("quoted_operator_not_redirect", 'echo ">" hello',
                 ("echo", ">", "hello"), ()),
    RedirectCase("quoted_operator_single_quote", "echo '>' hello",
                 ("echo", ">", "hello"), ()),
    RedirectCase("redirect_leading", ">out echo x",
                 ("echo", "x"),
                 (Redirect(fd=1, op=">", raw_target="out"),)),
    RedirectCase("redirect_middle", "echo a > f b",
                 ("echo", "a", "b"),
                 (Redirect(fd=1, op=">", raw_target="f"),)),
    RedirectCase("multiple_redirects", "cmd 2>e 1>&2",
                 ("cmd",),
                 (Redirect(fd=2, op=">", raw_target="e"),
                  Redirect(fd=1, op=">&", target_fd=2, raw_target="2"))),
    RedirectCase("glued_not_redirect", "echo foo>bar",
                 ("echo", "foo>bar"), ()),
    RedirectCase("glued_target_ok", ">out.txt echo hi",
                 ("echo", "hi"),
                 (Redirect(fd=1, op=">", raw_target="out.txt"),)),
    RedirectCase("missing_target_error", "echo >",
                 expected_err="Redirect operator missing target file"),
    RedirectCase("missing_target_2gt_error", "echo 2>",
                 expected_err="Redirect operator missing target file"),
    RedirectCase("fd_gt_2_error", "echo 3> f",
                 expected_err="Redirects only support fds 1 and 2 (got 3)"),
    RedirectCase("fd_0_error", "echo 0> f",
                 expected_err="Redirects only support fds 1 and 2 (got 0)"),
    RedirectCase("2gt3_error", "cmd 2>&3",
                 expected_err="Redirect dup target fd must be 1 or 2"),
    RedirectCase("input_redirect_error", "cmd < file",
                 ("cmd",),
                 (Redirect(fd=0, op="<", raw_target="file"),)),
    RedirectCase("input_redirect_glued", "cmd <file",
                 ("cmd",),
                 (Redirect(fd=0, op="<", raw_target="file"),)),
    RedirectCase("input_redirect_missing_target", "cmd <",
                 expected_err="Input redirect missing target file"),
    RedirectCase("input_heredoc_error", "cmd << EOF",
                 expected_err="heredoc delimiter 'EOF' not found"),
    RedirectCase("unbalanced_quotes_error", 'echo "hi',
                 expected_err="Unbalanced quotes in command"),
    RedirectCase("1gt_redirect", "cmd 1> out.txt",
                 ("cmd",),
                 (Redirect(fd=1, op=">", raw_target="out.txt"),)),
    RedirectCase("1gtgt_redirect", "cmd 1>> out.txt",
                 ("cmd",),
                 (Redirect(fd=1, op=">>", raw_target="out.txt"),)),
    RedirectCase("no_args_only_redirect", "> out.txt",
                 (),
                 (Redirect(fd=1, op=">", raw_target="out.txt"),)),
)


class RedirectsGoldenTest(unittest.TestCase):
    """Golden: exact (args, redirects, err) from extract_redirects."""

    def _stub_capture(self, outputs: Mapping[str, str]):
        def fake_capture(inner: str):
            val = outputs.get(inner, "")
            return 0, val.encode("utf-8")
        return fake_capture

    def test_redirect_golden(self) -> None:
        for case in REDIRECT_CASES:
            with self.subTest(name=case.name, command=case.command):
                try:
                    with tempfile.TemporaryDirectory() as td:
                        wd = Path(td)
                        cap = self._stub_capture(case.capture_outputs)
                        cleaned, exp, _prog = parse_command(
                            case.command, cap, wd, 30, 0,
                        )
                    args, redirs, err = extract_redirects(cleaned, exp)
                except ValueError:
                    # parse_command rejects these inputs in populate mode
                    # (bad fd, fd-dup target, missing heredoc terminator).
                    # Drive extract_redirects on the raw string instead, which
                    # is where these error messages originate.
                    args, redirs, err = extract_redirects(case.command)
                self.assertEqual(err, case.expected_err)
                self.assertEqual(args, list(case.expected_args))
                self.assertEqual(redirs, list(case.expected_redirects))

    # ------------------------------------------------------------------
    # Synthetic-input tests — keep method-based.
    #
    # These construct an input containing the runtime \x01H0\x01 heredoc
    # sentinel by slicing parse_command's cleaned output.  They are fragile
    # to the exact sentinel format the serializer emits, so they are pinned
    # here rather than as data rows.
    # ------------------------------------------------------------------

    def test_input_redirect_then_heredoc_rejected(self) -> None:
        # Parse a heredoc to get a real expansion + cleaned string
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            cleaned, exp, _prog = parse_command(
                "cat <<EOF\nbody\nEOF", lambda i: (0, b""), wd, 30, 0,
            )
        # Build test cmd: replace "cat" with "cmd < file"
        heredoc_tail = cleaned[len("cat "):]  # "<< \x01H0\x01" at runtime
        test_cmd = "cmd < file " + heredoc_tail
        args, redirs, err = extract_redirects(test_cmd, expansion=exp)
        self.assertIsNotNone(err)
        self.assertIn("Multiple stdin redirects", err)

    def test_heredoc_then_input_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            cleaned, exp, _prog = parse_command(
                "cat <<EOF\nbody\nEOF", lambda i: (0, b""), wd, 30, 0,
            )
        heredoc_tail = cleaned[len("cat "):]
        test_cmd = "cmd " + heredoc_tail + " < file"
        args, redirs, err = extract_redirects(test_cmd, expansion=exp)
        self.assertIsNotNone(err)
        self.assertIn("Multiple stdin redirects", err)


# ---------------------------------------------------------------------------
# AST helpers for opaque lookups
# ---------------------------------------------------------------------------

def _find_hd_sentinel(prog):
    """Return the first heredoc sentinel WordPart from the first command.

    Indexes ``[0]`` of the first chain/command only — every expansion case
    below uses a single-command program.
    """
    cmd = prog.chains[0].pipeline.commands[0]
    for rs in cmd.redirects:
        for p in rs.target.parts:
            if p.is_hd_sentinel:
                return p
    return None


def _find_arg_sentinel(prog):
    """Return the first arg-substitution sentinel WordPart from the first
    command.  Indexes ``[0]`` of the first chain/command only.
    """
    cmd = prog.chains[0].pipeline.commands[0]
    for w in cmd.words:
        for p in w.parts:
            if p.is_arg_sentinel:
                return p
    return None


# ---------------------------------------------------------------------------
# ExpansionGoldenTest — pins heredoc/here-string/$() expansion output
# ---------------------------------------------------------------------------

EXPANSION_CASES = (
    ExpansionCase("unquoted_heredoc", "cat <<EOF\nhello\nworld\nEOF",
                  expected_heredoc_body="hello\nworld\n"),
    ExpansionCase("single_quoted_delimiter_no_expansion",
                  "cat <<'EOF'\n$(echo hi)\nEOF",
                  {"echo hi": "hi"}, expected_heredoc_body="$(echo hi)\n"),
    ExpansionCase("unquoted_heredoc_expands_dollar_paren",
                  "cat <<EOF\n$(echo hello)\nEOF",
                  {"echo hello": "hello"}, expected_heredoc_body="hello\n"),
    ExpansionCase("escaped_dollar_paren_in_heredoc_not_expanded",
                  "cat <<EOF\n\\$(echo hi)\nEOF",
                  {"echo hi": "hi"}, expected_heredoc_body="\\$(echo hi)\n"),
    ExpansionCase("heredoc_tab_strip", "cat <<-EOF\n\t\thello\n\tEOF",
                  expected_heredoc_body="hello\n",
                  expected_cleaned_starts_with="cat <<-"),
    ExpansionCase("herestring_unquoted", "cat <<<hello",
                  expected_heredoc_body="hello\n",
                  expected_cleaned_starts_with="cat <<<"),
    ExpansionCase("herestring_quoted", "cat <<<'hello world'",
                  expected_heredoc_body="hello world\n"),
    ExpansionCase("herestring_expands_dollar_paren_unless_single_quoted",
                  "cat <<<$(echo hi)",
                  {"echo hi": "hi"}, expected_heredoc_body="hi\n"),
    ExpansionCase("herestring_single_quoted_no_expansion",
                  "cat <<<'$(echo hi)'",
                  {"echo hi": "SHOULD_NOT"}, expected_heredoc_body="$(echo hi)\n"),
    ExpansionCase("command_substitution_sentinel", "echo $(echo hello)",
                  {"echo hello": "hello"}, expected_arg_value="hello",
                  expected_cleaned_starts_with="echo "),
    ExpansionCase("unbalanced_dollar_paren_error", "echo $(unclosed",
                  raises_with_substring="Unbalanced"),
    ExpansionCase("missing_heredoc_terminator_error",
                  "cat <<EOF\nhello\nworld\n",
                  raises_with_substring="not found"),
    ExpansionCase("quotes_inside_heredoc_body_preserved",
                  "cat <<EOF\nline with \"quotes\" and 'apostrophes'\nEOF",
                  expected_heredoc_body="line with \"quotes\" and 'apostrophes'\n"),
    ExpansionCase("double_quoted_delimiter_no_expansion",
                  'cat <<"EOF"\n$(echo hi)\nEOF',
                  {"echo hi": "hi"}, expected_heredoc_body="$(echo hi)\n"),
)


class ExpansionGoldenTest(unittest.TestCase):
    """Golden: heredoc/here-string/$() expansion via parse_command."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _stub_capture(self, outputs: Mapping[str, str]):
        def fake_capture(inner: str):
            val = outputs.get(inner, "")
            return 0, val.encode("utf-8")
        return fake_capture

    def _parse(self, cmd: str, outputs: Mapping[str, str] = ()):
        capture_fn = self._stub_capture(outputs)
        return parse_command(cmd, capture_fn, self.work_dir, 30, 0)

    def test_expansion_golden(self) -> None:
        for case in EXPANSION_CASES:
            with self.subTest(name=case.name, command=case.command):
                if case.raises_with_substring is not None:
                    with self.assertRaises(ValueError) as ctx:
                        self._parse(case.command, case.capture_outputs)
                    self.assertIn(case.raises_with_substring, str(ctx.exception))
                    continue

                expanded, exp, prog = self._parse(
                    case.command, case.capture_outputs,
                )
                if case.expected_cleaned_starts_with is not None:
                    self.assertTrue(
                        expanded.startswith(case.expected_cleaned_starts_with),
                        f"{case.command!r}: cleaned {expanded!r} should start "
                        f"with {case.expected_cleaned_starts_with!r}",
                    )
                if case.expected_heredoc_body is not None:
                    part = _find_hd_sentinel(prog)
                    self.assertIsNotNone(part, f"no hd sentinel for {case.command!r}")
                    self.assertEqual(exp.heredoc_for(part), case.expected_heredoc_body)
                if case.expected_arg_value is not None:
                    part = _find_arg_sentinel(prog)
                    self.assertIsNotNone(part, f"no arg sentinel for {case.command!r}")
                    self.assertEqual(exp.arg_for(part), case.expected_arg_value)

    def test_nested_command_substitution(self) -> None:
        # Pins what capture_fn RECEIVES (not parser output) — kept method-based.
        cmd = "echo $(echo $(echo inner))"
        captured = []

        def fake_capture(inner: str):
            captured.append(inner)
            return 0, inner.encode()

        expanded, exp, prog = parse_command(cmd, fake_capture, self.work_dir, 30, 0)
        self.assertIn("echo $(echo inner)", captured)


if __name__ == "__main__":
    unittest.main()
