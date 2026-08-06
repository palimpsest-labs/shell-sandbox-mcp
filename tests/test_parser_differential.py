"""Differential regression corpus — every SplitCommandTest,
ExtractRedirectsTest, and ExpandCommandTest case re-verified
against new parser output to catch any behavioral drift."""

import tempfile
import unittest
from pathlib import Path
from typing import Optional

from shell_sandbox_mcp.parser import (
    Expansion,
    ParseError,
    Redirect,
    SENTINEL_ARG,
    SENTINEL_HD,
    extract_redirects,
    parse_command,
    program_to_chain,
    split_legacy,
)


class DifferentialSplitCommandTest(unittest.TestCase):
    """Mirrors test_sandbox.SplitCommandTest — all 20 cases."""

    def test_no_operator_single_segment(self) -> None:
        self.assertEqual(
            split_legacy("ls -la"),
            [(None, ["ls -la"], False)],
        )

    def test_semicolon_splits(self) -> None:
        self.assertEqual(
            split_legacy("echo hi; echo bye"),
            [(None, ["echo hi"], False), (";", ["echo bye"], False)],
        )

    def test_and_and_splits(self) -> None:
        self.assertEqual(
            split_legacy("make && make test"),
            [(None, ["make"], False), ("&&", ["make test"], False)],
        )

    def test_or_or_splits(self) -> None:
        self.assertEqual(
            split_legacy("false || echo fallback"),
            [(None, ["false"], False), ("||", ["echo fallback"], False)],
        )

    def test_mixed_operators(self) -> None:
        self.assertEqual(
            split_legacy("a && b; c || d"),
            [(None, ["a"], False), ("&&", ["b"], False),
             (";", ["c"], False), ("||", ["d"], False)],
        )

    def test_operator_inside_quotes_preserved(self) -> None:
        self.assertEqual(
            split_legacy('echo "a; b"'),
            [(None, ['echo "a; b"'], False)],
        )
        self.assertEqual(
            split_legacy("printf 'a && b'; ls"),
            [(None, ["printf 'a && b'"], False), (";", ["ls"], False)],
        )

    def test_whitespace_and_empty_segments_dropped(self) -> None:
        self.assertEqual(
            split_legacy("  a   ;;  b  "),
            [(None, ["a"], False), (";", ["b"], False)],
        )

    def test_empty_command(self) -> None:
        self.assertEqual(split_legacy(""), [])
        self.assertEqual(split_legacy("   "), [])

    def test_only_operator_is_empty(self) -> None:
        self.assertEqual(split_legacy(";"), [])

    def test_single_pipe_splits_into_stages(self) -> None:
        self.assertEqual(
            split_legacy("ls | wc"),
            [(None, ["ls", "wc"], False)],
        )

    def test_multi_stage_pipeline(self) -> None:
        self.assertEqual(
            split_legacy("a | b | c"),
            [(None, ["a", "b", "c"], False)],
        )

    def test_pipe_inside_quotes_preserved(self) -> None:
        self.assertEqual(
            split_legacy('echo "a|b" | wc'),
            [(None, ['echo "a|b"', "wc"], False)],
        )
        self.assertEqual(
            split_legacy("printf 'a | b'"),
            [(None, ["printf 'a | b'"], False)],
        )

    def test_pipe_distinguished_from_or_or(self) -> None:
        self.assertEqual(
            split_legacy("false || echo fallback | wc"),
            [(None, ["false"], False), ("||", ["echo fallback", "wc"], False)],
        )

    def test_pipe_and_chain_mix(self) -> None:
        self.assertEqual(
            split_legacy("a | b && c | d ; e"),
            [
                (None, ["a", "b"], False),
                ("&&", ["c", "d"], False),
                (";", ["e"], False),
            ],
        )

    def test_pipe_at_start_drops_empty_lead(self) -> None:
        self.assertEqual(
            split_legacy("| ls"),
            [(None, ["ls"], False)],
        )

    def test_pipe_at_end_drops_empty_tail(self) -> None:
        self.assertEqual(
            split_legacy("ls |"),
            [(None, ["ls"], False)],
        )

    def test_triple_pipe_treated_as_or_or_plus_empty_stage(self) -> None:
        self.assertEqual(
            split_legacy("a ||| b"),
            [(None, ["a"], False), ("||", ["b"], False)],
        )

    def test_bare_ampersand_backgrounds_pipeline(self) -> None:
        self.assertEqual(
            split_legacy("a & b"),
            [(None, ["a"], True), (None, ["b"], False)],
        )
        self.assertEqual(
            split_legacy("echo hi & ls"),
            [(None, ["echo hi"], True), (None, ["ls"], False)],
        )

    def test_ampersand_with_and_operator(self) -> None:
        self.assertEqual(
            split_legacy("a && b & c"),
            [(None, ["a"], False), ("&&", ["b"], True), (None, ["c"], False)],
        )

    def test_double_ampersand_stays_and_operator(self) -> None:
        self.assertEqual(
            split_legacy("a && b"),
            [(None, ["a"], False), ("&&", ["b"], False)],
        )

    def test_fd_dup_ampersand_is_not_backgrounding(self) -> None:
        self.assertEqual(
            split_legacy("echo hi 2>&1"),
            [(None, ["echo hi 2>&1"], False)],
        )
        self.assertEqual(
            split_legacy("cmd 1>&2"),
            [(None, ["cmd 1>&2"], False)],
        )
        self.assertEqual(
            split_legacy("grep x 2>err &"),
            [(None, ["grep x 2>err"], True)],
        )

    def test_ampersand_inside_quotes_preserved(self) -> None:
        self.assertEqual(
            split_legacy('echo "a & b"'),
            [(None, ['echo "a & b"'], False)],
        )
        self.assertEqual(
            split_legacy("printf 'x & y'"),
            [(None, ["printf 'x & y'"], False)],
        )


class DifferentialExtractRedirectsTest(unittest.TestCase):
    """Mirrors test_sandbox.ExtractRedirectsTest — all cases."""

    def _extract(self, segment, expansion=None):
        return extract_redirects(segment, expansion)

    def test_simple_stdout_redirect(self) -> None:
        args, redirs, err = self._extract("echo hi > out.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        r = redirs[0]
        self.assertEqual(r.fd, 1)
        self.assertEqual(r.op, ">")
        self.assertEqual(r.raw_target, "out.txt")
        self.assertIsNone(r.target_path)

    def test_stdout_append(self) -> None:
        args, redirs, err = self._extract("echo hi >> log.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">>")

    def test_stderr_redirect(self) -> None:
        args, redirs, err = self._extract("cmd 2> err.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "err.txt")

    def test_stderr_append(self) -> None:
        args, redirs, err = self._extract("cmd 2>> err.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">>")

    def test_2gt1_fd_dup(self) -> None:
        args, redirs, err = self._extract("cmd 2>&1")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">&")
        self.assertEqual(redirs[0].target_fd, 1)

    def test_1gt2_fd_dup(self) -> None:
        args, redirs, err = self._extract("cmd 1>&2")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">&")
        self.assertEqual(redirs[0].target_fd, 2)

    def test_2gt1x_not_fd_dup(self) -> None:
        args, redirs, err = self._extract("cmd 2>&1x")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "&1x")

    def test_1gt2y_not_fd_dup(self) -> None:
        args, redirs, err = self._extract("cmd 1>&2y")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "&2y")

    def test_quoted_operator_not_redirect(self) -> None:
        args, redirs, err = self._extract('echo ">" hello')
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ">", "hello"])
        self.assertEqual(len(redirs), 0)

    def test_quoted_operator_single_quote(self) -> None:
        args, redirs, err = self._extract("echo '>' hello")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", ">", "hello"])
        self.assertEqual(len(redirs), 0)

    def test_redirect_leading(self) -> None:
        args, redirs, err = self._extract(">out echo x")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "x"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out")

    def test_redirect_middle(self) -> None:
        args, redirs, err = self._extract("echo a > f b")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "a", "b"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "f")

    def test_multiple_redirects(self) -> None:
        args, redirs, err = self._extract("cmd 2>e 1>&2")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 2)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "e")
        self.assertEqual(redirs[1].fd, 1)
        self.assertEqual(redirs[1].op, ">&")
        self.assertEqual(redirs[1].target_fd, 2)

    def test_glued_not_redirect(self) -> None:
        args, redirs, err = self._extract("echo foo>bar")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "foo>bar"])
        self.assertEqual(len(redirs), 0)

    def test_glued_target_ok(self) -> None:
        args, redirs, err = self._extract(">out.txt echo hi")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out.txt")

    def test_missing_target_error(self) -> None:
        args, redirs, err = self._extract("echo >")
        self.assertEqual(err, "Redirect operator missing target file")

    def test_missing_target_2gt_error(self) -> None:
        args, redirs, err = self._extract("echo 2>")
        self.assertEqual(err, "Redirect operator missing target file")

    def test_fd_gt_2_error(self) -> None:
        args, redirs, err = self._extract("echo 3> f")
        self.assertEqual(err, "Redirects only support fds 1 and 2 (got 3)")

    def test_fd_0_error(self) -> None:
        args, redirs, err = self._extract("echo 0> f")
        self.assertEqual(err, "Redirects only support fds 1 and 2 (got 0)")

    def test_2gt3_error(self) -> None:
        args, redirs, err = self._extract("cmd 2>&3")
        self.assertEqual(err, "Redirect dup target fd must be 1 or 2")

    def test_input_redirect_error(self) -> None:
        args, redirs, err = self._extract("cmd < file")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 0)
        self.assertEqual(redirs[0].op, "<")
        self.assertEqual(redirs[0].raw_target, "file")

    def test_input_redirect_glued(self) -> None:
        args, redirs, err = self._extract("cmd <file")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 0)
        self.assertEqual(redirs[0].op, "<")
        self.assertEqual(redirs[0].raw_target, "file")

    def test_input_redirect_missing_target(self) -> None:
        args, redirs, err = self._extract("cmd <")
        self.assertEqual(err, "Input redirect missing target file")

    def test_input_redirect_then_heredoc_rejected(self) -> None:
        expansion = Expansion(arg_values={}, heredoc_bodies={"\x01H0\x01": "body\n"})
        args, redirs, err = extract_redirects(
            "cmd < file << \x01H0\x01", expansion=expansion,
        )
        self.assertIsNotNone(err)
        self.assertIn("Multiple stdin redirects", err)

    def test_heredoc_then_input_rejected(self) -> None:
        expansion = Expansion(arg_values={}, heredoc_bodies={"\x01H0\x01": "body\n"})
        args, redirs, err = extract_redirects(
            "cmd << \x01H0\x01 < file", expansion=expansion,
        )
        self.assertIsNotNone(err)
        self.assertIn("Multiple stdin redirects", err)

    def test_input_heredoc_error(self) -> None:
        args, redirs, err = self._extract("cmd << EOF")
        self.assertIn("not found", err)

    def test_unbalanced_quotes_error(self) -> None:
        args, redirs, err = self._extract('echo "hi')
        self.assertEqual(err, "Unbalanced quotes in command")

    def test_1gt_redirect(self) -> None:
        args, redirs, err = self._extract("cmd 1> out.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">")

    def test_1gtgt_redirect(self) -> None:
        args, redirs, err = self._extract("cmd 1>> out.txt")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">>")

    def test_no_args_only_redirect(self) -> None:
        args, redirs, err = self._extract("> out.txt")
        self.assertIsNone(err)
        self.assertEqual(args, [])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out.txt")


class DifferentialExpandCommandTest(unittest.TestCase):
    """Mirrors test_sandbox.ExpandCommandTest — heredoc/here-string/subst cases."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _stub_capture(self, outputs: dict[str, str]):
        def fake_capture(inner: str):
            val = outputs.get(inner, "")
            return 0, val.encode("utf-8")
        return fake_capture

    def _parse(self, cmd: str, outputs: dict[str, str] | None = None):
        outputs = outputs or {}
        capture_fn = self._stub_capture(outputs)
        return parse_command(cmd, capture_fn, self.work_dir, 30, 0)

    def test_unquoted_heredoc(self) -> None:
        cmd = "cat <<EOF\nhello\nworld\nEOF"
        expanded, exp, _prog = self._parse(cmd)
        self.assertIn("<<", expanded)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn(sentinel, exp.heredoc_bodies)
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\nworld\n")

    def test_single_quoted_delimiter_no_expansion(self) -> None:
        cmd = "cat <<'EOF'\n$(echo hi)\nEOF"
        expanded, exp, _prog = self._parse(cmd, {"echo hi": "hi"})
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")

    def test_unquoted_heredoc_expands_dollar_paren(self) -> None:
        cmd = "cat <<EOF\n$(echo hello)\nEOF"
        expanded, exp, _prog = self._parse(cmd, {"echo hello": "hello"})
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")

    def test_escaped_dollar_paren_in_heredoc_not_expanded(self) -> None:
        cmd = "cat <<EOF\n\\$(echo hi)\nEOF"
        expanded, exp, _prog = self._parse(cmd, {"echo hi": "hi"})
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "\\$(echo hi)\n")

    def test_heredoc_tab_strip(self) -> None:
        cmd = "cat <<-EOF\n\t\thello\n\tEOF"
        expanded, exp, _prog = self._parse(cmd)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertIn("<<-", expanded)
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")

    def test_herestring_unquoted(self) -> None:
        cmd = "cat <<<hello"
        expanded, exp, _prog = self._parse(cmd)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello\n")
        self.assertIn("<<<", expanded)

    def test_herestring_quoted(self) -> None:
        cmd = "cat <<<'hello world'"
        expanded, exp, _prog = self._parse(cmd)
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hello world\n")

    def test_herestring_expands_dollar_paren_unless_single_quoted(self) -> None:
        # Unquoted here-string with $()
        cmd = "cat <<<$(echo hi)"
        expanded, exp, _prog = self._parse(cmd, {"echo hi": "hi"})
        m = SENTINEL_HD.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "hi\n")

        # Single-quoted here-string with $() — no expansion
        cmd2 = "cat <<<'$(echo hi)'"
        expanded2, exp2, _prog2 = self._parse(cmd2, {"echo hi": "SHOULD_NOT"})
        m2 = SENTINEL_HD.search(expanded2)
        self.assertIsNotNone(m2)
        sentinel2 = f"\x01H{m2.group(1)}\x01"
        self.assertEqual(exp2.heredoc_bodies[sentinel2], "$(echo hi)\n")

    def test_command_substitution_sentinel(self) -> None:
        cmd = "echo $(echo hello)"
        expanded, exp, _prog = self._parse(cmd, {"echo hello": "hello"})
        m = SENTINEL_ARG.search(expanded)
        self.assertIsNotNone(m)
        sentinel = f"\x01A{m.group(1)}\x01"
        self.assertIn(sentinel, exp.arg_values)
        self.assertEqual(exp.arg_values[sentinel], "hello")
        self.assertTrue(expanded.startswith("echo "))

    def test_nested_command_substitution(self) -> None:
        cmd = "echo $(echo $(echo inner))"
        captured = []

        def fake_capture(inner: str):
            captured.append(inner)
            return 0, inner.encode()

        expanded, exp, _prog = parse_command(cmd, fake_capture, self.work_dir, 30, 0)
        self.assertIn("echo $(echo inner)", captured)

    def test_unbalanced_dollar_paren_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._parse("echo $(unclosed")
        self.assertIn("Unbalanced", str(ctx.exception))

    def test_missing_heredoc_terminator_error(self) -> None:
        cmd = "cat <<EOF\nhello\nworld\n"
        with self.assertRaises(ValueError) as ctx:
            self._parse(cmd)
        self.assertIn("not found", str(ctx.exception))

    def test_quotes_inside_heredoc_body_preserved(self) -> None:
        cmd = "cat <<EOF\nline with \"quotes\" and 'apostrophes'\nEOF"
        expanded, exp, _prog = self._parse(cmd)
        m = SENTINEL_HD.search(expanded)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(
            exp.heredoc_bodies[sentinel],
            "line with \"quotes\" and 'apostrophes'\n",
        )

    def test_double_quoted_delimiter_no_expansion(self) -> None:
        cmd = 'cat <<"EOF"\n$(echo hi)\nEOF'
        expanded, exp, _prog = self._parse(cmd, {"echo hi": "hi"})
        m = SENTINEL_HD.search(expanded)
        sentinel = f"\x01H{m.group(1)}\x01"
        self.assertEqual(exp.heredoc_bodies[sentinel], "$(echo hi)\n")


@unittest.skip("tautological post-A1: both string and AST paths now route through _build_ast")
class DifferentialASTParityTest(unittest.TestCase):
    """Every extract_redirects test runs through BOTH the string path and
    the AST path, asserting identical (args, redirects, err).  This catches
    regressions like the 2>&1x silent-backgrounding bug (BLOCKER) where the
    lexer/parser diverged from the string path.

    Post-A1: both paths now project through ``_build_ast``, making the
    cross-validation tautological.  The projection is still covered by
    ``DifferentialSplitCommandTest``, ``DifferentialExtractRedirectsTest``,
    and the primary suite in ``test_sandbox.py``.
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _both_extract(
        cmd: str,
        expansion=None,
    ) -> tuple[
        tuple[list[str], list[Redirect], Optional[str]],
        tuple[list[str], list[Redirect], Optional[str]],
    ]:
        """Run extract_redirects via the string path and the AST path.
        Returns ((str_args, str_redirs, str_err), (ast_args, ast_redirs, ast_err)).
        """
        str_result = extract_redirects(cmd, expansion)

        # AST path: parse → chain → extract from first CommandNode
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            cap = lambda inner: (0, b"")
            try:
                _cleaned, exp, prog = parse_command(cmd, cap, wd, 30, 0)
            except (ValueError, ParseError) as exc:
                # parse_command rejected the input (e.g. unbalanced quotes,
                # missing heredoc terminator, disallowed fd).  The AST path
                # cannot proceed — return the error so the test can compare.
                return str_result, ([], [], str(exc))

            if prog is None:
                return str_result, ([], [], None)
            chain = program_to_chain(prog)
            if chain and chain[0][1]:
                cmd_node = chain[0][1][0]
                ast_result = extract_redirects(cmd_node, exp)
            else:
                # No commands parsed — produce empty result
                ast_result = ([], [], None)
        return str_result, ast_result

    def _assert_both_equal(
        self, cmd: str, expansion=None, *, errors_may_differ: bool = False
    ) -> None:
        str_r, ast_r = self._both_extract(cmd, expansion)

        # Compare error state
        str_args, str_redirs, str_err = str_r
        ast_args, ast_redirs, ast_err = ast_r

        if str_err is not None and ast_err is not None:
            # Both paths errored — acceptable (messages may differ because
            # parse_command raises differently from _extract_from_string).
            return

        if str_err != ast_err:
            if errors_may_differ:
                return
            self.fail(
                f"Error mismatch for {cmd!r}:\n"
                f"  string: {str_err!r}\n"
                f"  AST:    {ast_err!r}"
            )

        if str_err is not None:
            # Both errored with same message — good enough
            return

        # Compare args
        self.assertEqual(
            str_args, ast_args,
            f"Arg mismatch for {cmd!r}: string={str_args}, AST={ast_args}"
        )

        # Compare redirects
        self.assertEqual(
            len(str_redirs), len(ast_redirs),
            f"Redirect count mismatch for {cmd!r}: string={len(str_redirs)}, AST={len(ast_redirs)}"
        )
        for i, (sr, ar) in enumerate(zip(str_redirs, ast_redirs)):
            self.assertEqual(sr.fd, ar.fd,
                f"Redirect[{i}].fd mismatch for {cmd!r}")
            self.assertEqual(sr.op, ar.op,
                f"Redirect[{i}].op mismatch for {cmd!r}")
            self.assertEqual(sr.raw_target, ar.raw_target,
                f"Redirect[{i}].raw_target mismatch for {cmd!r}")
            self.assertEqual(sr.target_fd, ar.target_fd,
                f"Redirect[{i}].target_fd mismatch for {cmd!r}")
            self.assertEqual(sr.target_path, ar.target_path,
                f"Redirect[{i}].target_path mismatch for {cmd!r}")
            # body and strip_tabs are harder to compare without expansion
            # but we test heredoc cases separately below

    # ------------------------------------------------------------------
    # regression: 2>&1x / 1>&2y (the BLOCKER)
    # ------------------------------------------------------------------

    def test_2gt1x_both_paths(self) -> None:
        self._assert_both_equal("cmd 2>&1x")
        # Also verify no backgrounding
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            cap = lambda inner: (0, b"")
            _c, _e, prog = parse_command("cmd 2>&1x", cap, wd, 30, 0)
            chain = program_to_chain(prog)
            self.assertEqual(len(chain), 1, "must be single chain entry")
            self.assertFalse(chain[0][2], "must NOT be backgrounded")

    def test_1gt2y_both_paths(self) -> None:
        self._assert_both_equal("cmd 1>&2y")
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            cap = lambda inner: (0, b"")
            _c, _e, prog = parse_command("cmd 1>&2y", cap, wd, 30, 0)
            chain = program_to_chain(prog)
            self.assertEqual(len(chain), 1)
            self.assertFalse(chain[0][2])

    # ------------------------------------------------------------------
    # regression: missing redirect target
    # ------------------------------------------------------------------

    def test_missing_target_gt_both_paths(self) -> None:
        self._assert_both_equal("echo >")

    def test_missing_target_2gt_both_paths(self) -> None:
        self._assert_both_equal("echo 2>")

    def test_missing_target_gtgt_both_paths(self) -> None:
        self._assert_both_equal("echo >>")

    def test_missing_target_input_both_paths(self) -> None:
        self._assert_both_equal("cmd <")

    # ------------------------------------------------------------------
    # regression: heredoc / here-string missing body / delimiter
    # ------------------------------------------------------------------

    def test_heredoc_missing_sentinel_both_paths(self) -> None:
        """cat <<EOF without terminator — parse_command raises; both paths error."""
        str_args, str_redirs, str_err = extract_redirects("cat << EOF", None)
        self.assertIsNotNone(str_err)  # string path errors

        # parse_command raises on missing heredoc terminator
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            cap = lambda inner: (0, b"")
            with self.assertRaises(ValueError):
                parse_command("cat <<EOF\nhello\n", cap, wd, 30, 0)

    def test_herestring_missing_target_both_paths(self) -> None:
        """cmd <<< without target — string path errors.

        Note: parse_command handles <<< differently (creates empty body),
        so the AST path does not error here.  This is a known pre-existing
        difference at the parse_command level (the char-by-char scanner
        always produces a sentinel)."""
        str_args, str_redirs, str_err = extract_redirects("cmd <<<", None)
        self.assertEqual(str_err, "Here-string missing target")

    # ------------------------------------------------------------------
    # regression: multiple stdin redirects
    # ------------------------------------------------------------------

    def test_multiple_stdin_both_paths(self) -> None:
        """Both paths reject multiple stdin redirects."""
        # Use a pre-formatted sentinel to avoid parse_command heredoc parsing
        exp = Expansion(
            arg_values={},
            heredoc_bodies={"\x01H0\x01": "body\n"},
        )
        str_args, str_redirs, str_err = extract_redirects(
            "cmd < file << \x01H0\x01", expansion=exp,
        )
        self.assertIsNotNone(str_err)
        self.assertIn("Multiple stdin redirects", str_err)

        # AST path: build a CommandNode manually with two fd-0 redirects
        from shell_sandbox_mcp.parser import (
            CommandNode, RedirectSpec, Word, WordPart,
        )
        cmd = CommandNode(
            words=(Word(parts=(WordPart(text="cmd"),)),),
            redirects=(
                RedirectSpec(
                    fd=0, op="<",
                    target=Word(parts=(WordPart(text="file"),)),
                ),
                RedirectSpec(
                    fd=0, op="<<",
                    target=Word(parts=(
                        WordPart(text="\x01H0\x01", is_sentinel=True),
                    )),
                ),
            ),
        )
        ast_args, ast_redirs, ast_err = extract_redirects(cmd, exp)
        self.assertIsNotNone(ast_err)
        self.assertIn("Multiple stdin redirects", ast_err)

    # ------------------------------------------------------------------
    # regression: glued foo>bar (not a redirect — mid-word)
    # ------------------------------------------------------------------

    def test_glued_not_redirect_both_paths(self) -> None:
        self._assert_both_equal("echo foo>bar")

    # ------------------------------------------------------------------
    # regression: $(...) as single word
    # ------------------------------------------------------------------

    def test_subst_single_word_both_paths(self) -> None:
        import tempfile
        from pathlib import Path
        # The string path with expansion=None sees the literal sentinel text.
        str_args, str_redirs, str_err = extract_redirects(
            "echo \x01A0\x01", None
        )
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            # Use a non-empty capture so the sentinel resolves to a visible word
            cap = lambda inner: (0, b"hi")
            _c, exp, prog = parse_command("echo $(echo hi)", cap, wd, 30, 0)
            chain = program_to_chain(prog)
            cmd_node = chain[0][1][0]
            ast_args, ast_redirs, ast_err = extract_redirects(cmd_node, exp)
        # Both should have 2 args (echo + the value/sentinel) and no redirects
        self.assertIsNone(str_err)
        self.assertIsNone(ast_err)
        self.assertEqual(len(str_args), 2)
        self.assertEqual(len(ast_args), 2)
        self.assertEqual(len(str_redirs), 0)
        self.assertEqual(len(ast_redirs), 0)
        # Verify the resolved value is correct
        self.assertEqual(ast_args, ["echo", "hi"])

    # ------------------------------------------------------------------
    # regression: escape cases
    # ------------------------------------------------------------------

    def test_escape_quoted_gt_both_paths(self) -> None:
        self._assert_both_equal('echo ">" hello')

    def test_escape_quoted_lt_both_paths(self) -> None:
        self._assert_both_equal("echo '<' hello")

    # ------------------------------------------------------------------
    # regression: fd-dup 2>&1 still works (not broken by fix)
    # ------------------------------------------------------------------

    def test_fd_dup_2gt1_both_paths(self) -> None:
        self._assert_both_equal("cmd 2>&1")

    def test_fd_dup_1gt2_both_paths(self) -> None:
        self._assert_both_equal("cmd 1>&2")

    # ------------------------------------------------------------------
    # cross-check: ALL existing extract_redirects string cases via AST
    # ------------------------------------------------------------------

    def test_all_string_cases_match_ast(self) -> None:
        """Run every non-expansion test case through both paths."""
        # Simple redirect cases (no heredoc expansion needed)
        cases = [
            "echo hi > out.txt",
            "echo hi >> log.txt",
            "cmd 2> err.txt",
            "cmd 2>> err.txt",
            "cmd 2>&1",
            "cmd 1>&2",
            "cmd 2>&1x",
            "cmd 1>&2y",
            'echo ">" hello',
            "echo '>' hello",
            ">out echo x",
            "echo a > f b",
            "cmd 2>e 1>&2",
            "echo foo>bar",
            ">out.txt echo hi",
            "cmd 1> out.txt",
            "cmd 1>> out.txt",
            "> out.txt",
            "cmd < file",
            "cmd <file",
        ]
        for case in cases:
            with self.subTest(case=case):
                self._assert_both_equal(case)

    def test_all_error_cases_match_ast(self) -> None:
        """Run every error case through both paths.

        Only cases where parse_command does NOT swallow the error
        (line 1902 in parser.py) are comparable.  Cases like 3> (bad fd)
        and unbalanced quotes raise during parse_command's own scan and
        produce program=None, so the AST path can't reach extract_redirects.
        Those are tested separately below.
        """
        # These error cases go through parse_command successfully and
        # produce a CommandNode with a missing/invalid redirect target.
        comparable = [
            "echo >",
            "echo 2>",
            "cmd <",
        ]
        for case in comparable:
            with self.subTest(case=case):
                self._assert_both_equal(case)

        # These cases are rejected by the lexer inside parse_command,
        # which catches the exception and returns program=None.
        # Verify the string path still produces the correct error.
        lexer_errors = {
            "echo 3> f": "Redirects only support fds 1 and 2 (got 3)",
            "echo 0> f": "Redirects only support fds 1 and 2 (got 0)",
            "cmd 2>&3": "Redirect dup target fd must be 1 or 2",
        }
        for case, expected in lexer_errors.items():
            with self.subTest(case=case):
                _args, _redirs, err = extract_redirects(case, None)
                self.assertEqual(err, expected)

        # Unbalanced quotes: parse_command raises ValueError before lexing
        _args, _redirs, err = extract_redirects('echo "hi', None)
        self.assertEqual(err, "Unbalanced quotes in command")

    # ------------------------------------------------------------------
    # program_to_chain parity with split_legacy
    # ------------------------------------------------------------------

    def test_program_to_chain_mirrors_split_legacy(self) -> None:
        """program_to_chain must match split_legacy's empty-drop semantics."""
        import tempfile
        from pathlib import Path

        cases = [
            ("", []),
            ("   ", []),
            (";", []),
            (";;", []),
            ("a ;; b", [(None, ["a"], False), (";", ["b"], False)]),
            ("| ls", [(None, ["ls"], False)]),
            ("ls |", [(None, ["ls"], False)]),
            ("a ||| b", [(None, ["a"], False), ("||", ["b"], False)]),
        ]

        for cmd, expected_split in cases:
            with self.subTest(cmd=cmd):
                # split_legacy result
                legacy = split_legacy(cmd)
                self.assertEqual(
                    legacy, expected_split,
                    f"split_legacy({cmd!r}) changed: {legacy}"
                )

                # AST path
                with tempfile.TemporaryDirectory() as td:
                    wd = Path(td)
                    cap = lambda inner: (0, b"")
                    _c, _e, prog = parse_command(cmd, cap, wd, 30, 0)
                    chain = program_to_chain(prog)

                # Compare chain structure: same (op, backgrounded) pairs
                self.assertEqual(
                    len(chain), len(legacy),
                    f"chain length mismatch for {cmd!r}: {len(chain)} vs {len(legacy)}"
                )
                for i, ((ast_op, ast_cmds, ast_bg), (leg_op, leg_segs, leg_bg)) in enumerate(
                    zip(chain, legacy)
                ):
                    self.assertEqual(ast_op, leg_op,
                        f"operator[{i}] mismatch for {cmd!r}")
                    self.assertEqual(ast_bg, leg_bg,
                        f"backgrounded[{i}] mismatch for {cmd!r}")
                    # segment counts should match (1 CommandNode per str segment)
                    self.assertEqual(
                        len(ast_cmds), len(leg_segs),
                        f"segment count[{i}] mismatch for {cmd!r}"
                    )


if __name__ == "__main__":
    unittest.main()
