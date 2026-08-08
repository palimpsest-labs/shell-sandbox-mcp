"""Tests for glob/pathname expansion in command args.

Unquoted ``*``, ``?`` and ``[...]`` in COMMAND ARGS expand against the
filesystem (``work_dir``) like a real shell.  Quoted metacharacters are
literal; unmatched globs stay literal; matches are filtered by path
containment.  Redirect targets are never globbed.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shell_sandbox_mcp import containment
from shell_sandbox_mcp.config import EXTRA_REDIRECT_ROOTS, MAX_ARGS
from shell_sandbox_mcp.parser import (
    Expansion,
    ParseError,
    extract_redirects,
    parse_command,
    program_to_chain,
)


def _parse(cmd: str, wd: Path, env=None, outputs=None):
    """Run parse_command against *wd* and return (cleaned, exp, prog)."""
    outputs = outputs or {}
    cap = lambda inner: (0, outputs.get(inner, "").encode("utf-8"))
    return parse_command(cmd, cap, wd, 30, 0, env=env or {})


def _assert_contained(test, paths, work_dir):
    """Assert every path stays within {work_dir} + extra redirect roots."""
    roots = [work_dir, *EXTRA_REDIRECT_ROOTS]
    for p in paths:
        test.assertIsNotNone(
            containment._contained_in_any(p, roots),
            f"{p!r} escapes all allowed roots",
        )


class GlobPlainTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.wd = Path(self._td.name)
        (self.wd / "a.py").write_text("")
        (self.wd / "b.py").write_text("")
        (self.wd / "c.txt").write_text("")
        (self.wd / "file1").write_text("")
        (self.wd / "file2").write_text("")
        (self.wd / ".hidden").write_text("")

    def tearDown(self):
        self._td.cleanup()

    def test_star_expands_to_matches(self):
        args, redirs, err = extract_redirects("echo *.py", None, self.wd)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", str(self.wd / "a.py"), str(self.wd / "b.py")])

    def test_quoted_star_is_literal(self):
        args, redirs, err = extract_redirects('echo "*.py"', None, self.wd)
        self.assertEqual(args, ["echo", "*.py"])

    def test_glob_char_only_in_quoted_part(self):
        args, redirs, err = extract_redirects('echo a"*"b', None, self.wd)
        self.assertEqual(args, ["echo", "a*b"])

    def test_question_mark_class(self):
        (self.wd / "f1.txt").write_text("")
        (self.wd / "f2.txt").write_text("")
        (self.wd / "g1.txt").write_text("")
        args, redirs, err = extract_redirects("echo f?.txt", None, self.wd)
        self.assertEqual(
            args, ["echo", str(self.wd / "f1.txt"), str(self.wd / "f2.txt")]
        )
        args, redirs, err = extract_redirects("echo [fg]1.txt", None, self.wd)
        self.assertEqual(
            args, ["echo", str(self.wd / "f1.txt"), str(self.wd / "g1.txt")]
        )

    def test_no_match_stays_literal(self):
        args, redirs, err = extract_redirects("echo nope*.py", None, self.wd)
        self.assertEqual(args, ["echo", "nope*.py"])

    def test_multiple_matches_sorted(self):
        args, redirs, err = extract_redirects("echo *", None, self.wd)
        # `*` matches non-hidden files only, sorted. "echo" has no magic so
        # stays literal.
        paths = sorted(str(self.wd / n) for n in ["a.py", "b.py", "c.txt", "file1", "file2"])
        self.assertEqual(args, ["echo", *paths])

    def test_dotfiles_not_matched(self):
        args, redirs, err = extract_redirects("echo *", None, self.wd)
        self.assertNotIn(str(self.wd / ".hidden"), args)

    def test_no_work_dir_means_no_globbing(self):
        # With work_dir=None (default), glob chars stay literal.
        args, redirs, err = extract_redirects("echo *.py", None)
        self.assertEqual(args, ["echo", "*.py"])

    def test_redirect_target_not_globbed(self):
        args, redirs, err = extract_redirects("echo hi > *.txt", None, self.wd)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "*.txt")


class GlobSecurityTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.wd = Path(self._td.name)
        (self.wd / "keep.py").write_text("")

    def tearDown(self):
        self._td.cleanup()

    def test_symlink_escape_filtered_to_literal(self):
        # A symlink pointing outside the working tree: globbing through it
        # yields matches outside all allowed roots, which are dropped → literal.
        try:
            os.symlink("/etc", self.wd / "escape")
        except OSError:
            self.skipTest("cannot create symlink in this environment")
        args, redirs, err = extract_redirects("echo escape/*", None, self.wd)
        self.assertEqual(args, ["echo", "escape/*"])

    def test_all_matches_stay_contained(self):
        (self.wd / "sub").mkdir()
        (self.wd / "sub" / "x.c").write_text("")
        args, redirs, err = extract_redirects("echo **", None, self.wd)
        self.assertIsNone(err)
        # ** matches nested paths; every returned match must be contained.
        _assert_contained(self, args[1:], self.wd)


class GlobExpansionTest(unittest.TestCase):
    """$VAR / $() results are subject to pathname expansion when unquoted."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.wd = Path(self._td.name)
        (self.wd / "data1.c").write_text("")
        (self.wd / "data2.c").write_text("")

    def tearDown(self):
        self._td.cleanup()

    def test_unquoted_var_result_globbed(self):
        env = {"PAT": "*.c"}
        cleaned, exp, prog = _parse("echo $PAT", self.wd, env=env)
        args, redirs, err = extract_redirects(cleaned, exp, self.wd)
        self.assertEqual(
            args, ["echo", str(self.wd / "data1.c"), str(self.wd / "data2.c")]
        )

    def test_quoted_var_result_not_globbed(self):
        env = {"PAT": "*.c"}
        cleaned, exp, prog = _parse('echo "$PAT"', self.wd, env=env)
        args, redirs, err = extract_redirects(cleaned, exp, self.wd)
        self.assertEqual(args, ["echo", "*.c"])

    def test_unquoted_subst_result_globbed(self):
        # $() output "*.c" (unquoted) is subject to pathname expansion.
        outputs = {'echo "*.c"': "*.c"}
        cleaned, exp, prog = _parse('echo $(echo "*.c")', self.wd, outputs=outputs)
        args, redirs, err = extract_redirects(cleaned, exp, self.wd)
        self.assertEqual(
            args, ["echo", str(self.wd / "data1.c"), str(self.wd / "data2.c")]
        )


class GlobAbsoluteTest(unittest.TestCase):
    """Absolute patterns resolve against /tmp (an extra redirect root)."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory(dir="/tmp")
        self.abs = Path(self._td.name)
        (self.abs / "alpha.log").write_text("")
        (self.abs / "beta.log").write_text("")

    def tearDown(self):
        self._td.cleanup()

    def test_absolute_tmp_glob(self):
        # work_dir can be anything; the absolute pattern targets the /tmp dir.
        with tempfile.TemporaryDirectory() as wd:
            args, redirs, err = extract_redirects(
                f"echo {self.abs}/*.log", None, Path(wd)
            )
        self.assertEqual(
            args,
            ["echo", str(self.abs / "alpha.log"), str(self.abs / "beta.log")],
        )


class MaxArgsTest(unittest.TestCase):
    """MAX_ARGS ceiling on argv entries (defense-in-depth)."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.wd = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_max_args_exceeded_glob(self):
        # A glob fanning out to MAX_ARGS+1 matches is an explicit error.
        big = [f"f{i}" for i in range(MAX_ARGS + 1)]
        with mock.patch(
            "shell_sandbox_mcp.parser._expand_glob_arg", return_value=big
        ):
            args, redirs, err = extract_redirects("echo *", None, self.wd)
        self.assertEqual(args, [])
        self.assertIn(f"Argument list too long (max {MAX_ARGS})", err or "")

    def test_max_args_exceeded_literal(self):
        # >MAX_ARGS literal argv entries → explicit error, not truncation.
        words = " ".join(f"w{i}" for i in range(MAX_ARGS + 1))
        args, redirs, err = extract_redirects(f"echo {words}", None, self.wd)
        self.assertEqual(args, [])
        self.assertIn(f"Argument list too long (max {MAX_ARGS})", err or "")

    def test_max_args_exactly(self):
        # Exactly MAX_ARGS argv entries succeed.
        words = " ".join(f"w{i}" for i in range(MAX_ARGS - 1))
        args, redirs, err = extract_redirects(f"echo {words}", None, self.wd)
        self.assertIsNone(err)
        self.assertEqual(len(args), MAX_ARGS)


if __name__ == "__main__":
    unittest.main()