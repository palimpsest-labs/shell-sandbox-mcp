"""Parser/AST shape tests — nested $(), heredoc bodies, sentinel emission,
lenient drop semantics, AST node construction."""

import unittest

from shell_sandbox_mcp.parser import (
    CommandNode,
    Expansion,
    PipelineNode,
    ProgramNode,
    Redirect,
    SENTINEL_ARG,
    SENTINEL_HD,
    Word,
    WordPart,
    extract_redirects,
    parse_command,
    split_legacy,
)


class SplitLegacyLenientTest(unittest.TestCase):
    """Test lenient drop semantics in split_legacy."""

    def test_double_semicolon_drops_empties(self) -> None:
        result = split_legacy("a ;; b")
        self.assertEqual(result, [(None, ["a"], False), (";", ["b"], False)])

    def test_triple_pipe_drops_empty_stage(self) -> None:
        result = split_legacy("a ||| b")
        self.assertEqual(result, [(None, ["a"], False), ("||", ["b"], False)])

    def test_leading_pipe_drops_empty(self) -> None:
        result = split_legacy("| ls")
        self.assertEqual(result, [(None, ["ls"], False)])

    def test_trailing_pipe_drops_empty(self) -> None:
        result = split_legacy("ls |")
        self.assertEqual(result, [(None, ["ls"], False)])

    def test_lone_semicolon_empty(self) -> None:
        self.assertEqual(split_legacy(";"), [])

    def test_empty_command(self) -> None:
        self.assertEqual(split_legacy(""), [])
        self.assertEqual(split_legacy("   "), [])

    def test_only_whitespace_between_operators(self) -> None:
        result = split_legacy("  a   ;;  b  ")
        self.assertEqual(result, [(None, ["a"], False), (";", ["b"], False)])


class SplitLegacyFdDupTest(unittest.TestCase):
    """Test that split_legacy correctly handles 2>&1 / 1>&2 as non-backgrounding."""

    def test_2gt1_not_background(self) -> None:
        result = split_legacy("echo hi 2>&1")
        self.assertEqual(result, [(None, ["echo hi 2>&1"], False)])

    def test_1gt2_not_background(self) -> None:
        result = split_legacy("cmd 1>&2")
        self.assertEqual(result, [(None, ["cmd 1>&2"], False)])

    def test_redirect_then_bare_ampersand(self) -> None:
        # 2>err & — the & after a space is backgrounding
        result = split_legacy("grep x 2>err &")
        self.assertEqual(result, [(None, ["grep x 2>err"], True)])


class ParseCommandSentinelTest(unittest.TestCase):
    """Test that parse_command emits correct sentinels."""

    def _parse(self, cmd: str) -> tuple[str, Expansion]:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, inner.encode()
        cleaned, expansion, _prog = parse_command(
            cmd, fake_capture, None, 30, 0,
        )
        return cleaned, expansion

    def test_subst_sentinel_id_increment(self) -> None:
        cleaned, exp = self._parse("echo $(a) $(b) $(c)")
        # Three sentinels with ids 0, 1, 2
        matches = SENTINEL_ARG.findall(cleaned)
        self.assertEqual(len(matches), 3)
        ids = [int(m) for m in matches]
        self.assertEqual(ids, [0, 1, 2])

    def test_heredoc_sentinel_id_increment(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b""
        cleaned, exp, _prog = parse_command(
            "cat <<A\nx\nA\n<<B\ny\nB", fake_capture, None, 30, 0,
        )
        matches = SENTINEL_HD.findall(cleaned)
        self.assertEqual(len(matches), 2)
        ids = [int(m) for m in matches]
        self.assertEqual(ids, [0, 1])

    def test_arg_sentinel_single_word(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"a b c"
        cleaned, exp, _prog = parse_command(
            "echo $(cmd)", fake_capture, None, 30, 0,
        )
        m = SENTINEL_ARG.search(cleaned)
        self.assertIsNotNone(m)
        sentinel = f"\x01A{m.group(1)}\x01"
        # Value has spaces but is stored as one word
        self.assertEqual(exp.arg_values[sentinel], "a b c")

    def test_compound_word_sentinel(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"b"
        cleaned, exp, _prog = parse_command(
            "echo a$(cmd)c", fake_capture, None, 30, 0,
        )
        # The sentinel is in the middle of the cleaned command
        self.assertIn("\x01A0\x01", cleaned)

    def test_cleaned_command_structure(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"result"
        cleaned, exp, _prog = parse_command(
            "echo $(cmd) > out.txt", fake_capture, None, 30, 0,
        )
        # The cleaned command should not contain $(...) or body text
        self.assertNotIn("$(cmd)", cleaned)
        self.assertIn("\x01A0\x01", cleaned)
        self.assertIn(">", cleaned)
        self.assertIn("out.txt", cleaned)


class ExtractRedirectsEdgeCasesTest(unittest.TestCase):
    """Additional edge cases for extract_redirects."""

    def test_glued_target_ok(self) -> None:
        args, redirs, err = extract_redirects(">out.txt echo hi")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out.txt")

    def test_2gt1x_not_fd_dup(self) -> None:
        args, redirs, err = extract_redirects("cmd 2>&1x")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 2)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "&1x")

    def test_1gt2y_not_fd_dup(self) -> None:
        args, redirs, err = extract_redirects("cmd 1>&2y")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 1)
        self.assertEqual(redirs[0].op, ">")
        self.assertEqual(redirs[0].raw_target, "&2y")

    def test_no_args_only_redirect(self) -> None:
        args, redirs, err = extract_redirects("> out.txt")
        self.assertIsNone(err)
        self.assertEqual(args, [])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out.txt")

    def test_multiple_redirects_mixed(self) -> None:
        args, redirs, err = extract_redirects("cmd 2>e 1>&2 > f")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 3)

    def test_input_redirect_glued(self) -> None:
        args, redirs, err = extract_redirects("cmd <file")
        self.assertIsNone(err)
        self.assertEqual(args, ["cmd"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].fd, 0)
        self.assertEqual(redirs[0].op, "<")
        self.assertEqual(redirs[0].raw_target, "file")

    def test_redirect_leading(self) -> None:
        args, redirs, err = extract_redirects(">out echo x")
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "x"])
        self.assertEqual(len(redirs), 1)


class ASTNodeConstructionTest(unittest.TestCase):
    """Test AST node construction and properties."""

    def test_empty_word_text(self) -> None:
        w = Word()
        self.assertEqual(w.text, "")

    def test_word_part_sentinel_detection(self) -> None:
        wp_arg = WordPart(text="\x01A0\x01", is_sentinel=True)
        wp_hd = WordPart(text="\x01H0\x01", is_sentinel=True)
        wp_normal = WordPart(text="hello", is_sentinel=False)

        self.assertTrue(wp_arg.is_arg_sentinel)
        self.assertFalse(wp_arg.is_hd_sentinel)
        self.assertTrue(wp_hd.is_hd_sentinel)
        self.assertFalse(wp_hd.is_arg_sentinel)
        self.assertFalse(wp_normal.is_arg_sentinel)
        self.assertFalse(wp_normal.is_hd_sentinel)

    def test_command_node_construction(self) -> None:
        cmd = CommandNode(
            words=(Word(parts=(WordPart(text="echo"),)),),
            redirects=(),
            backgrounded=False,
        )
        self.assertEqual(len(cmd.words), 1)
        self.assertEqual(cmd.words[0].text, "echo")

    def test_pipeline_node(self) -> None:
        p = PipelineNode(commands=(CommandNode(), CommandNode()))
        self.assertEqual(len(p.commands), 2)

    def test_program_node(self) -> None:
        from shell_sandbox_mcp.parser import AndOrNode
        prog = ProgramNode(chains=(
            AndOrNode(operator=None, pipeline=PipelineNode(), backgrounded=False),
        ))
        self.assertEqual(len(prog.chains), 1)

    def test_redirect_frozen(self) -> None:
        r = Redirect(fd=2, op=">&", target_fd=1, raw_target="1")
        self.assertEqual(r.fd, 2)
        self.assertEqual(r.op, ">&")
        with self.assertRaises(Exception):
            r.fd = 1  # frozen dataclass rejects attribute assignment

    def test_expansion_mutable(self) -> None:
        e = Expansion()
        e.arg_values["key"] = "val"
        e.heredoc_bodies["hd"] = "body"
        self.assertEqual(e.arg_values["key"], "val")
        self.assertEqual(e.heredoc_bodies["hd"], "body")


# ---------------------------------------------------------------------------
# REAL AST tests — verify parse_command returns a non-None ProgramNode
# ---------------------------------------------------------------------------

class RealASTFromParseCommandTest(unittest.TestCase):
    """Test that parse_command returns a real, populated ProgramNode."""

    def _parse(self, cmd, outputs=None):
        outputs = outputs or {}
        def fake_capture(inner):
            val = outputs.get(inner, "")
            return 0, val.encode()
        return parse_command(cmd, fake_capture, None, 30, 0)

    def test_simple_command_ast(self) -> None:
        cleaned, exp, prog = self._parse("echo hello")
        self.assertIsNotNone(prog)
        self.assertIsInstance(prog, ProgramNode)
        self.assertEqual(len(prog.chains), 1)
        chain = prog.chains[0]
        self.assertEqual(len(chain.pipeline.commands), 1)
        cmd = chain.pipeline.commands[0]
        self.assertEqual(len(cmd.words), 2)
        self.assertEqual(cmd.words[0].text, "echo")
        self.assertEqual(cmd.words[1].text, "hello")
        self.assertFalse(cmd.backgrounded)

    def test_command_with_subst_ast(self) -> None:
        cleaned, exp, prog = self._parse("echo $(whoami)", {"whoami": "root"})
        self.assertIsNotNone(prog)
        cmd = prog.chains[0].pipeline.commands[0]
        self.assertEqual(len(cmd.words), 2)
        self.assertEqual(cmd.words[0].text, "echo")
        # Second word is a sentinel
        self.assertTrue(cmd.words[1].parts[0].is_sentinel)
        self.assertTrue(cmd.words[1].parts[0].is_arg_sentinel)

    def test_command_with_redirect_ast(self) -> None:
        cleaned, exp, prog = self._parse("echo hi > out.txt")
        self.assertIsNotNone(prog)
        cmd = prog.chains[0].pipeline.commands[0]
        self.assertEqual(len(cmd.words), 2)
        self.assertEqual(cmd.words[0].text, "echo")
        self.assertEqual(cmd.words[1].text, "hi")
        self.assertEqual(len(cmd.redirects), 1)
        self.assertEqual(cmd.redirects[0].op, ">")
        self.assertEqual(cmd.redirects[0].target.text, "out.txt")

    def test_pipeline_ast(self) -> None:
        cleaned, exp, prog = self._parse("a | b | c")
        self.assertIsNotNone(prog)
        pipe = prog.chains[0].pipeline
        self.assertEqual(len(pipe.commands), 3)
        self.assertEqual(pipe.commands[0].words[0].text, "a")
        self.assertEqual(pipe.commands[1].words[0].text, "b")
        self.assertEqual(pipe.commands[2].words[0].text, "c")

    def test_compound_word_subst_ast(self) -> None:
        cleaned, exp, prog = self._parse("echo a$(cmd)b", {"cmd": "X"})
        self.assertIsNotNone(prog)
        cmd = prog.chains[0].pipeline.commands[0]
        # "echo" is word 1, "a"+sentinel+"b" is word 2
        self.assertEqual(len(cmd.words), 2)
        # Check the compound word
        w = cmd.words[1]
        self.assertEqual(len(w.parts), 3)
        self.assertEqual(w.parts[0].text, "a")
        self.assertTrue(w.parts[1].is_sentinel)
        self.assertEqual(w.parts[2].text, "b")

    def test_fd_dup_ast(self) -> None:
        cleaned, exp, prog = self._parse("cmd 2>&1")
        self.assertIsNotNone(prog)
        cmd = prog.chains[0].pipeline.commands[0]
        self.assertEqual(len(cmd.redirects), 1)
        self.assertEqual(cmd.redirects[0].op, ">&")
        self.assertEqual(cmd.redirects[0].fd, 2)
        self.assertEqual(cmd.redirects[0].target.text, "1")

    def test_background_ast(self) -> None:
        cleaned, exp, prog = self._parse("sleep 10 &")
        self.assertIsNotNone(prog)
        chain = prog.chains[0]
        self.assertTrue(chain.backgrounded)
        cmd = chain.pipeline.commands[0]
        self.assertEqual(cmd.words[0].text, "sleep")
        self.assertEqual(cmd.words[1].text, "10")

    def test_heredoc_ast_sentinel(self) -> None:
        cleaned, exp, prog = self._parse("cat <<EOF\nhello\nEOF")
        self.assertIsNotNone(prog)
        cmd = prog.chains[0].pipeline.commands[0]
        self.assertEqual(len(cmd.redirects), 1)
        self.assertEqual(cmd.redirects[0].op, "<<")
        self.assertTrue(cmd.redirects[0].target.parts[0].is_hd_sentinel)

    def test_extract_redirects_consumes_commandnode(self) -> None:
        """_extract_redirects accepts a CommandNode and projects correctly."""
        cleaned, exp, prog = self._parse("echo hi > out.txt")
        cmd = prog.chains[0].pipeline.commands[0]
        args, redirs, err = extract_redirects(cmd, expansion=exp)
        self.assertIsNone(err)
        self.assertEqual(args, ["echo", "hi"])
        self.assertEqual(len(redirs), 1)
        self.assertEqual(redirs[0].raw_target, "out.txt")


class SerializeProgramTest(unittest.TestCase):
    """Test serialize_program round-trips correctly."""

    def _parse(self, cmd, outputs=None):
        outputs = outputs or {}
        def fake_capture(inner):
            val = outputs.get(inner, "")
            return 0, val.encode()
        return parse_command(cmd, fake_capture, None, 30, 0)

    def test_simple_serialize(self) -> None:
        from shell_sandbox_mcp.parser import serialize_program
        cleaned, exp, prog = self._parse("echo hello")
        s = serialize_program(prog)
        self.assertEqual(s, "echo hello")

    def test_serialize_with_redirect(self) -> None:
        from shell_sandbox_mcp.parser import serialize_program
        cleaned, exp, prog = self._parse("echo hi > out.txt")
        s = serialize_program(prog)
        self.assertIn("echo hi", s)
        self.assertIn("> out.txt", s)

    def test_serialize_with_subst(self) -> None:
        from shell_sandbox_mcp.parser import serialize_program
        cleaned, exp, prog = self._parse("echo $(whoami)", {"whoami": "root"})
        s = serialize_program(prog)
        # Should contain sentinel
        self.assertIn("\x01A0\x01", s)

    def test_serialize_pipeline(self) -> None:
        from shell_sandbox_mcp.parser import serialize_program
        cleaned, exp, prog = self._parse("a | b | c")
        s = serialize_program(prog)
        self.assertEqual(s, "a | b | c")


if __name__ == "__main__":
    unittest.main()
