"""Parser/AST shape tests — nested $(), heredoc bodies, sentinel emission,
lenient drop semantics, AST node construction."""

import unittest

from shell_sandbox_mcp.parser import (
    CommandNode,
    Expansion,
    PipelineNode,
    ProgramNode,
    Redirect,
    Word,
    WordPart,
    extract_redirects,
    parse_command,
)


class ParseCommandSentinelTest(unittest.TestCase):
    """Test that parse_command emits correct sentinels."""

    def _parse(self, cmd: str):
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, inner.encode()
        cleaned, expansion, prog = parse_command(
            cmd, fake_capture, None, 30, 0,
        )
        return cleaned, expansion, prog

    def _all_arg_sentinels(self, prog):
        """Return all arg-sentinel WordParts from the first command."""
        cmd = prog.chains[0].pipeline.commands[0]
        result = []
        for w in cmd.words:
            for p in w.parts:
                if p.is_arg_sentinel:
                    result.append(p)
        return result

    def _all_hd_sentinels(self, prog):
        """Return all heredoc-sentinel WordParts from the first command."""
        cmd = prog.chains[0].pipeline.commands[0]
        result = []
        for rs in cmd.redirects:
            for p in rs.target.parts:
                if p.is_hd_sentinel:
                    result.append(p)
        return result

    def test_subst_sentinel_id_increment(self) -> None:
        cleaned, exp, prog = self._parse("echo $(a) $(b) $(c)")
        parts = self._all_arg_sentinels(prog)
        self.assertEqual(len(parts), 3)
        ids = []
        for p in parts:
            # Extract the numeric id from the sentinel text (internal format)
            val = exp.arg_for(p)
            self.assertIsNotNone(val)
            ids.append(p.text)
        # Verify three distinct sentinels (ids 0, 1, 2 by text content)
        self.assertEqual(len(set(ids)), 3)

    def test_heredoc_sentinel_id_increment(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b""
        cleaned, exp, prog = parse_command(
            "cat <<A\nx\nA\n<<B\ny\nB", fake_capture, None, 30, 0,
        )
        parts = self._all_hd_sentinels(prog)
        self.assertEqual(len(parts), 2)
        # Verify two distinct sentinels
        self.assertEqual(len(set(p.text for p in parts)), 2)

    def test_arg_sentinel_single_word(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"a b c"
        cleaned, exp, prog = parse_command(
            "echo $(cmd)", fake_capture, None, 30, 0,
        )
        parts = self._all_arg_sentinels(prog)
        self.assertEqual(len(parts), 1)
        # Value has spaces but is stored as one word
        self.assertEqual(exp.arg_for(parts[0]), "a b c")

    def test_compound_word_sentinel(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"b"
        cleaned, exp, prog = parse_command(
            "echo a$(cmd)c", fake_capture, None, 30, 0,
        )
        # There should be an arg sentinel in the middle of the word
        parts = self._all_arg_sentinels(prog)
        self.assertEqual(len(parts), 1)

    def test_cleaned_command_structure(self) -> None:
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"result"
        cleaned, exp, prog = parse_command(
            "echo $(cmd) > out.txt", fake_capture, None, 30, 0,
        )
        # The cleaned command should not contain $(...) or body text
        self.assertNotIn("$(cmd)", cleaned)
        parts = self._all_arg_sentinels(prog)
        self.assertEqual(len(parts), 1)
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
        # Get sentinel WordParts from real parse_command output instead of
        # constructing them with sentinel bytes.
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"arg_val"
        _, _, prog_arg = parse_command(
            "echo $(x)", fake_capture, None, 30, 0,
        )
        wp_arg = None
        for w in prog_arg.chains[0].pipeline.commands[0].words:
            for p in w.parts:
                if p.is_arg_sentinel:
                    wp_arg = p
                    break

        _, _, prog_hd = parse_command(
            "cat <<EOF\nx\nEOF",
            lambda i: (0, b""), None, 30, 0,
        )
        wp_hd = None
        for rs in prog_hd.chains[0].pipeline.commands[0].redirects:
            for p in rs.target.parts:
                if p.is_hd_sentinel:
                    wp_hd = p
                    break

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
        """Verify that Expansion stores values set via the internal write API
        and retrieved via the opaque lookup API."""
        def fake_capture(inner: str) -> tuple[int, bytes]:
            return 0, b"val"
        _, exp, prog = parse_command(
            "echo $(key)", fake_capture, None, 30, 0,
        )
        # The arg sentinel should be populated and retrievable
        part = None
        for w in prog.chains[0].pipeline.commands[0].words:
            for p in w.parts:
                if p.is_arg_sentinel:
                    part = p
                    break
        self.assertIsNotNone(part)
        self.assertEqual(exp.arg_for(part), "val")


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
        # Should contain a sentinel (not the resolved value)
        cmd = prog.chains[0].pipeline.commands[0]
        self.assertTrue(any(p.is_sentinel for w in cmd.words for p in w.parts))

    def test_serialize_pipeline(self) -> None:
        from shell_sandbox_mcp.parser import serialize_program
        cleaned, exp, prog = self._parse("a | b | c")
        s = serialize_program(prog)
        self.assertEqual(s, "a | b | c")


class CmdToDisplayTest(unittest.TestCase):
    """Test that cmd_to_display / _serialize_command produce human-readable strings."""

    def _parse(self, cmd, outputs=None, env=None):
        outputs = outputs or {}
        def fake_capture(inner):
            val = outputs.get(inner, "")
            return 0, val.encode()
        return parse_command(cmd, fake_capture, None, 30, 0, env=env)

    def _first_cmd(self, prog):
        for chain in prog.chains:
            for cmd in chain.pipeline.commands:
                return cmd
        return None

    def _assert_no_control_chars(self, s: str) -> None:
        """Assert a display string contains no control characters (like sentinels)."""
        for ch in s:
            self.assertTrue(
                ord(ch) >= 32 or ch in '\n\t',
                f"Display string contains control char 0x{ord(ch):02x}: {s!r}",
            )

    def test_display_renders_subst_no_sentinel(self) -> None:
        """cmd_to_display on a dq-embedded $(whoami) shows $(whoami), not a sentinel."""
        from shell_sandbox_mcp.parser import cmd_to_display
        _cleaned, _exp, prog = self._parse('echo "$(whoami)"', {"whoami": "root"})
        cmd = self._first_cmd(prog)
        display = cmd_to_display(cmd)
        self._assert_no_control_chars(display)
        self.assertIn("$(whoami)", display,
                      f"display must show the original $(whoami): {display!r}")

    def test_serialize_command_display_no_sentinel(self) -> None:
        """_serialize_command on a dq-embedded subst shows human-readable text."""
        from shell_sandbox_mcp.parser import _serialize_command
        _cleaned, _exp, prog = self._parse('echo "$(whoami)"', {"whoami": "root"})
        cmd = self._first_cmd(prog)
        display = _serialize_command(cmd)
        self._assert_no_control_chars(display)
        self.assertIn("$(whoami)", display)

    def test_display_multiple_subst(self) -> None:
        """Multiple dq-embedded substs all render as $(...) not sentinels."""
        from shell_sandbox_mcp.parser import _serialize_command
        _cleaned, _exp, prog = self._parse(
            'echo "$(echo a)$(echo b)"',
            {"echo a": "alpha", "echo b": "beta"},
        )
        cmd = self._first_cmd(prog)
        display = _serialize_command(cmd)
        self._assert_no_control_chars(display)
        self.assertIn("$(echo a)", display)
        self.assertIn("$(echo b)", display)

    def test_display_compound_dq(self) -> None:
        """Display of compound dq 'pre$(cmd)post' shows original text."""
        from shell_sandbox_mcp.parser import _serialize_command
        _cleaned, _exp, prog = self._parse(
            'echo "pre$(echo mid)post"',
            {"echo mid": "mid"},
        )
        cmd = self._first_cmd(prog)
        display = _serialize_command(cmd)
        self._assert_no_control_chars(display)
        self.assertIn("pre$(echo mid)post", display)

    def test_display_heredoc_shows_operator(self) -> None:
        """Heredoc operators display correctly (sentinel unavoidable for target)."""
        from shell_sandbox_mcp.parser import _serialize_command
        _cleaned, _exp, prog = self._parse("cat <<EOF\nhello\nEOF")
        cmd = self._first_cmd(prog)
        display = _serialize_command(cmd)
        # heredoc target is a sentinel (raw == sentinel), but operator is human-readable
        self.assertIn("<<", display)

    def test_display_simple_command(self) -> None:
        """Simple command without substs displays normally."""
        from shell_sandbox_mcp.parser import _serialize_command
        _cleaned, _exp, prog = self._parse("echo hello > out.txt")
        cmd = self._first_cmd(prog)
        display = _serialize_command(cmd)
        self.assertIn("echo hello", display)
        self.assertIn("> out.txt", display)
        self._assert_no_control_chars(display)


if __name__ == "__main__":
    unittest.main()
