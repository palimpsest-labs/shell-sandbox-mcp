"""Property-style tests for the path-containment helpers in
``shell_sandbox_mcp.containment``.

These complement the example tests in ``tests/test_containment.py`` (which
exercise specific hand-picked cases) by running *many* deterministic random
inputs and asserting invariants rather than exact outputs.

This file imports only ``shell_sandbox_mcp.containment`` (+ ``parser.Redirect``
and ``config``) — deliberately NOT ``server`` — so it stays runnable under the
sandboxed musl python without needing ``mcp``.

Runs in ~2-3s. All randomness is seeded (``random.Random(0xC3)``) for
reproducibility.
"""

import os
import random
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from shell_sandbox_mcp.config import DEFAULT_ALLOWED_DIRS
from shell_sandbox_mcp.containment import (
    _binary_still_contained,
    _contained_in_any,
    _contained_path,
    _validate_cwd,
    _validate_redirect_paths,
)
from shell_sandbox_mcp.parser import Redirect

# Single module-level RNG so the whole file is reproducible from one seed.
_RNG = random.Random(0xC3)

# A pool of path segment names. Most exist inside the test topology; a few
# ("z", "w", "x", "nope") intentionally don't, to exercise non-existent paths.
_SEGS = ["bin", "tool", "a", "b", "c", "plain", "notexec", "lnk_ok",
         "lnk_esc", "z", "w", "x", "nope"]

# For non-ASCII / shell-char raw inputs in the cwd-message test.
_RAW_INPUTS = [
    "~user typed /path with spaces",
    "h\u00e9llo w\u00f6rld /tmp",
    "a&b|c;d>e <f>",
    "sym $VAR (parens) 'q'",
    "\u00dftr\u00e4ng\u0117 \u5b57\u7b26",
]


def _rand_rel(rng: random.Random) -> str:
    """A random relative path (no leading ``./`` or ``..``)."""
    n = rng.randint(1, 4)
    return "/".join(rng.choice(_SEGS) for _ in range(n))


def _make_exec(path: Path) -> None:
    path.write_text("#!/bin/sh\necho hi\n")
    path.chmod(0o755)


def _state_at(events: list[tuple[int, bool]], version: int) -> bool:
    """State (REAL if True) for a given event version, walking event history.

    ``version`` is the number of completed swaps: it counts the events, so the
    state after ``version`` swaps is the state of the swap at index
    ``version - 1`` (i.e. the latest event with index ``v < version``). State
    before any swap (version 0) is REAL (the initial state).
    """
    real = True
    for v, s in events:
        if v < version:
            real = s
        else:
            break
    return real


# ---------------------------------------------------------------------------
# _contained_path
# ---------------------------------------------------------------------------


class ContainedPathPropertyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()  # resolved: /tmp->/private/tmp safe
        # tree
        (self.root / "a" / "b" / "c").mkdir(parents=True)
        (self.root / "bin").mkdir()
        _make_exec(self.root / "bin" / "tool")
        _make_exec(self.root / "tool")
        # a few non-executable files
        (self.root / "plain").write_text("data")
        (self.root / "plain").chmod(0o644)
        (self.root / "a" / "notexec").write_text("data")
        (self.root / "a" / "notexec").chmod(0o644)
        # symlinks
        (self.root / "lnk_esc").symlink_to("/etc")          # escape
        (self.root / "lnk_ok").symlink_to(self.root / "bin" / "tool")  # intra-tree

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _rand_input(self) -> str:
        """A random input string mixing interesting forms."""
        mode = _RNG.randint(0, 6)
        rel = _rand_rel(_RNG)
        if mode == 0:
            return rel
        if mode == 1:
            return "./" + rel
        if mode == 2:
            return "../" * _RNG.randint(1, 3) + rel
        if mode == 3:
            return str(self.root / rel)
        if mode == 4:
            return rel + "/"
        if mode == 5:
            return "/etc/" + rel            # absolute, outside root
        return str(self.root / ".." / rel)  # absolute-ish that escapes via parent

    def test_strict_containment_invariant(self) -> None:
        # If the function returns a path, it must (a) be inside root and
        # (b) already be in fully-resolved (canonical) form.
        for _ in range(100):
            got = _contained_path(self._rand_input(), self.root)
            if got is not None:
                got.relative_to(self.root)          # must not raise
                self.assertEqual(got, got.resolve())

    def test_idempotence_invariant(self) -> None:
        # Feeding back the returned path must produce the identical path.
        for _ in range(100):
            got = _contained_path(self._rand_input(), self.root)
            if got is not None:
                self.assertEqual(
                    _contained_path(str(got), self.root), got
                )

    def test_dotdot_traversal_classification(self) -> None:
        # For arbitrary `..`/`.`-laden sequences, the function's verdict must
        # match an independent pathlib resolve + containment classification.
        for _ in range(100):
            parts = [
                _RNG.choice(["a", "b", "c", "tool", "bin", "plain", "..", "."])
                for _ in range(_RNG.randint(1, 6))
            ]
            s = "/".join(parts)
            try:
                expected = (self.root / s).resolve()
            except OSError:
                continue  # defensive; our topology has no loops
            try:
                expected.relative_to(self.root)
                inside = True
            except ValueError:
                inside = False
            got = _contained_path(s, self.root)
            if inside:
                self.assertIsNotNone(got)
                self.assertEqual(got, expected)
            else:
                self.assertIsNone(got)

    def test_trailing_slash_invariant(self) -> None:
        # Trailing slashes (and `./` prefixes) are ignored — all resolve to the
        # same file. (bin/tool is a DIFFERENT file, so it is deliberately not
        # part of this invariant.)
        base = _contained_path("tool", self.root)
        self.assertIsNotNone(base)
        for s in ("tool", "tool/", "./tool/", "./tool//", str(self.root / "tool") + "/"):
            self.assertEqual(_contained_path(s, self.root), base)

    def test_nonexistent_contained_inside_not_outside(self) -> None:
        # _contained_path is purely lexical: a non-existent path that stays
        # INSIDE root is still "contained" (non-None); only a non-existent path
        # that escapes root (lexically or via a symlink) yields None.
        for _ in range(20):
            name = "no_such_%08x" % _RNG.getrandbits(32)
            self.assertIsNotNone(_contained_path(name, self.root))
            self.assertIsNotNone(_contained_path(str(self.root / name), self.root))
            self.assertIsNone(_contained_path("/etc/" + name, self.root))

    def test_containment_ignores_exec_bit(self) -> None:
        # _contained_path checks CONTAINMENT only, not executability, so a
        # non-executable file inside the tree is still "contained" (non-None).
        # (Executability is enforced separately by _binary_still_contained.)
        for p in (self.root / "plain", self.root / "a" / "notexec"):
            self.assertIsNotNone(_contained_path(str(p), self.root))

    def test_symlink_in_cwd_escaping_rejected(self) -> None:
        for _ in range(30):
            rel = ("lnk_esc/" + _rand_rel(_RNG)) if _RNG.random() < 0.5 else "lnk_esc"
            self.assertIsNone(_contained_path(rel, self.root))

    def test_absolute_inside_root_accepted(self) -> None:
        # Absolute paths lexically inside root are accepted regardless of
        # whether the leaf exists (containment, not existence, is the rule).
        for _ in range(30):
            rel = _rand_rel(_RNG).replace("lnk_esc", "x").replace("lnk_ok", "x")
            rel = rel.replace("../", "").replace("..", "x")
            p = self.root / rel
            got = _contained_path(str(p), self.root)
            self.assertIsNotNone(got)
            self.assertEqual(got, p.resolve())

    def test_cwd_is_symlink_relative_to_resolved(self) -> None:
        # `_contained_path` compares the resolved candidate against the
        # `work_dir` argument with `relative_to`. If the caller passes a
        # work_dir that is itself a symlink, `relative_to` fails and the
        # result is None even though the target is contained.
        base = Path(tempfile.mkdtemp()).resolve()
        root = base / "root"
        root.mkdir()
        _make_exec(root / "tool")
        sym = base / "sym"
        sym.symlink_to(root)
        try:
            # Resolved work_dir: containment against the real tree works.
            self.assertEqual(
                _contained_path("tool", sym.resolve()),
                (root / "tool").resolve(),
            )
            # Raw (unresolved) symlink work_dir: currently None.
            # NOTE: this pins the CURRENT contract — callers must pre-resolve
            # work_dir. It is deliberately NOT a fix for cwd-is-a-symlink.
            self.assertIsNone(_contained_path("tool", sym))
        finally:
            shutil.rmtree(base)


# ---------------------------------------------------------------------------
# _contained_in_any
# ---------------------------------------------------------------------------


class ContainedInAnyPropertyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "lnk_esc").symlink_to("/tmp")
        self.extra = [Path("/tmp").resolve()]
        self.roots = [self.root, *self.extra]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_relative_escape_not_rescued_by_extra_roots(self) -> None:
        # THE key security property: a relative target escaping the work dir
        # must NOT be re-interpreted under an extra root (here /tmp). The
        # naive "re-resolve under /tmp" alternative would wrongly accept
        # `lnk_esc/foo` -> /tmp/foo; this pins that it must NOT.
        for _ in range(30):
            rel = "lnk_esc/" + _rand_rel(_RNG)
            self.assertIsNone(_contained_in_any(rel, self.roots))
        self.assertIsNone(_contained_in_any("lnk_esc", self.roots))

    def test_absolute_in_extra_root_accepted(self) -> None:
        for _ in range(20):
            p = Path("/tmp") / ("prop_%08x" % _RNG.getrandbits(32))
            got = _contained_in_any(str(p), self.roots)
            self.assertIsNotNone(got)
            self.assertEqual(got, p.resolve())

    def test_work_dir_takes_priority_over_extra_roots(self) -> None:
        # work_dir is under /tmp (mkdtemp), so these paths are inside BOTH.
        # roots[0] is tried first, so the result must be the work-dir path.
        for _ in range(20):
            rel = _rand_rel(_RNG).replace("lnk_esc", "x").replace("..", "x")
            p = self.root / rel
            got = _contained_in_any(str(p), self.roots)
            self.assertIsNotNone(got)
            got.relative_to(self.root)  # must be contained in work_dir

    def test_relative_inside_work_dir_accepted(self) -> None:
        for _ in range(30):
            rel = _rand_rel(_RNG).replace("lnk_esc", "x").replace("..", "x")
            self.assertIsNotNone(_contained_in_any(rel, self.roots))

    def test_absolute_outside_all_roots_rejected(self) -> None:
        for p in ("/etc/passwd", "/usr/bin/env", "/var/log/syslog", "/bin/sh"):
            self.assertIsNone(_contained_in_any(p, self.roots))


# ---------------------------------------------------------------------------
# _validate_cwd
# ---------------------------------------------------------------------------


class ValidateCwdPropertyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path.home().resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_matrix_allowed_dirs_accepted(self) -> None:
        # _validate_cwd accepts any *existing* dir inside an allowed root.
        # Under the sandbox's unveil, writes are confined to cwd + /tmp, so we
        # create real dirs at depths 0..3 only under /tmp; for non-writable
        # allowed roots (e.g. ~/projects) we exercise an existing dir.
        tmp_root = Path(tempfile.gettempdir()).resolve()
        for root_str in DEFAULT_ALLOWED_DIRS:
            base = Path(root_str).expanduser().resolve()
            try:
                writable = str(base) == str(tmp_root) or str(base).startswith(
                    str(tmp_root) + "/"
                )
            except Exception:
                writable = False
            if writable:
                for _ in range(6):
                    cur = base
                    for _d in range(_RNG.randint(0, 3)):
                        cur = cur / ("sub_%08x" % _RNG.getrandbits(32))
                    cur.mkdir(parents=True, exist_ok=True)
                    self.assertIsNone(_validate_cwd(cur.resolve(), str(cur)))
            else:
                # base already exists (e.g. ~/projects) and is within an
                # allowed root -> valid.
                self.assertIsNone(_validate_cwd(base, str(base)))

    def test_matrix_disallowed_dirs_rejected(self) -> None:
        # Any EXISTING dir not under an allowed root (~/projects or /tmp) is
        # rejected. We use real system dirs (and, when present, an existing
        # subdir of each) since the sandbox forbids creating dirs outside the
        # allowed roots (which would be needed to fabricate disallowed dirs).
        candidates: list[Path] = [self.home]
        for base in (Path("/etc"), Path("/usr/bin"), Path("/var"), Path("/opt")):
            if not base.is_dir():
                continue
            candidates.append(base)
            try:
                sub = next(p for p in base.iterdir() if p.is_dir())
                candidates.append(sub)
            except (OSError, StopIteration):
                pass
        seen = 0
        for d in candidates:
            err = _validate_cwd(d.resolve(), str(d))
            self.assertIsNotNone(err)
            self.assertIn("not in allowed paths", err)
            seen += 1
        self.assertGreaterEqual(seen, 1)

    def test_missing_dir_message(self) -> None:
        for _ in range(10):
            missing = Path("/tmp") / ("missing_%08x" % _RNG.getrandbits(32))
            err = _validate_cwd(missing.resolve(), "some-raw")
            self.assertIsNotNone(err)
            self.assertIn("Directory not found", err)

    def test_raw_input_preserved_in_message(self) -> None:
        # home is a real dir but disallowed, so the error message must contain
        # the literal raw string the user typed (spaces/unicode/shell chars).
        for raw in _RAW_INPUTS:
            err = _validate_cwd(self.home, raw)
            self.assertIsNotNone(err)
            self.assertIn(raw, err)


# ---------------------------------------------------------------------------
# _binary_still_contained (non-concurrency)
# ---------------------------------------------------------------------------


class BinaryStillContainedPropertyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "bin").mkdir()
        _make_exec(self.root / "tool")
        _make_exec(self.root / "bin" / "tool")
        (self.root / "plain").write_text("data")
        (self.root / "plain").chmod(0o644)
        (self.root / "lnk_esc").symlink_to("/etc")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_nonexistent_false(self) -> None:
        for _ in range(15):
            p = self.root / ("nope_%08x" % _RNG.getrandbits(32))
            self.assertFalse(_binary_still_contained(str(p), self.root))

    def test_non_executable_false(self) -> None:
        self.assertFalse(_binary_still_contained("plain", self.root))

    def test_outside_tree_false(self) -> None:
        for p in ("/etc/hostname", "/usr/bin/true", "/var/tmp", "/bin/sh"):
            self.assertFalse(_binary_still_contained(p, self.root))

    def test_via_symlink_escape_false(self) -> None:
        for _ in range(10):
            rel = "lnk_esc/" + _rand_rel(_RNG)
            self.assertFalse(_binary_still_contained(rel, self.root))

    def test_executable_inside_true(self) -> None:
        # Known executables always True.
        for s in ("tool", "bin/tool", str((self.root / "tool").resolve())):
            self.assertTrue(_binary_still_contained(s, self.root))
        # Random absolute paths: assert True only when they are real execs.
        for _ in range(30):
            rel = _rand_rel(_RNG).replace("lnk_esc", "x").replace("..", "x")
            p = self.root / rel
            if p.is_file() and os.access(p, os.X_OK):
                self.assertTrue(_binary_still_contained(str(p), self.root))

    def test_post_swap_state_false(self) -> None:
        # Deterministic TOCTOU catch: atomically rename a real dir aside and
        # drop in a symlink escaping the tree. The re-resolve must detect it.
        real_dir = self.root / "swapdir"
        real_dir.mkdir()
        _make_exec(real_dir / "tool")
        ph = self.root / ".swap_tmp"
        real_dir.rename(ph)
        real_dir.symlink_to("/etc")
        self.assertFalse(_binary_still_contained(str(real_dir / "tool"), self.root))


# ---------------------------------------------------------------------------
# _binary_still_contained (TOCTOU stress)
# ---------------------------------------------------------------------------


class BinaryStillContainedConcurrencyTest(unittest.TestCase):
    """Stress the TOCTOU narrowing under concurrent directory swaps.

    Non-flaky by design: it asserts the INVARIANT (a True result implies the
    directory was in the REAL state at that snapshot version), never timing.
    """

    def test_swap_invariance(self) -> None:
        base = Path(tempfile.mkdtemp()).resolve()
        work_dir = base / "wd"
        work_dir.mkdir()
        escape_target = base / "outside"   # sibling of work_dir: a true escape
        escape_target.mkdir()

        inner = work_dir / "inner"
        real_placeholder = work_dir / ".inner_real"
        inner.mkdir()
        _make_exec(inner / "bin")

        state = {"real": True}
        events: list[tuple[int, bool]] = []
        calls: list[tuple[int, bool]] = []
        lock = threading.Lock()
        start = threading.Event()

        def writer() -> None:
            try:
                start.wait()
                for _ in range(50):
                    with lock:
                        if state["real"]:
                            # real dir -> escape symlink
                            inner.rename(real_placeholder)
                            inner.symlink_to(escape_target)
                            state["real"] = False
                        else:
                            # escape symlink -> real dir
                            inner.unlink()
                            real_placeholder.rename(inner)
                            state["real"] = True
                        events.append((len(events), state["real"]))
                    time.sleep(0)
            finally:
                # Restore REAL state so TemporaryDirectory cleanup is clean.
                with lock:
                    if not state["real"]:
                        inner.unlink()
                        real_placeholder.rename(inner)
                        state["real"] = True

        def reader() -> None:
            start.wait()
            for _ in range(200):
                # Snapshot version AND run the check under the same lock, so the
                # result reflects exactly the state at that version (no race).
                with lock:
                    version = len(events)
                    res = _binary_still_contained("inner/bin", work_dir)
                    calls.append((version, res))

        t_w = threading.Thread(target=writer)
        t_r = threading.Thread(target=reader)
        t_w.start()
        t_r.start()
        start.set()
        t_w.join(timeout=5)
        t_r.join(timeout=5)
        try:
            if t_w.is_alive() or t_r.is_alive():
                raise TimeoutError("TOCTOU stress threads failed to join")
        finally:
            # Never leave a live thread behind on failure.
            for t in (t_w, t_r):
                if t.is_alive():
                    t.join(timeout=5)
            if os.path.islink(inner) or (real_placeholder.exists() and not inner.exists()):
                pass  # writer's finally should have restored REAL; nothing to force
            shutil.rmtree(base, ignore_errors=True)

        # Invariant checks — no timing assumptions.
        self.assertGreater(len(calls), 0)
        self.assertEqual(len(events), 50)
        for version, res in calls:
            if res:
                self.assertTrue(_state_at(events, version), "True result in ESCAPE state")

        # Sanity log (do not fail): did we actually observe both outcomes?
        outcomes = {res for _, res in calls}
        if len(outcomes) < 2:
            print("  [info] TOCTOU stress observed only %r outcomes" % (outcomes,))


# ---------------------------------------------------------------------------
# _validate_redirect_paths
# ---------------------------------------------------------------------------


class ValidateRedirectPathsPropertyTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        (self.root / "lnk_esc").symlink_to("/tmp")
        self.work_dir = self.root

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _mk_redirect(fd, op, raw=None, target_fd=None) -> Redirect:
        return Redirect(fd=fd, op=op, target_path=None,
                        target_fd=target_fd, raw_target=raw)

    def test_all_inside_work_dir_accepted(self) -> None:
        for _ in range(20):
            rel = _rand_rel(_RNG).replace("lnk_esc", "x").replace("..", "x")
            redirects = [
                self._mk_redirect(1, ">", rel),
                self._mk_redirect(1, ">>", rel + ".log"),
                self._mk_redirect(0, "<", rel),
                self._mk_redirect(1, ">", "/tmp/prop_%08x" % _RNG.getrandbits(32)),
            ]
            out, err = _validate_redirect_paths(redirects, self.work_dir)
            self.assertIsNone(err)
            self.assertEqual(len(out), len(redirects))
            for r in out:
                self.assertIsNotNone(r.target_path)   # populated
                self.assertTrue(Path(r.target_path).is_absolute())

    def test_escape_returns_empty_with_error(self) -> None:
        for target in ("/etc/passwd", "/usr/bin/sh", "/var/log/syslog"):
            out, err = _validate_redirect_paths(
                [self._mk_redirect(1, ">", target)], self.work_dir
            )
            self.assertEqual(out, [])
            self.assertIn("escapes allowed roots", err)

    def test_relative_escape_not_rescued_by_tmp(self) -> None:
        # work_dir/lnk_esc -> /tmp; relative target must NOT be rescued by the
        # extra /tmp redirect root.
        for _ in range(10):
            rel = "lnk_esc/" + _rand_rel(_RNG)
            out, err = _validate_redirect_paths(
                [self._mk_redirect(1, ">", rel)], self.work_dir
            )
            self.assertEqual(out, [])
            self.assertIn("escapes allowed roots", err)

    def test_fd_redirect_passes_through(self) -> None:
        for fd, tfd in ((1, 2), (2, 1)):
            r = self._mk_redirect(fd, ">&", None, target_fd=tfd)
            out, err = _validate_redirect_paths([r], self.work_dir)
            self.assertIsNone(err)
            self.assertEqual(len(out), 1)
            self.assertIs(out[0], r)               # unchanged object
            self.assertIsNone(out[0].target_path)
            self.assertEqual(out[0].op, ">&")
            self.assertEqual(out[0].target_fd, tfd)

    def test_mixed_batch_atomic(self) -> None:
        # A batch is all-or-nothing: any escaping target empties the batch.
        for _ in range(20):
            n = _RNG.randint(1, 5)
            has_escape = _RNG.random() < 0.5
            redirects = []
            n_valid = n if not has_escape else max(n - 1, 0)
            for _i in range(n_valid):
                rel = _rand_rel(_RNG).replace("lnk_esc", "x").replace("..", "x")
                redirects.append(self._mk_redirect(
                    1, _RNG.choice([">", ">>", "<"]), rel
                ))
            if has_escape:
                escape = _RNG.choice(
                    ["/etc/x", "/usr/bin/y", "lnk_esc/z", "/var/log/q"]
                )
                redirects.insert(_RNG.randint(0, len(redirects)),
                                 self._mk_redirect(1, ">", escape))
            out, err = _validate_redirect_paths(redirects, self.work_dir)
            if has_escape:
                self.assertEqual(out, [])
                self.assertIsNotNone(err)
            else:
                self.assertIsNone(err)
                self.assertEqual(len(out), len(redirects))
                for r in out:
                    self.assertIsNotNone(r.target_path)


if __name__ == "__main__":
    unittest.main()
