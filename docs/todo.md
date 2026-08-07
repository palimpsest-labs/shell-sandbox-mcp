# shell-sandbox-mcp — Architecture Refactor Plan

Status of this repo: **works well** — 440+ passing tests, and the security boundary
(pledge/unveil + per-command allowlist + path containment) is genuinely solid. The
complexity cost is concentrated in two places:

1. `server.py` is an **1829-line monolith** mixing five distinct responsibilities.
2. `parser.py` carries a documented **"two parser paths" duplication** — a legacy
   char-by-char scanner and a real Lexer/AST kept in lockstep by hand.

The goal is **graceful evolution, not a rewrite**. Keep the test suite green at every
commit; the suite is the safety net.

---

## Scope / current shape

| File | Lines | Notes |
|------|-------|-------|
| `src/shell_sandbox_mcp/server.py` | 1829 | Monolith: allowlist, env, `cd`, execution, redirects, subprocess, reaper, MCP wiring |
| `src/shell_sandbox_mcp/parser.py` | 2230 | Lexer + AST; TWO overlapping parse paths + `split_legacy` + `_extract_from_string` |
| `tests/test_sandbox.py` | 2949 | 28 test classes in one file |
| `tests/test_parser_*.py` (7 files) | ~2747 | lexer/ast/escapes/differential/quoted_subst/varexp |
| `tests/test_env_allowlist.py` | 133 | |
| `vendor/sandbox.c`, `Makefile`, `bin/` | — | LFS-tracked binaries + sandbox C source |

---

## What to preserve (security boundary)

- **Containment layer**: `_contained_path` / `_contained_in_any` / `_binary_still_contained`
  (server.py:397/415/436). TOCTOU narrowing via exec-time re-validation + `O_NOFOLLOW`
  on redirect targets (server.py:755) is correct.
- **Reaper thread** PID-set discipline (server.py:1342–1378): only reaps registered PIDs,
  correctly avoiding the foreground/background race.
- **Declarative per-command `COMMANDS` dict** (server.py:201) with callables for
  `extra_unveil*` / `path_prefix` — clean way to express per-tool policy.
- **AST dataclasses** (parser.py:129–198): frozen, small, well-typed; currently underused.
- **`Expansion` sentinel side-table design**: cleanly separates parsed structure from
  expanded values, letting the AST stay literal.

---

## (a) High-value refactors

### A1 — Unify the two parser paths onto the AST (highest leverage, riskiest)
- `parse_command` (parser.py:1782–2189, ~400 lines) runs **two parses** over every input:
  1. A char-by-char scanner producing `cleaned` + filling the `Expansion` side table.
  2. A `Lexer().tokenize()` + `_build_ast` pass producing the `ProgramNode`.
- `split_legacy` (parser.py:1196) and `_extract_from_string` (parser.py:1416) are 3rd/4th
  re-implementations of the same grammar.
- `shell_run` has **two dispatch paths** (server.py:1627 vs server.py:1700) plus a legacy
  fallback branch when `program is None` (server.py:1700–1773).

**Change:** make the AST the only path.
- Move `$()` capture, `$VAR`/`${VAR}` expansion, and heredoc-body resolution into the AST
  build (`_build_ast` accepts `capture_fn` + `env`, populates `expansion` directly).
- Delete the char-by-char scanner; `parse_command` becomes a thin
  `Lexer → _build_ast → (program, expansion)` wrapper. Drop the `cleaned` string return.
- Delete `split_legacy` and `_extract_from_string`. `program_to_chain` and
  `_extract_from_node` become the only execution-entry projections.
- Delete the `program is None` legacy fallback in `shell_run` — parse failures should raise
  `ParseError` and surface as a clean error, not silently fall back to a different parser.

**Why better:** one quote/escape/subst/redirect grammar, one splitting grammar, one
redirect-extraction grammar. Divergence impossible by construction.

**Risks:**
- The scanner is "proven, tested". **Keep `test_parser_differential.py` green throughout**;
  delete/repurpose it last (see C4).
- Edge cases the two paths handle differently (e.g. empty `$()` preserved as empty arg on
  scanner path, dropped on AST path) are **behavior changes**. Audit the differential tests
  for what they pin; decide each deliberately.
- Land incrementally: first make `_build_ast` populate `expansion` and assert equality with
  the scanner, then flip `shell_run` to AST-only, then delete the scanner.

### A2 — Split `server.py` into cohesive modules (low risk)
```
src/shell_sandbox_mcp/
  config.py      # REPO_ROOT, paths, DEFAULT_ALLOWED_DIRS, EXTRA_REDIRECT_ROOTS,
                 # DEFAULT_TIMEOUT, MAX_*, _ENV_ALLOWLIST, _base_env   (dumb, no subprocess)
  policy.py      # COMMANDS, BUSYBOX_APPLETS, _git_*, _cosmo_toolchain_*, _resolve_command,
                 # _resolve_local_binary, _stage_git_global_config  (per-tool unveil/pledges)
  containment.py # _contained_path, _contained_in_any, _binary_still_contained,
                 # _validate_cwd, _validate_redirect_paths
  redirects.py   # Redirect re-export, _resolve_fd_targets (pure fd→target resolution)
  executor.py    # _build_invocation, _run_segment_core, _run_pipeline_core, _run_background,
                 # _format_output, _expand_command, _expand_subst_in_text,
                 # reaper (_start_reaper, _bg_pids)
  builtins.py    # _try_cd (the cd builtin)
  server.py      # mcp = FastMCP(...), @mcp.tool shell_run / shell_list, main()
```
- Each module has a single reason to change; the boundary between `policy` and `containment`
  is real (`_resolve_command` → `_resolve_local_binary` → `_contained_path`).
- **Back-compat shim:** keep `server.py` re-exporting the symbols tests import from it
  (`from shell_sandbox_mcp.server import Expansion, Redirect, …`). Do NOT move re-exports
  into a sub-module — tests import from `server`, so it stays the facade.

**Risks:** helpers tests call directly (`_resolve_command`, `_validate_cwd`,
`_resolve_fd_targets`, `_try_cd`, `_build_invocation`) must remain importable from `server`,
or update the test imports (acceptable but more churn). Reaper globals → prefer a small
`_Reaper` singleton over bare module globals. Keep `_stage_git_global_config` lazy (only on
`is_git`), not at import time.

### A3 — Extract a `PipelineExecutor` to collapse the three near-duplicate runners (tricky)
`_run_segment_core` (server.py:812), `_run_pipeline_core` (server.py:928), and
`_run_background` (server.py:1381) all do the same 5-step dance with variations:
1. Loop segments → `_build_invocation` → checks → collect invocations.
2. Reject stdout redirects on intermediate stages.
3. Resolve fd targets per stage via `_resolve_fd_targets`.
4. Launch `Popen`s chaining `prev.stdout → next.stdin`; write first-stage stdin from a thread.
5. Cleanup: kill procs, wait, close fds, close shared pipe.

**Change:** one `PipelineExecutor` (or `Pipeline` class) parameterized by:
- `mode`: `"foreground" | "background"`
- `stderr_drain`: bool (foreground drains intermediate stderr on threads)
- `last_stage_default`: `subprocess.PIPE` (foreground) or a log-sentinel (background)
- `register_pids`: optional callback (background → reaper)
- `timeout` / `deadline`

The single-stage fast path collapses to `Pipeline(stages=[one]).run_foreground()`.

**Risks:** the three runners have slightly different error-message wording/ordering; the
`Run*RedirectTest` classes (test_sandbox.py:1141/1324/1424) may pin exact strings. Diff the
error paths before merging; unify strings (update tests) or preserve per-mode strings. The
background `LOG_SENTINEL` lazy-open should become an explicit `LastStageSink` abstraction.

### A4 — Give `_build_invocation` a real result type (small, mechanical)
Today it returns `(binary, sandbox_args, env, cfg, redirects)` on success but
`(error_msg, None, None, None, [])` on failure — first element is path-or-error,
disambiguated only by element 2 being `None`. Every caller does the
`if sandbox_args is None: if binary is None: empty else error` dance.

```python
@dataclass
class Invocation:
    binary: str
    sandbox_args: list[str]
    env: dict[str, str]
    cfg: dict
    redirects: list[Redirect]

@dataclass
class InvocationError:
    message: str

@dataclass
class EmptyInvocation:
    pass

# _build_invocation(...) -> Invocation | InvocationError | EmptyInvocation
```
Callers become `match`/`isinstance` checks. Touches every caller (mostly mechanical);
tests asserting tuple shape need updating.

---

## (b) Elegance / consistency

### B1 — Split the 2949-line `test_sandbox.py` by module (zero risk, pure move)
28 classes already grouped. Split to mirror A2: `test_containment.py`, `test_policy.py`,
`test_redirects.py`, `test_executor.py`, `test_builtins_cd.py`, `test_e2e.py`,
`test_ast_consumption.py`. Pure file move, no test-body edits, no renames. Extract
`tests/helpers.py` if a test imports a helper from another class.

### B2 — Magic constants & latent bug
- **Latent bug (FIXED):** server.py hardcoded `lib/python3.12/site-packages` for the python3
  sandbox-local site dir. Now derived dynamically from the vendored python's
  `_python_version()` (was the cosmo python, now the musl python's version).
- `MAX_SUBST_DEPTH/COUNT/OUTPUT/HEREDOC_BODY` defined in parser.py:36–39 then re-imported in
  server.py:65 "for backward compatibility" and re-referenced in `shell_list`'s docstring.
  Define once in `config.py`, import everywhere.
- Bare octals `0o600` (staged git config, secret) vs `0o666` (redirect targets, honors umask)
  — worth a comment explaining the contrast.

### B3 — Dead/duplicated code & naming
- `cmd_to_display` (parser.py:1319) vs `_serialize_command` (parser.py:1171) vs
  `serialize_program` (parser.py:1144) — three render-to-string functions. Pick one canonical
  serializer (`serialized`); others delegate.
- `_split_command` (server.py:529) is a 1-line wrapper around `split_legacy` → delete after A1.
- `_extract_redirects` (server.py:542) is a 1-line wrapper around `parser.extract_redirects` →
  keep only if tests import it from `server`, else delete.
- `_expand_subst_in_text` (server.py:1272) collapses into the AST build after A1.

### B4 — Confine the `Expansion` sentinel scheme to `parser.py`
`SENTINEL_ARG`/`SENTINEL_HD` regexes + literal `\x01A…\x01` strings leak into ~6 functions
(`_extract_from_string`, `_emit_var_sentinel`, `_try_expand_var`, scanner). After A1, expose
`Expansion` only as an opaque lookup: `expansion.arg_for(part) -> str`,
`expansion.heredoc_for(part) -> str`. The `\x01` bytes should never appear outside parser.py.

### B5 — `_resolve_fd_targets` returns a 7-tuple → `FdPlan` dataclass
Currently returns `(stdout_target, stderr_target, files_to_close, report_lines,
shared_pipe_read_fd, stdin_bytes, stdin_file)`. Callers unpack 7 positional names.
```python
@dataclass
class FdPlan:
    stdout: object
    stderr: object
    to_close: list
    report: list[str]
    shared_read_fd: Optional[int]
    stdin_bytes: Optional[bytes]
    stdin_file: Optional[object]
```
Mechanical win; unblocks A3.

---

## (c) Longer-term / aspirational

### C1 — Explicit parse → plan → execute pipeline
`shell_run` (server.py:1552–1773) currently interleaves parsing, expansion, AST-walking,
builtin interception, invocation-building, and execution in one ~270-line function.
Intended shape:
```
command string
  → Lexer.tokenize() → _build_ast(capture_fn, env) → ProgramNode + Expansion   [parse]
  → program_to_chain(program) → list[(op, [CommandNode], bg)]                  [plan]
  → for each chain: BuiltinResolver.try_cd(node) | Executor.run(node)           [execute]
  → FormattedResult                                                              [render]
```
Concretize as a small `Plan` type + `Runner` owning mutable `work_dir` (across `cd`) and
`prev_rc` (across `&&`/`||`). `shell_run` becomes ~30 lines. **Land after A1+A3+A4.**

### C2 — `Redirect` model that owns its fd resolution
`_resolve_fd_targets` (server.py:693, 117 lines) knows POSIX `2>&1` snapshot semantics. The
`snapshot_2gt1` bool threaded through it leaks "am I the last stage of a foreground
pipeline?" into a function that shouldn't know about pipelines. Model as:
```python
class RedirectPlan:
    def apply(self, defaults: FdDefaults) -> FdPlan: ...
```
Each `Redirect` applies itself in sequence; shared-pipe-for-`1>&2` becomes a method on
`FdPlan`. Land **after A3**.

### C3 — Property/edge-case tests for the security layer
Containment functions are pure + security-critical but tested by examples. Add a
property-style suite over: absolute/relative, symlink-in-cwd-escaping, `..` traversal,
`./` prefix, trailing-slash, non-existent, non-executable, `cwd`-is-symlink. The TOCTOU
re-check (`_binary_still_contained`) deserves a concurrency-stress test.

### C4 — Decide the `test_parser_differential.py` end-state
After A1 the differential tests are tautological (AST vs AST). Options: (1) delete, or
(2) repurpose as a **golden suite** — pin a curated set of `(input, expected_args,
expected_redirects, expected_expansion)` triples the AST must satisfy. **Recommend option 2.**

---

## Alternative overall shape (weighed against the monolith)

**`Shell` facade class with injected collaborators:**
```python
class Shell:
    def __init__(self, policy, containment, executor): ...
    def run(self, command, cwd, timeout) -> str: ...
```
`Runner` owns `work_dir` + `prev_rc`; `cd` mutates `runner.work_dir`; `&&`/`||` consult
`runner.prev_rc`. MCP `shell_run` becomes `Shell(...).run(...)`.
- **Pros:** state explicit instead of 5-positional-arg chains; easier to test (inject a fake
  Executor); `work_dir` mutation is a method call not a closure-captured reassignment.
- **Cons:** bigger change to call sites; tests calling `_run_segment_core` directly must go
  through the facade or a test-only entry.
- **Recommendation:** do A2 (flat modules) first — the safe, suite-preserving enabler. Adopt
  the `Shell`/`Runner` facade (C1) later, after signatures stabilize. The facade is the right
  endpoint; the flat modules are the safe path there.

---

## Recommended sequencing

1. **A4** (Invocation result type) — small, mechanical, unblocks A3.
2. **B5** (FdPlan dataclass) — small, mechanical, unblocks A3.
3. **A2** (split server.py into modules, keep re-exports) — pure move, low risk.
4. **B1** (split test_sandbox.py to mirror A2) — pure move, zero risk, eases review.
5. **A3** (PipelineExecutor collapses three runners) — tricky; do after A4+B5. Diff error
   strings carefully against `Run*RedirectTest`.
6. **A1** (unify parser paths onto AST, delete scanner + `split_legacy` +
   `_extract_from_string` + `shell_run` legacy branch) — highest leverage, riskiest; keep
   `test_parser_differential.py` green, delete/repurpose last (C4).
7. **B2–B4** (constants, naming, sentinel encapsulation) — opportunistically during the above.
8. **C1/C2** (Plan/Runner facade, Redirect-as-applier) — only after 1–6 land and stabilize.

---

## Test-suite-breakage risk (ranked)

1. **A3** — error-string/ordering diffs in `RunSegmentRedirectTest` /
   `RunPipelineRedirectTest` / `RunBackgroundRedirectTest` (test_sandbox.py:1141/1324/1424).
2. **A1** — empty `$()` and other edge cases pinned by `test_parser_differential.py`.
3. **A4** — tuple→dataclass in `BuildInvocationRedirectTest` / `BuildInvocationHeredocTest`.
4. **A2** — only if re-exports are missed.
5. Everything else — near zero if done as pure moves.

Net target: **−500 to −800 LOC**, one parser grammar, one executor, `shell_run` on one screen.
