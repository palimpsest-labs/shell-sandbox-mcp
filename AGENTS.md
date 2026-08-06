# AGENTS.md — shell-sandbox-mcp

Guidance for agents working in this repository. Read this before running tests or
using the sandbox's Python tooling.

## How to run the test suite

There are TWO test runners. Pick based on which files you are testing.

### 1. Host runner — FULL suite (needs `mcp`)

The full suite imports `shell_sandbox_mcp.server`, which imports `mcp` at module
level. `mcp` cannot run inside the sandbox (its Rust `pydantic-core` dependency is
incompatible with the Cosmopolitan python), so the full suite must run on the host
venv, OUTSIDE the sandbox shell:

```bash
PYTHONPATH=src <venv>/bin/python -m unittest discover -s tests -v
```

Use the host venv that has `mcp` installed. The MCP server venv at
`~/.vibe/mcp-venvs/palimpsest/bin/python` is one such python (it also runs the
sandbox tool itself).

### 2. Sandbox runner — subset (no `mcp` needed)

Tests that import ONLY `shell_sandbox_mcp.parser`, `.config`, `.policy`, or
`.executor` (not `.server`) run fine inside the sandbox via the cosmo python:

```bash
# one-time: install pytest into the sandbox-local site dir
python3 -m pip install --user --disable-pip-version-check pytest

# run the non-mcp subset
python3 -m pytest tests/test_parser_*.py -q
```

Which files need `mcp` (i.e. must run on the host): anything importing
`from shell_sandbox_mcp import server` — e.g. `test_sandbox.py`, `test_policy.py`,
`test_executor.py`, `test_containment.py`, `test_redirects.py`,
`test_env_allowlist.py`, `test_builtins_*.py`, `test_e2e.py`,
`test_ast_consumption.py`, `test_python_env.py`. The pure-parser tests
(`test_parser_*.py`) do NOT need `mcp`.

## Sandbox Python environment

- The sandbox `python3` command is a vendored Cosmopolitan static build at
  `bin/cosmo/python` (Python 3.12). It runs with `-S`, so `.pth` files are NOT
  processed and the sandbox-local site dir is exposed purely via `PYTHONPATH`.
- `pip install --user <pkg>` installs into `<cwd>/.py-site/lib/python<ver>/site-packages`
  and is importable by `python3`. This is the supported way to add packages inside
  the sandbox.
- The project's own package (`src/`) is automatically added to `PYTHONPATH`, so
  `import shell_sandbox_mcp` works inside the sandbox without an editable install.
- The cosmo python has NO `ensurepip`, so `python3 -m venv <dir>` (with pip
  bootstrap) FAILS. Use `python3 -m venv --without-pip <dir>` instead; the venv
  python is then fully functional (its own site-packages + `src/` both importable).
- `pip install --user mcp` will NOT work in the sandbox (native deps). Don't try.

## Misc

- `make` builds `bin/sandbox` via the vendored Cosmopolitan toolchain.
- `make sandbox-deps` runs `pip install --user pytest mcp` for host-side test deps.
- Security: the sandbox confines the filesystem to the working directory + `/tmp`
  via unveil. Commands are allowlisted in `src/shell_sandbox_mcp/policy.py`.
