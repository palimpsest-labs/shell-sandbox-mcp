# AGENTS.md — shell-sandbox-mcp

Guidance for agents working in this repository. Read this before running tests or
using the sandbox's Python tooling.

## How to run the test suite

The full suite runs **inside the sandbox** via the vendored `python3` command.
`mcp` and its Rust `pydantic-core` dependency are vendored into the sandbox-local
site dir (`.py-site/`), so everything (including tests that import
`shell_sandbox_mcp.server`) works under the sandboxed python:

```bash
# full suite (unittest form — ~67s; background it, it exceeds the ~60s MCP call cap)
python3 -m unittest discover -s tests -v

# or pytest (faster output; counts subtests separately from test methods)
python3 -m pytest tests/ -q
```

The in-sandbox runner is authoritative. The old "host venv is required because
`mcp` can't run in-sandbox" split is obsolete — it predates vendoring
`ssl`/`zlib` + `mcp` into `.py-site/`.

If you ever need the host venv (e.g. a fresh checkout before `.py-site` is
provisioned), the MCP server venv at `~/.vibe/mcp-venvs/palimpsest/bin/python`
has `mcp` installed:

```bash
PYTHONPATH=src ~/.vibe/mcp-venvs/palimpsest/bin/python -m unittest discover -s tests -v
```

## Sandbox Python environment

- The sandbox `python3` command is a **vendored musl CPython 3.12.11** at
  `bin/python-musl/install/bin/python3.12` (built in-sandbox with the Bootlin
  musl toolchain, LFS-tracked). It is dynamically linked against the staged
  `ld-musl` loader + `libc.so` under `bin/python-musl/install/lib/rtlib/`. It
  is real CPython — site.py and `.pth` files work normally, and it can
  `dlopen` native `.so` extensions (including ones compiled with the musl gcc).
- `pip install --user <pkg>` installs into `<cwd>/.py-site/lib/python<ver>/site-packages`
  and is importable by `python3`. This is the supported way to add packages inside
  the sandbox.
- The project's own package (`src/`) is automatically added to `PYTHONPATH`, so
  `import shell_sandbox_mcp` works inside the sandbox without an editable install.
- The vendored musl python was built with `--without-ensurepip`, so it has NO
  bundled pip. Use `python3 -m venv --without-pip <dir>` to create a venv; the
  venv python is then fully functional (its own site-packages + `src/` both
  importable). Bootstrap pip separately if needed (e.g. get-pip.py).
- `pip install --user mcp` works in-sandbox now that `ssl`/`zlib` (and the Rust
  `pydantic-core` wheel) are vendored. It installs into `.py-site/` and is
  importable by `python3`. (Older CPython sandboxes without those vendored deps
  could not build/load `mcp`; this is no longer a limitation.)

## Misc

- `make` builds `bin/sandbox` via the vendored Cosmopolitan toolchain.
- `make sandbox-deps` runs `pip install --user pytest mcp` — historically for
  host-side deps, now effectively provisioning the sandbox-local `.py-site/`.
- Security: the sandbox confines the filesystem to the working directory + `/tmp`
  via unveil. Commands are allowlisted in `src/shell_sandbox_mcp/policy.py`.
