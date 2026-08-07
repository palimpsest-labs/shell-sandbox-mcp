.PHONY: all clean sandbox-deps

all: bin/sandbox

# The compiler is the vendored Cosmopolitan toolchain under bin/cosmo-toolchain,
# so builds work without relying on the host's ~/.local/cosmo install.
COSMOCC = bin/cosmo-toolchain/bin/cosmocc

bin/sandbox: vendor/sandbox.c
	$(COSMOCC) -O2 -o bin/sandbox vendor/sandbox.c

# Note: `bin/sandbox` is intentionally NOT removed by `clean`. It is a
# tracked binary (Git LFS) that bin/run-sandbox execs to run every sandboxed
# command — deleting it would break the sandbox tool entirely. `clean` only
# removes the cross-build intermediates cosmocc leaves behind.
clean:
	rm -f bin/sandbox.aarch64.elf bin/sandbox.com.dbg

# Install common test/development packages into the sandbox-local .py-site
# so they are importable by the vendored musl python (the sandbox `python3`).
# Uses --user so pip installs into the sandbox workspace rather than the host
# site-packages.
sandbox-deps:
	python3 -m pip install --user --disable-pip-version-check pytest mcp
