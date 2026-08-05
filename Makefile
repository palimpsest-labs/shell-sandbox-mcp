.PHONY: all clean

all: bin/sandbox

# The compiler is the vendored Cosmopolitan toolchain under bin/cosmo-toolchain,
# so builds work without relying on the host's ~/.local/cosmo install.
COSMOCC = bin/cosmo-toolchain/bin/cosmocc

bin/sandbox: vendor/sandbox.c
	$(COSMOCC) -O2 -o bin/sandbox vendor/sandbox.c

clean:
	rm -f bin/sandbox bin/sandbox.aarch64.elf bin/sandbox.com.dbg
