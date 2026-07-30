.PHONY: all clean

all: bin/sandbox

bin/sandbox: vendor/sandbox.c
	cosmocc -O2 -o bin/sandbox vendor/sandbox.c

clean:
	rm -f bin/sandbox bin/sandbox.aarch64.elf bin/sandbox.com.dbg
