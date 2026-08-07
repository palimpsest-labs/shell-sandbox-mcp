#!/bin/bash
# Build a STATIC musl OpenSSL into bin/python-musl/deps/openssl.
# Invoked from a `make` recipe (which runs via /bin/bash with proc prot_exec),
# so perl/gcc/make subprocesses work.
#
# The musl toolchain binaries are named x86_64-buildroot-linux-musl-{gcc,ar,ranlib}.
# OpenSSL's Configure emits CC=$(CROSS_COMPILE)gcc; we give it a shim dir on
# PATH containing gcc/cc/ar/ranlib symlinks to the .br_real drivers, so it
# finds the musl toolchain without tripping the buildroot toolchain-wrapper
# symlink (which breaks under the sandbox because argv[0] is resolved).
#
# no-shared + -fPIC is REQUIRED: CPython's _ssl extension statically links
# libcrypto/libssl into its .so; non-PIC archives fail to dlopen.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MUSL="$ROOT/bin/musl-toolchain/bin"
SR="$ROOT/bin/python-musl/deps/openssl-src"
DST="$ROOT/bin/python-musl/deps/openssl"
SHIM="$ROOT/build/python-musl/openssl-shim/bin"

mkdir -p "$SHIM"
ln -sf "$MUSL/x86_64-buildroot-linux-musl-gcc.br_real"    "$SHIM/gcc"
ln -sf "$MUSL/x86_64-buildroot-linux-musl-gcc.br_real"    "$SHIM/cc"
ln -sf "$MUSL/x86_64-buildroot-linux-musl-ar"             "$SHIM/ar"
ln -sf "$MUSL/x86_64-buildroot-linux-musl-ranlib"         "$SHIM/ranlib"

export PATH="$SHIM:$PATH"

cd "$SR"
perl Configure linux-x86_64 \
    no-shared no-tests no-ui-console no-afalgeng no-fips no-asm \
    --prefix="$DST" --libdir=lib --openssldir=ssl \
    -fPIC
make -j2
make install_sw
echo "OpenSSL built -> $DST"
