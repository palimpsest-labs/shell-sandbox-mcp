#!/bin/bash
# Build a STATIC musl zlib into bin/python-musl/deps/zlib.
# Invoked from a `make` recipe (runs via /bin/bash with proc prot_exec).
# zlib's ./configure uses CC; we pass the musl .br_real gcc directly via env.
# Static (no shared) so CPython's _zlib extension links libz.a into its .so.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MUSL="$ROOT/bin/musl-toolchain/bin"
SR="$ROOT/build/python-musl/zlib-src/zlib-1.3.2"
DST="$ROOT/bin/python-musl/deps/zlib"

CC="$MUSL/x86_64-buildroot-linux-musl-gcc.br_real"
AR="$MUSL/x86_64-buildroot-linux-musl-ar"

cd "$SR"
CC="$CC" AR="$AR" RANLIB="$MUSL/x86_64-buildroot-linux-musl-ranlib" \
    ./configure --static --prefix="$DST" -fPIC
make -j2
make install
echo "zlib built -> $DST"
