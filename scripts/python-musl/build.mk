# scripts/python-musl/build.mk — build a real CPython (musl, dynamic, dlopen-capable)
# inside the shell-sandbox-mcp sandbox. Driven via `make` (recipes run via /bin/bash).
#
# Usage (from repo root, cwd = repo root so the musl toolchain under bin/ is rwcx):
#   make -f scripts/python-musl/build.mk rtlib        # one-time: stage musl loader+libc
#   make -f scripts/python-musl/build.mk configure      # ./configure (foreground, ~60s)
#   make -f scripts/python-musl/build.mk build  &       # BACKGROUND (long). poll .build-done
#   make -f scripts/python-musl/build.mk install       # ~1-2 min (foreground ok)
#   make -f scripts/python-musl/build.mk verify        # run + compile+import a .so ext
#
# Fetch+extract of the source is done OUTSIDE this makefile (needs inet+dns which the
# `make` promise set lacks) via the allowlisted top-level `python3`:
#   python3 scripts/python-musl/fetch.py

SHELL := /bin/bash
.SHELLFLAGS := --norc --noprofile -c

ROOT   := $(CURDIR)
MUSL   := $(ROOT)/bin/musl-toolchain/bin
SR     := $(ROOT)/bin/musl-toolchain/x86_64-buildroot-linux-musl/sysroot/lib
BLD    := $(ROOT)/build/python-musl
SRC    := $(BLD)/src/Python-3.12.11
# Install into the VENDORED tree under bin/python-musl/ (LFS-tracked) so the
# built interpreter is a stable, committed first-class command. The binary's
# PT_INTERP + rpath are baked to $(RT), so rebuilding with this prefix makes
# the vendored loader path self-consistent.
INST   := $(ROOT)/bin/python-musl/install
RT     := $(INST)/lib/rtlib
PY     := $(INST)/bin/python3.12

VER    := 3.12.11

# --- musl cross tools (full paths; bare gcc/cc/ar resolve to HOST glibc on PATH) ---
CC      := $(MUSL)/x86_64-buildroot-linux-musl-gcc.br_real
CXX     := $(MUSL)/x86_64-buildroot-linux-musl-g++.br_real
AR      := $(MUSL)/x86_64-buildroot-linux-musl-ar
RANLIB  := $(MUSL)/x86_64-buildroot-linux-musl-ranlib
READELF := $(MUSL)/x86_64-buildroot-linux-musl-readelf
LD      := $(MUSL)/x86_64-buildroot-linux-musl-ld

# --- runtime model: DYNAMIC musl, workdir-local interpreter + libc.so, rpath=RT ---
# (Static musl cannot dlopen => "Dynamic loading not supported". Dynamic + local
#  loader makes dlopen work, proven in-sandbox.)
RT_FLAGS  := -Wl,--dynamic-linker=$(RT)/ld-musl-x86_64.so.1 -Wl,-rpath,$(RT)
EXP_FLAGS := -Wl,--export-dynamic          # let dlopen'd .so resolve Py* from python bin

CFLAGS   := -O2 -fPIC -fno-semantic-interposition
LDFLAGS  := $(RT_FLAGS) $(EXP_FLAGS)
# CPython builds extensions via LDSHARED/BLDSHARED; give them the rpath too.
LDSHARED  := $(CC) -shared -fPIC $(RT_FLAGS)
BLDSHARED := $(CC) -shared -fPIC $(RT_FLAGS)

# CPython's configure variables (exported on the ./configure line below).
CONF_ENV := CC='$(CC)' CXX='$(CXX)' AR='$(AR)' RANLIB='$(RANLIB)' READELF='$(READELF)' \
            LD='$(LD)' CFLAGS='$(CFLAGS)' LDFLAGS='$(LDFLAGS)' \
            LDSHARED='$(LDSHARED)' BLDSHARED='$(BLDSHARED)' \
            PKG_CONFIG= Ac_cv_prog_pkg_config=

JOBS := 2   # sandbox shares the host (os.cpu_count()==4); be polite

# --- vendored static musl OpenSSL (built once; _ssl/_hashlib link against it) ---
OPENSSL_PREFIX := $(ROOT)/bin/python-musl/deps/openssl
OPENSSL_STAMP  := $(OPENSSL_PREFIX)/.openssl-built
OPENSSL_SRC    := $(ROOT)/bin/python-musl/deps/openssl-src
OPENSSL_VER    := 3.0.21

# --- vendored static musl zlib (needed for the _zlib extension + pip bootstrap) ---
ZLIB_PREFIX := $(ROOT)/bin/python-musl/deps/zlib
ZLIB_STAMP  := $(ZLIB_PREFIX)/.zlib-built

.PHONY: rtlib openssl zlib configure build install verify clean

# Fetch + build the static musl OpenSSL (once). The source clone needs inet/dns,
# which the `make` promise set lacks, so it must be cloned from a TOP-LEVEL
# `git` call first:
#   git clone --depth 1 --branch openssl-$(OPENSSL_VER) \
#       https://github.com/openssl/openssl.git $(OPENSSL_SRC)
# Then this target builds it (make recipes run via /bin/bash with proc prot_exec,
# so perl/make subprocesses work).
openssl:
	@if [ ! -f $(OPENSSL_STAMP) ]; then \
	    [ -d $(OPENSSL_SRC) ] || { \
	        echo "OpenSSL source missing. Run (top-level): git clone --depth 1 --branch openssl-$(OPENSSL_VER) https://github.com/openssl/openssl.git $(OPENSSL_SRC)"; \
	        exit 1; \
	    }; \
	    bash $(ROOT)/scripts/python-musl/build-openssl.sh && touch $(OPENSSL_STAMP); \
	fi

# Build the static musl zlib (once). Source must be fetched + extracted to
# $(ROOT)/build/python-musl/zlib-src/zlib-1.3.2 first (via a top-level python3
# download + busybox tar, since the musl python lacks zlib to unpack itself).
zlib:
	@if [ ! -f $(ZLIB_STAMP) ]; then \
	    [ -d $(ROOT)/build/python-musl/zlib-src/zlib-1.3.2 ] || { \
	        echo "zlib source missing. See scripts/python-musl/build-zlib.sh for the fetch/extract steps."; \
	        exit 1; \
	    }; \
	    bash $(ROOT)/scripts/python-musl/build-zlib.sh && touch $(ZLIB_STAMP); \
	fi

rtlib:
	@mkdir -p $(RT)
	@cp -f $(SR)/ld-musl-x86_64.so.1 $(RT)/
	@cp -f $(SR)/libc.so            $(RT)/
	@chmod 0755 $(RT)/ld-musl-x86_64.so.1 $(RT)/libc.so
	@echo "rtlib staged at $(RT)"

configure: rtlib openssl zlib
	@cd $(SRC) && $(CONF_ENV) ./configure \
	    --build=x86_64-linux-musl \
	    --host=x86_64-linux-musl \
	    --prefix=$(INST) \
	    --without-ensurepip \
	    --with-openssl=$(OPENSSL_PREFIX) \
	    OPENSSL_LDFLAGS='-L$(OPENSSL_PREFIX)/lib -lssl -lcrypto' \
	    OPENSSL_LIBS='-lssl -lcrypto' \
	    OPENSSL_INCLUDES='-I$(OPENSSL_PREFIX)/include' \
	    CPPFLAGS='-I$(ZLIB_PREFIX)/include' \
	    LDFLAGS='$(LDFLAGS) -L$(ZLIB_PREFIX)/lib' \
	    ac_cv_buggy_getaddrinfo=no \
	    > $(BLD)/configure.log 2>&1
	@echo "configure done (see $(BLD)/configure.log)"

build: configure
	@cd $(SRC) && make -j$(JOBS) > $(BLD)/build.log 2>&1
	@touch $(BLD)/.build-done
	@echo "build done"

install: build
	@cd $(SRC) && make install > $(BLD)/install.log 2>&1
	@echo "install done -> $(INST)"

verify: install
	@echo "--- python version ---"
	@$(PY) -c "import sys,platform; print(sys.version); print(platform.system(), platform.machine())"
	@echo "--- dlopen capability ---"
	@$(PY) -c "import ctypes; print('ctypes dlopen ok:', ctypes.CDLL(None))"
	@echo "--- compile + import a native .so extension ---"
	@mkdir -p $(BLD)/extdemo
	@printf '%s\n' '#include <Python.h>' 'static PyObject* demo(PyObject*s,PyObject*a){return PyLong_FromLong(7*6);}' 'static PyMethodDef m[]={{"answer",demo,METH_NOARGS,NULL},{NULL,NULL,0,NULL}};' 'static struct PyModuleDef mod={PyModuleDef_HEAD_INIT,"_demo",NULL,-1,m};' 'PyMODINIT_FUNC PyInit__demo(void){return PyModule_Create(&mod);}' > $(BLD)/extdemo/_demo.c
	@$(CC) -shared -fPIC -I$(SRC)/Include -I$(SRC) $(RT_FLAGS) $(BLD)/extdemo/_demo.c -o $(BLD)/extdemo/_demo.so
	@cd $(BLD)/extdemo && $(PY) -c "import _demo; print('_demo.answer()=', _demo.answer())"
	@echo "VERIFY OK"

clean:
	rm -rf $(BLD)/src $(BLD)/.build-done $(BLD)/configure.log $(BLD)/build.log $(BLD)/install.log
	# Note: $(INST) is the vendored tree under bin/python-musl/ and is NOT
	# touched by clean. To re-vendor from scratch: rm -rf bin/python-musl/install.
