#!/usr/bin/env python3
"""Fetch + extract CPython source .tgz from python.org (gzip, no xz needed).
Run as a TOP-LEVEL sandbox `python3` call (needs inet+dns, which the `make`
promise set does not include)."""
import os, ssl, tarfile, urllib.request, pathlib
VER = "3.12.11"
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # repo root
BLD = ROOT / "build" / "python-musl"
SRC = BLD / "src"
SRC.mkdir(parents=True, exist_ok=True)
tgz = SRC / f"Python-{VER}.tgz"
url = f"https://www.python.org/ftp/python/{VER}/Python-{VER}.tgz"
print("fetching", url)
ctx = ssl.create_default_context()
with urllib.request.urlopen(url, context=ctx, timeout=120) as r:
    data = r.read()
tgz.write_bytes(data)
print("wrote", tgz, len(data), "bytes")
print("extracting (gzip)...")
with tarfile.open(tgz, "r:gz") as t:
    t.extractall(SRC)
print("extracted ->", SRC / f"Python-{VER}")
