"""Path containment — validation helpers for working directories, redirects,
and local binaries.

Functions that ensure filesystem paths stay within allowed roots (the
working directory, allowed trees, and /tmp for redirects).
"""

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .config import DEFAULT_ALLOWED_DIRS, EXTRA_REDIRECT_ROOTS
from .parser import Redirect

# ---------------------------------------------------------------------------
# Module-level cache: resolve DEFAULT_ALLOWED_DIRS once at import time so
# _validate_cwd doesn't re-resolve them on every call.
# ---------------------------------------------------------------------------
_RESOLVED_ALLOWED_DIRS: tuple[Path, ...] = tuple(
    Path(allowed).expanduser().resolve() for allowed in DEFAULT_ALLOWED_DIRS
)


def _contained_path(cmd_name: str, work_dir: Path) -> Optional[Path]:
    """Resolve `cmd_name` relative to `work_dir`, returning the resolved
    path only if it stays within the working tree; else None.

    Uses `resolve()` (which follows symlinks), so a symlink inside the cwd
    pointing outside it is caught by the containment check.
    """
    try:
        raw = Path(cmd_name)
        candidate = raw if raw.is_absolute() else (work_dir / raw)
        candidate = candidate.resolve()
        # Must stay inside the working directory (or below it).
        candidate.relative_to(work_dir)
    except (ValueError, OSError):
        return None
    return candidate


def _contained_in_any(target: str, roots: list[Path]) -> Optional[Path]:
    """Resolve target against the first root that contains it; else None.

    The first root (the working directory) is tried first for all targets.
    Extra roots (e.g. /tmp) are only consulted for absolute targets — a
    *relative* target is always relative to the working directory, so if it
    escapes the work dir it must not be re-interpreted against another root
    (that would let a work-dir symlink escape by "resolving" under /tmp).
    """
    cand = _contained_path(target, roots[0])
    if cand is not None:
        return cand
    if not Path(target).expanduser().is_absolute():
        return None
    for root in roots[1:]:
        cand = _contained_path(target, root)
        if cand is not None:
            return cand
    return None


def _binary_still_contained(binary: str, work_dir: Path) -> bool:
    """Re-verify a previously-resolved local binary path is still an
    executable file contained within the work dir, right before exec.

    Narrows the TOCTOU window between initial resolution and exec: if a
    directory component was swapped for a symlink escaping the tree in the
    meantime, the re-resolve + containment check catches it. Called with the
    path actually passed to the sandbox.
    """
    candidate = _contained_path(binary, work_dir)
    if candidate is None:
        return False
    return candidate.is_file() and os.access(candidate, os.X_OK)



def _validate_cwd(resolved: Path, raw: str) -> Optional[str]:
    """Validate a working directory. `resolved` must be an already-resolved
    absolute path; `raw` is the user's original input for error messages.

    Returns an error string, or None if valid.
    """
    if not resolved.is_dir():
        return f"Directory not found: {raw}"

    # Must be within an allowed tree or a subdirectory thereof
    for allowed_path in _RESOLVED_ALLOWED_DIRS:
        try:
            resolved.relative_to(allowed_path)
            return None
        except ValueError:
            continue

    return f"Directory not in allowed paths: {raw}"


def _validate_redirect_paths(
    redirects: list[Redirect],
    work_dir: Path,
) -> tuple[list[Redirect], Optional[str]]:
    """Resolve and validate the paths in a list of redirects.

    For each ``>`` / ``>>`` / ``<`` redirect, resolves ``raw_target`` against
    ``work_dir`` (or an extra redirect root such as /tmp) via
    :func:`_contained_in_any`.  Returns the updated list (with ``target_path``
    populated) or ``([], error_msg)`` if a target escapes all allowed roots.
    ``>&`` redirects pass through unchanged.
    """
    validated: list[Redirect] = []
    for r in redirects:
        if r.op in (">", ">>", "<"):
            cand = _contained_in_any(r.raw_target, [work_dir, *EXTRA_REDIRECT_ROOTS])
            if cand is None:
                return [], f"Redirect target escapes allowed roots: {r.raw_target}"
            validated.append(replace(r, target_path=str(cand)))
        else:
            validated.append(r)
    return validated, None
