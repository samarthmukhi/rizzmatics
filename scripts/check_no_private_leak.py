#!/usr/bin/env python3
"""Fail if anything private has leaked into git-tracked (public) files.

Two layers, so the public script itself carries no private vocabulary:

1. **Always-on structural checks** (no secret list needed): no tracked file may
   live under ``private/``, be a ``.env`` / secrets file, or embed the access-key
   hash assignment.
2. **Token checks** (local only): if a git-ignored ``private/forbidden_tokens.txt``
   exists, every token in it must be absent from tracked files. Public CI (which
   has no private package) simply skips this layer.

Run standalone (exit 1 on any leak) or via tests/test_public_private_separation.py.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOKENS_FILE = _REPO_ROOT / "private" / "forbidden_tokens.txt"
_HASH_ASSIGN = re.compile(r"RIZZMATICS_PRIVATE_HASH\s*=\s*[0-9a-fA-F]{16,}")


def tracked_files() -> list[Path]:
    """Files that are (or would become) public: tracked + untracked-not-ignored.

    ``--exclude-standard`` honors .gitignore, so anything under ``private/`` or
    other ignored paths is correctly excluded — this reflects exactly what a
    ``git add -A`` would stage into the public repo.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=True)
    seen = dict.fromkeys(line for line in out.stdout.splitlines() if line)
    return [_REPO_ROOT / line for line in seen]


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def structural_leaks(files: list[Path]) -> list[str]:
    leaks: list[str] = []
    for p in files:
        rel = p.relative_to(_REPO_ROOT).as_posix()
        if rel.startswith("private/"):
            leaks.append(f"{rel}: private/ path is tracked (must be git-ignored)")
            continue
        if rel == ".env" or rel.endswith("secrets.toml"):
            leaks.append(f"{rel}: secret file is tracked")
            continue
        if _HASH_ASSIGN.search(_read(p)):
            leaks.append(f"{rel}: contains a committed access-key hash")
    return leaks


def token_leaks(files: list[Path]) -> list[str]:
    if not _TOKENS_FILE.exists():
        return []  # public build: no token list, skip this layer
    tokens = [t.strip() for t in _TOKENS_FILE.read_text(encoding="utf-8").splitlines()
              if t.strip() and not t.strip().startswith("#")]
    leaks: list[str] = []
    for p in files:
        low = _read(p).lower()
        rel = p.relative_to(_REPO_ROOT).as_posix()
        for tok in tokens:
            if tok.lower() in low:
                leaks.append(f"{rel}: contains private token {tok!r}")
    return leaks


def find_leaks() -> list[str]:
    files = tracked_files()
    return structural_leaks(files) + token_leaks(files)


def main() -> int:
    leaks = find_leaks()
    if leaks:
        print("PRIVATE LEAK DETECTED:", file=sys.stderr)
        for leak in leaks:
            print(f"  ✗ {leak}", file=sys.stderr)
        return 1
    print("✓ no private lore, vocabulary, secrets, or identifiers in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
