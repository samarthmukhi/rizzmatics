#!/usr/bin/env python3
"""Set the After-Hours access key — locally, without ever storing plaintext.

Prompts for a passphrase (hidden), prints only its SHA-256 hash, and optionally
writes that hash to a git-ignored ``.env``. The plaintext never touches disk,
logs, or the terminal echo. Choose your own key — do NOT reuse a birthday; those
are low-entropy and only meant to inspire the on-screen clue, not to be the key.

Usage:
    python scripts/set_private_password.py            # print the hash
    python scripts/set_private_password.py --write     # also append to .env
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from getpass import getpass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = _REPO_ROOT / ".env"
HASH_KEY = "RIZZMATICS_PRIVATE_HASH"


def main() -> int:
    parser = argparse.ArgumentParser(description="Set the private access key.")
    parser.add_argument("--write", action="store_true",
                        help="Append the hash to a git-ignored .env file.")
    args = parser.parse_args()

    pw = getpass("New After-Hours access key (hidden): ")
    if not pw or len(pw) < 6:
        print("Refusing: choose at least 6 characters.", file=sys.stderr)
        return 1
    if getpass("Confirm: ") != pw:
        print("Mismatch. Nothing changed.", file=sys.stderr)
        return 1

    digest = hashlib.sha256(pw.encode("utf-8")).hexdigest()
    del pw  # do not keep the plaintext around

    print(f"\n{HASH_KEY}={digest}")

    if args.write:
        line = f"{HASH_KEY}={digest}\n"
        existing = ENV_PATH.read_text() if ENV_PATH.exists() else ""
        kept = [l for l in existing.splitlines(keepends=True)
                if not l.startswith(f"{HASH_KEY}=")]
        ENV_PATH.write_text("".join(kept) + line)
        print(f"\n✓ wrote hash to {ENV_PATH.relative_to(_REPO_ROOT)} (git-ignored).")
        print("  Load it before running Streamlit, e.g.:")
        print("    set -a && source .env && set +a && streamlit run app/streamlit_app.py")
    else:
        print("\nSet it in your environment before launching:")
        print(f"    export {HASH_KEY}={digest}")
    print("\nThe plaintext was never stored. Good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
