"""Prove the private layer stays physically out of the public repo."""

import subprocess
from pathlib import Path

from scripts.check_no_private_leak import find_leaks

REPO = Path(__file__).resolve().parents[1]


def test_no_private_leak_in_would_be_public_tree():
    leaks = find_leaks()
    assert leaks == [], f"private content is exposed: {leaks}"


def test_private_directory_is_not_tracked():
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True, check=True)
    tracked = out.stdout.splitlines()
    assert not any(p.startswith("private/") for p in tracked), \
        "the private/ package is tracked by git!"


def test_secret_files_are_not_tracked():
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True, check=True)
    tracked = set(out.stdout.splitlines())
    assert ".env" not in tracked
    assert not any(p.endswith("secrets.toml") for p in tracked)


def test_gitignore_blocks_private_and_secrets():
    gi = (REPO / ".gitignore").read_text()
    assert "/private/" in gi
    assert "secrets.toml" in gi


def test_public_component_source_has_no_real_protocol_vocabulary():
    # Belt-and-suspenders: the public renderer/gate/engine must be content-free.
    tokens_file = REPO / "private" / "forbidden_tokens.txt"
    if not tokens_file.exists():
        return  # public CI: nothing to check against
    tokens = [t.strip().lower() for t in tokens_file.read_text().splitlines()
              if t.strip() and not t.startswith("#")]
    for rel in ("app/components/lorekit.py", "app/components/gate.py",
                "app/components/private_view.py", "app/streamlit_app.py"):
        text = (REPO / rel).read_text().lower()
        for tok in tokens:
            assert tok not in text, f"{rel} contains private token {tok!r}"
