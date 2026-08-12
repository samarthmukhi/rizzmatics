"""Real-data safety gate and raw-text leak auditing.

Two jobs:

1. **Gate** — when the app is handed something that looks like a *real* personal
   conversation (a parseable WhatsApp export that isn't the bundled synthetic
   demo), surface a clear warning. We do not block processing (it's the user's
   data on the user's machine), but we make the privacy expectation explicit.

2. **Audit** — provide a scanner that verifies no raw conversation *content* has
   leaked into artifacts that might be committed or shared (experiment result
   JSONs, logs, etc.). Aggregate numbers are fine; message text is not.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "SAFETY_WARNING",
    "looks_like_whatsapp",
    "is_probably_real_conversation",
    "count_conversation_lines",
    "audit_no_raw_text",
    "SYNTHETIC_PARTICIPANTS",
]

SAFETY_WARNING = (
    "Rizzmatics processes conversational data locally. "
    "Do not upload or commit private conversations."
)

# The fictional cast of the bundled synthetic demos. A real export will contain
# other names.
SYNTHETIC_PARTICIPANTS = {"Alex", "Sam"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_DIR = _REPO_ROOT / "data" / "demo"

# A WhatsApp message line with an actual "Sender: text" body (i.e. real content,
# not just a header). Kept intentionally close to the parser's expectations.
_CONTENT_LINE = re.compile(
    r"^\[?\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4},?\s+\d{1,2}:\d{2}"
    r"[^\n]*?\]?\s*[-–—]?\s*[^:\n]{1,60}:\s+\S",
)


def looks_like_whatsapp(text: str, *, min_lines: int = 3) -> bool:
    """True if the text contains at least ``min_lines`` WhatsApp message lines."""
    return count_conversation_lines(text) >= min_lines


def count_conversation_lines(text: str) -> int:
    """Count lines that look like WhatsApp message content."""
    return sum(1 for line in text.splitlines() if _CONTENT_LINE.match(line))


def _human_senders(text: str) -> set[str]:
    """Distinct non-system senders, via the real parser (not a fragile regex)."""
    from ..parser import parse_chat  # local import avoids any import cycle

    return {
        m.sender for m in parse_chat(text)
        if m.sender is not None and not m.is_system_message
    }


def is_probably_real_conversation(text: str) -> bool:
    """Heuristic: does this look like a *real* (non-demo) personal chat?

    True when the text parses as a WhatsApp conversation and its participants are
    not the synthetic demo cast. This is intentionally conservative — it would
    rather warn on a synthetic file with unusual names than stay silent on a real
    one.
    """
    if not looks_like_whatsapp(text):
        return False
    senders = _human_senders(text)
    if not senders:
        return False
    # If every human sender is part of the synthetic cast, treat as demo.
    return not senders.issubset(SYNTHETIC_PARTICIPANTS)


def audit_no_raw_text(paths: list[str | Path], *, threshold: int = 1) -> list[str]:
    """Return files that appear to contain raw conversation content.

    Scans each path (files, or all files under a directory) and flags any that
    contain ``threshold`` or more WhatsApp content lines. Use this on artifacts
    that must never carry raw text — experiment results, logs, model dumps.
    """
    offenders: list[str] = []
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else (p.rglob("*") if p.is_dir() else [])
        for f in files:
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeError):
                continue
            if count_conversation_lines(text) >= threshold:
                offenders.append(str(f))
    return offenders
