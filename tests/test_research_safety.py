"""Phase 19 — real-data safety gate tests."""

from pathlib import Path

from src.research.safety import (
    SAFETY_WARNING,
    audit_no_raw_text,
    count_conversation_lines,
    is_probably_real_conversation,
    looks_like_whatsapp,
)

DEMO = Path("data/demo/demo_chat.txt")


# --------------------------------------------------------------------------- #
# The gate.
# --------------------------------------------------------------------------- #
def test_warning_text_is_explicit():
    assert "locally" in SAFETY_WARNING
    assert "Do not upload or commit private conversations" in SAFETY_WARNING


def test_demo_is_not_flagged_as_real():
    assert is_probably_real_conversation(DEMO.read_text()) is False


def test_real_looking_conversation_is_flagged():
    real = (
        "12/08/2026, 14:32 - Priya: hey did you get home ok\n"
        "12/08/2026, 14:33 - Rahul: yeah just got back\n"
        "12/08/2026, 14:34 - Priya: good night\n"
    )
    assert is_probably_real_conversation(real) is True


def test_non_conversation_text_is_not_flagged():
    assert is_probably_real_conversation("shopping list\nmilk\neggs") is False
    assert looks_like_whatsapp("just a note") is False


def test_count_conversation_lines_on_demo():
    assert count_conversation_lines(DEMO.read_text()) > 100


# --------------------------------------------------------------------------- #
# The audit: artifacts must not carry raw conversation text.
# --------------------------------------------------------------------------- #
def test_experiment_results_contain_no_raw_text():
    offenders = audit_no_raw_text(["experiments/results"])
    assert offenders == [], f"raw text leaked into results: {offenders}"


def test_audit_flags_a_file_that_contains_chat_lines(tmp_path):
    leak = tmp_path / "oops.log"
    leak.write_text(
        "12/08/2026, 14:32 - Priya: this is private\n"
        "12/08/2026, 14:33 - Rahul: yes it is\n"
    )
    offenders = audit_no_raw_text([tmp_path])
    assert str(leak) in offenders


# --------------------------------------------------------------------------- #
# The privacy firewall is actually in .gitignore.
# --------------------------------------------------------------------------- #
def test_gitignore_blocks_raw_data_formats():
    gi = Path(".gitignore").read_text()
    for rule in ("*.txt", "*.csv", "*.json", "/data/*"):
        assert rule in gi, f"missing privacy rule: {rule}"
