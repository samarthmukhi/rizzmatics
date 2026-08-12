#!/usr/bin/env python3
"""Synthetic WhatsApp export generator.

Nobody should have to hand Rizzmatics a real conversation to see it work. This
script fabricates a WhatsApp ``.txt`` export from scratch — realistic-looking,
completely fictional, and safe to commit to a public repo.

It stitches together conversational *archetypes* (balanced marathons, one-sided
dead-ends, rapid-fire volleys, delayed slow-burns, transactional check-ins,
long technical deep-dives) across a span of days, so downstream sessionization,
feature extraction, and modeling all have something varied to chew on.

Usage:
    python scripts/generate_demo_data.py                 # writes data/demo/*
    python scripts/generate_demo_data.py --seed 7 --days 120
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

# Fictional participants. Any resemblance to real people is the whole joke's
# absence — these two do not exist.
PARTICIPANTS = ("Alex", "Sam")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "data" / "demo"

# --------------------------------------------------------------------------- #
# Message content pools (generic CS/college/startup chatter)
# --------------------------------------------------------------------------- #
OPENERS = [
    "yo", "hey", "hii", "you up?", "guess what", "ok so", "quick q",
    "random but", "bro", "omg", "wait", "so",
]
CHATTER = [
    "did you finish the assignment", "this lecture was so dry ngl",
    "i think i finally understand recursion", "the professor lowkey ate today",
    "wanna grab food later", "i've been debugging this for 2 hours",
    "the wifi in the library is criminal", "should we start that project",
    "i can't stop thinking about this idea", "just pushed to main sorry",
    "the demo actually worked first try", "i pulled an all nighter oops",
    "coffee is the only thing keeping me alive", "how was your day",
    "i got the internship!!", "we should build something this weekend",
    "the campus tour was actually fun", "i keep getting merge conflicts",
    "reading week cannot come soon enough", "found a bug in my own code again",
]
QUESTIONS = [
    "wait what did you mean by that?", "are you free tonight?",
    "did you see the email?", "what time works for you?",
    "have you started studying yet?", "should we meet at the union?",
    "is the deadline friday or monday?", "wanna call?",
]
EXCITED = [
    "LETS GOOO", "no way that's insane", "i'm literally so hyped",
    "that's actually huge", "STOP that's amazing", "okay this is fire",
]
EMOJI_MSGS = ["haha 😂", "🔥🔥", "😭😭😭 real", "lol 💀", "yesss 🙌", "aw 🥹"]
LINKS = [
    "check this out https://example.com/thing",
    "the repo is at https://github.com/example/project",
    "www.example.edu/deadlines",
]
MEDIA = ["<Media omitted>", "image omitted", "sticker omitted"]
CLOSERS = ["gtg", "ttyl", "ok cya", "goodnight", "talk later", "brb class"]


def _sentence(rng: random.Random, allow_special: bool = True) -> str:
    """Produce one synthetic message body."""
    roll = rng.random()
    if allow_special and roll < 0.10:
        return rng.choice(EMOJI_MSGS)
    if allow_special and roll < 0.16:
        return rng.choice(LINKS)
    if allow_special and roll < 0.20:
        return rng.choice(MEDIA)
    if roll < 0.40:
        return rng.choice(QUESTIONS)
    if roll < 0.50:
        return rng.choice(EXCITED)
    return rng.choice(CHATTER)


# --------------------------------------------------------------------------- #
# Archetypes: each returns (sender_sequence, latency_seconds_sequence)
# --------------------------------------------------------------------------- #
def _archetype(rng: random.Random, kind: str):
    a, b = PARTICIPANTS
    if kind == "balanced_marathon":
        n = rng.randint(30, 70)
        senders = [a if i % 2 == 0 else b for i in range(n)]
        # occasionally double-text
        latencies = [rng.randint(15, 240) for _ in range(n)]
    elif kind == "one_sided":
        n = rng.randint(3, 8)
        senders = [a] * (n - 1) + [b]  # a keeps texting, b barely replies
        latencies = [rng.randint(60, 3600) for _ in range(n)]
    elif kind == "rapid_fire":
        n = rng.randint(20, 45)
        senders = [rng.choice(PARTICIPANTS) for _ in range(n)]
        latencies = [rng.randint(3, 30) for _ in range(n)]
    elif kind == "delayed_slow_burn":
        n = rng.randint(8, 18)
        senders = [a if i % 2 == 0 else b for i in range(n)]
        latencies = [rng.randint(1200, 7000) for _ in range(n)]
    elif kind == "transactional":
        n = rng.randint(2, 5)
        senders = [a if i % 2 == 0 else b for i in range(n)]
        latencies = [rng.randint(30, 600) for _ in range(n)]
    elif kind == "technical_deepdive":
        n = rng.randint(25, 55)
        senders = [a if i % 3 == 0 else b for i in range(n)]  # uneven, b talks more
        latencies = [rng.randint(20, 400) for _ in range(n)]
    else:  # uneven_participation
        n = rng.randint(15, 30)
        senders = [a if rng.random() < 0.75 else b for i in range(n)]
        latencies = [rng.randint(30, 900) for _ in range(n)]
    return senders, latencies


ARCHETYPES = [
    "balanced_marathon", "one_sided", "rapid_fire", "delayed_slow_burn",
    "transactional", "technical_deepdive", "uneven_participation",
]


def _fmt(ts: datetime, sender: str, text: str) -> str:
    """Android-style WhatsApp line: DD/MM/YYYY, HH:MM - Sender: text."""
    return f"{ts.strftime('%d/%m/%Y, %H:%M')} - {sender}: {text}"


def generate_export(seed: int = 42, days: int = 90, start: datetime | None = None) -> str:
    """Generate a full synthetic WhatsApp export as a single string."""
    rng = random.Random(seed)
    start = start or datetime(2026, 4, 1, 9, 0)

    # Open with a realistic WhatsApp system line (no sender -> is_system_message).
    lines = [
        f"{(start - timedelta(minutes=5)).strftime('%d/%m/%Y, %H:%M')} - "
        "Messages and calls are end-to-end encrypted. No one outside of this "
        "chat, not even WhatsApp, can read or listen to them."
    ]

    cursor = start
    day = 0
    while day < days:
        # Some days have no conversation at all (realistic gaps).
        if rng.random() < 0.25:
            day += rng.randint(1, 3)
            cursor = start + timedelta(days=day, hours=rng.randint(8, 22))
            continue

        kind = rng.choice(ARCHETYPES)
        senders, latencies = _archetype(rng, kind)

        # Start the session at a plausible hour.
        hour = rng.choice([9, 11, 13, 15, 17, 19, 21, 22, 23, 0, 1])
        cursor = start + timedelta(days=day, hours=hour, minutes=rng.randint(0, 59))

        for i, (sender, lat) in enumerate(zip(senders, latencies)):
            cursor += timedelta(seconds=lat)
            if i == 0:
                text = f"{rng.choice(OPENERS)} {_sentence(rng)}"
            elif i == len(senders) - 1 and rng.random() < 0.4:
                text = rng.choice(CLOSERS)
            else:
                text = _sentence(rng)
            lines.append(_fmt(cursor, sender, text))

        # Advance to the next day (or skip a few).
        day += rng.randint(1, 3)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic WhatsApp demo data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--out", type=Path, default=DEMO_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # A rich export for the full experience.
    main_path = args.out / "demo_chat.txt"
    main_path.write_text(generate_export(seed=args.seed, days=args.days), encoding="utf-8")

    # A deliberately tiny one to demo the "bro, you gave me 14 conversations" path.
    small_path = args.out / "demo_small.txt"
    small_path.write_text(generate_export(seed=args.seed + 1, days=12), encoding="utf-8")

    for p in (main_path, small_path):
        n_lines = p.read_text(encoding="utf-8").count("\n")
        print(f"✓ wrote {p.relative_to(REPO_ROOT)}  ({n_lines} lines)")

    print("\nSynthetic data ready. No real humans were quantified in the making of this file.")


if __name__ == "__main__":
    main()
