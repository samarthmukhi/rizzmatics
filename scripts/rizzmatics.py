#!/usr/bin/env python3
"""Rizzmatics command-line experience.

Runs the full pipeline on a WhatsApp export and prints the ceremonial
boot screen, a real analysis, and the Final Rizzmatics Moment™ — all in
your terminal, where overengineering feels most at home.

Usage:
    python scripts/rizzmatics.py                      # uses the demo export
    python scripts/rizzmatics.py path/to/chat.txt
    python scripts/rizzmatics.py chat.txt --prefix 15 --gap 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation import (  # noqa: E402
    evaluate_classifiers,
    evaluate_regressors,
    prediction_drivers,
)
from src.features import FEATURE_NAMES  # noqa: E402
from src.models import make_regressors  # noqa: E402
from src.parser import parse_file  # noqa: E402
from src.preprocessing import build_dataset  # noqa: E402
from src.rizz import boot_screen, dataset_health, final_moment  # noqa: E402
from src.sessions import detect_sessions  # noqa: E402

DEFAULT_CHAT = _REPO_ROOT / "data" / "demo" / "demo_chat.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rizzmatics CLI.")
    parser.add_argument("chat", nargs="?", default=str(DEFAULT_CHAT),
                        help="Path to a WhatsApp .txt export (default: demo data).")
    parser.add_argument("--prefix", type=int, default=10,
                        help="Early-portion message prefix used for features.")
    parser.add_argument("--gap", type=float, default=6.0,
                        help="Session inactivity gap in hours.")
    args = parser.parse_args()

    print(boot_screen())

    chat_path = Path(args.chat)
    if not chat_path.exists():
        print(f"✗ No such file: {chat_path}")
        print("  Run: python scripts/generate_demo_data.py")
        return 1

    messages = parse_file(chat_path)
    sessions = detect_sessions(messages, inactivity_hours=args.gap)

    status, health_msg = dataset_health(len(sessions))
    print(f"\nDATASET HEALTH: {status}\n  {health_msg}\n")

    try:
        ds = build_dataset(sessions, prefix=args.prefix)
    except ValueError as exc:
        print(f"✗ {exc}")
        return 1

    reg = evaluate_regressors(ds.X, ds.y_regression)
    clf = evaluate_classifiers(ds.X, ds.y_classification)

    for note in reg.notes + clf.notes:
        print(f"⚠ {note}")

    print("\nREGRESSION — predicting the Engagement Index")
    print("─" * 60)
    for name, r in reg.results.items():
        m = r.metrics
        print(f"  {name:20s}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R²={m['R2']:+.3f}")
    print(f"  → best: {reg.best_model}")

    print("\nCLASSIFICATION — predicting HIGH engagement")
    print("─" * 60)
    for name, r in clf.results.items():
        m = r.metrics
        print(f"  {name:20s}  acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}")
    print(f"  → best: {clf.best_model}")

    print("\nPREDICTION DRIVERS (permutation importance)")
    print("─" * 60)
    drivers = prediction_drivers(
        make_regressors()[reg.best_model], ds.X, ds.y_regression,
        task="regression", top_k=6,
    )
    for d in drivers:
        arrow = {"up": "↑", "down": "↓", "flat": "·"}[d.direction]
        print(f"  {arrow} {d.feature:32s} {d.importance:+.4f}")

    print()
    print(final_moment({
        "messages": len(messages), "sessions": len(sessions),
        "features": len(FEATURE_NAMES) + 3, "models": 4,
        "predictions": len(ds),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
