"""Experiment registry — reproducible, file-based records.

Every experiment writes one JSON file capturing exactly what was run: the code
path, the seed, the dataset fingerprint, the methodology, and the metrics with
uncertainty. No database — just JSON on disk plus a flat CSV index — so results
are diffable, greppable, and safe to commit (they contain no raw conversation
text, only aggregate numbers).
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .experiment import CVMetrics

__all__ = [
    "RESULTS_DIR",
    "dataset_version",
    "cvmetrics_to_dict",
    "make_record",
    "save_experiment",
    "load_experiments",
]

RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"


def dataset_version(path: str | Path) -> dict:
    """Fingerprint a dataset file so a record pins the exact data it used."""
    p = Path(path)
    if not p.exists():
        return {"name": str(p), "sha256_12": None, "n_bytes": 0}
    raw = p.read_bytes()
    return {
        "name": p.name,
        "sha256_12": hashlib.sha256(raw).hexdigest()[:12],
        "n_bytes": len(raw),
    }


def _round_metrics(summary: dict, ndigits: int = 6) -> dict:
    """Round metric floats for clean, bit-reproducible records.

    Six decimals is far more precision than a cross-validated R²±std deserves,
    and it makes saved records deterministic across BLAS threading (whose
    summation order otherwise wobbles the last ~1e-16 of a std).
    """
    out = {}
    for metric, stats in summary.items():
        out[metric] = {
            k: (round(v, ndigits) if isinstance(v, float) else v)
            for k, v in stats.items()
        }
    return out


def cvmetrics_to_dict(cv: CVMetrics | None) -> dict | None:
    """Serialize a CVMetrics into a plain dict (mean/std/n per metric)."""
    if cv is None:
        return None
    return {
        "task": cv.task,
        "n_samples": cv.n_samples,
        "n_splits": cv.n_splits,
        "n_repeats": cv.n_repeats,
        "cross_validated": cv.cross_validated,
        "metrics": _round_metrics(cv.summary()),
        "notes": cv.notes,
    }


def make_record(
    experiment_id: str,
    *,
    seed: int,
    dataset: dict,
    prefix,
    feature_groups: list[str],
    model: str,
    hyperparameters: dict,
    methodology: dict,
    results: list[dict],
    n_samples: int,
) -> dict:
    """Assemble the canonical experiment record schema."""
    return {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "dataset": dataset,
        "prefix": prefix,
        "feature_groups": feature_groups,
        "model": model,
        "hyperparameters": hyperparameters,
        "methodology": methodology,
        "n_samples": n_samples,
        "results": results,
    }


def save_experiment(record: dict, out_dir: str | Path = RESULTS_DIR) -> Path:
    """Write ``<experiment_id>.json`` and append a row to ``index.csv``.

    Re-running the same experiment id overwrites its JSON deterministically
    (only the timestamp differs), and refreshes its single index row.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{record['experiment_id']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=False), encoding="utf-8")
    _update_index(record, out)
    return path


def _update_index(record: dict, out: Path) -> None:
    index = out / "index.csv"
    rows: dict[str, dict] = {}
    if index.exists():
        with index.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows[row["experiment_id"]] = row
    rows[record["experiment_id"]] = {
        "experiment_id": record["experiment_id"],
        "timestamp_utc": record["timestamp_utc"],
        "dataset": record["dataset"].get("name"),
        "dataset_sha256_12": record["dataset"].get("sha256_12"),
        "seed": record["seed"],
        "prefix": record["prefix"],
        "model": record["model"],
        "n_samples": record["n_samples"],
        "n_results": len(record["results"]),
    }
    fields = ["experiment_id", "timestamp_utc", "dataset", "dataset_sha256_12",
              "seed", "prefix", "model", "n_samples", "n_results"]
    with index.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)


def load_experiments(out_dir: str | Path = RESULTS_DIR) -> list[dict]:
    """Load all saved experiment records, newest first."""
    out = Path(out_dir)
    if not out.exists():
        return []
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in out.glob("*.json")]
    records.sort(key=lambda r: r.get("timestamp_utc", ""), reverse=True)
    return records
