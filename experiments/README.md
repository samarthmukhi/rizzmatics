# experiments/

Reproducible research experiments for Rizzmatics.

Results are written to `experiments/results/` as JSON (one file per experiment)
plus a flat `index.csv`. **These outputs are git-ignored on purpose** — they are
fully regenerable and the repo prefers reproducibility over committed artifacts.
Regenerate everything from scratch with:

```bash
python scripts/run_experiment.py --experiment all
```

Individual experiments:

```bash
python scripts/run_experiment.py --experiment baseline      # RF on the demo data
python scripts/run_experiment.py --experiment shuffled      # target-permutation control
python scripts/run_experiment.py --experiment null          # negative control (must be ~chance)
python scripts/run_experiment.py --experiment ablation      # per-feature-family study
python scripts/run_experiment.py --experiment prefix        # observation-window sweep
python scripts/run_experiment.py --experiment robustness    # leave-one-group-out + sensitivity
```

Every record captures the seed, dataset fingerprint (SHA-256), prefix length,
model, hyperparameters, CV methodology, sample count, and metrics **with
uncertainty (mean ± std)**. Records contain only aggregate numbers — never any
raw conversation text.
