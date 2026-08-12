# RIZZMATICS

**Applied mathematics for completely unnecessary flirting.**

```text
╔══════════════════════════════════════════════╗
║                R I Z Z M A T I C S           ║
║                                              ║
║  Applied mathematics for completely          ║
║  unnecessary flirting.                       ║
╚══════════════════════════════════════════════╝

RIZZ:                 ONLINE
STATISTICAL RIGOR:    QUESTIONABLE
OVERENGINEERING:      ENTERPRISE-GRADE
ACTUAL NECESSITY:     0%
EMOTIONAL MATURITY:   PLEASE CONSULT USER
```

> Rizzmatics is applied mathematics for completely unnecessary flirting. It is an
> intentionally overengineered machine-learning system for analyzing
> conversational dynamics, extracting behavioral signals from chat data, and
> testing whether the early part of a conversation can predict future
> conversational engagement. It cannot detect attraction, read minds, or explain
> why someone took four hours to reply. We built it anyway.

---

## The one-sentence version

Rizzmatics turns a conversation into measurable signals and tests whether the
**early part** of that conversation can predict how **engaged** its later part
becomes — because apparently talking to humans wasn't enough.

## What this actually is

Under the jokes, it's a real, leakage-safe ML experiment:

```text
Conversation → Parse → Detect sessions → Extract features → Engagement Index
             → Leakage-safe ML → Predict future engagement → Evaluate → Explain
```

**The research question:** *Given only the early portion of a conversation, can
measurable conversational signals predict the engagement of the later portion?*

## What this is NOT

Rizzmatics does **not** predict attraction, feelings, intentions, or whether you
should text someone. It studies **observable conversational behavior** and
predicts **observable future conversational behavior**. A fast reply is not
attraction. A long conversation is not romance. A signal is not a meaning.

That boundary isn't a limitation we're apologizing for. It's the entire point.

---

## Quickstart

```bash
# 1. Install (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic demo data (no private chats required)
python scripts/generate_demo_data.py

# 3. Prove the science works
pytest

# 4. Launch the Executive Dashboard™
streamlit run app/streamlit_app.py
```

Then open the dashboard, keep "Demo data (safe)" selected, and click around. To
analyze your own chat, export a WhatsApp conversation (`⋮ → More → Export chat →
Without media`) and upload the `.txt`. **It is parsed locally and never saved.**

---

## How it works

| Stage | Module | What it does |
|-------|--------|--------------|
| Parse | [`src/parser.py`](src/parser.py) | WhatsApp `.txt` → typed messages. Handles iOS/Android, DD/MM vs MM/DD (auto-detected), 12/24h, multiline, media, system lines, and iOS's cursed invisible characters. |
| Sessionize | [`src/sessions.py`](src/sessions.py) | Splits the stream into conversations using a configurable inactivity gap (default 6h). |
| Features | [`src/features.py`](src/features.py) | 26 within-session signals + 3 cross-session temporal ones. Runs on *any* message slice — that's what makes leakage prevention possible. |
| Target | [`src/engagement.py`](src/engagement.py) | The configurable **Conversational Engagement Index™** (a behavioral proxy, not a love score). |
| Dataset | [`src/preprocessing.py`](src/preprocessing.py) | Assembles features (from prefixes) and targets (from full sessions), leakage-safe. |
| Models | [`src/models.py`](src/models.py) | Baselines + Linear/Logistic + Random Forest + Gradient Boosting. No neural nets on purpose. |
| Evaluate | [`src/evaluation.py`](src/evaluation.py) | Cross-validated MAE/RMSE/R² and accuracy/precision/recall/F1/AUC, plus permutation-importance drivers. |
| Personality | [`src/rizz.py`](src/rizz.py) | The Oracle™, the Rizz Coefficient™, and the Human Override™. |

## Leakage prevention (non-negotiable)

Never use the future to predict the future.

- **Features** are extracted from the **first N messages** of a session.
- **The target** is computed from the **entire** session.
- Sessions without more than N messages are dropped — there's no future to forecast.
- Preprocessing (imputation, scaling) is refit inside each CV fold, so no stats
  leak across the split.

This is enforced by [`tests/test_leakage.py`](tests/test_leakage.py), which fails
loudly if mutating a conversation's *future* ever changes its *features*.

## Evaluation, honestly

Metrics are cross-validated when the dataset allows, and **explicitly flagged as
in-sample when it doesn't**. If you feed it 14 conversations, it will say:

> BRO, YOU GAVE ME 14 CONVERSATIONS. I am an ML model, not a fortune teller.

No manufacturing impressive statistics from tiny samples.

## Scientific validation

Rizzmatics ships with a full validation battery so you can check *where it's
right, where it's wrong, and how much confidence it deserves*. Full write-up:
[`docs/technical_report.md`](docs/technical_report.md). Regenerate all results:

```bash
python scripts/run_experiment.py --experiment all
```

On the synthetic demo (~31 sessions, first 10 messages, 5×5 CV, mean ± std):

- **Sanity checks** — shuffled-target collapses to AUC 0.31 and the noise-only
  **null control** to AUC 0.44 (chance). If either scored well, we'd suspect
  leakage. They don't.
- **Ablation** — predictive signal lives in **participation** (AUC 0.89) and
  **response-latency** (AUC 0.89). Message content and calendar time score *at
  chance* — correctly, because the generator encodes no signal there.
- **Prefix sweep** — classification signal appears by the **first ~5 messages**;
  the apparent dip at 20–30 is a sample-size confound (fewer, longer sessions),
  reported with `n` at every point.
- **Robustness** — remove participation and the model collapses (AUC 0.89→0.61);
  remove any other family and it barely moves. **One family does most of the work.**

**Generator honesty:** the demo's latent variable is a per-session *archetype*
that drives features **and** the target, so the task is closer to recovering a
hidden label than open-world prediction (see `GENERATOR_METADATA` in
[`scripts/generate_demo_data.py`](scripts/generate_demo_data.py)). With only ~31
sessions, every metric is a fragile point estimate with wide error bars.

> **Performance on synthetic data demonstrates that the pipeline can recover
> structured relationships in the generated environment. It does not establish
> generalization to real human conversations.**

---

## The features (Rizz Engine™ intake)

**Volume** — message counts, characters, message-length stats ·
**Participation** — balance (entropy), dominance share, consecutive runs,
back-and-forth rate · **Response behavior** — median/mean/p90 reply latency,
reply-within-threshold rate · **Linguistic** — word count, question &
exclamation rates, emoji count/density, links, media, lexical diversity ·
**Temporal** — hour of day, weekend, late-night, time since previous session,
rolling volume & session frequency.

Every one is a measurement. None is a diagnosis.

## Things Rizzmatics Cannot Tell You

❌ Does someone like you? ❌ Are they attracted to you? ❌ Are they secretly mad
at you? ❌ Should you text them? ❌ Why did they take four hours to reply?
❌ Are you compatible? ❌ Is this a situationship? ❌ Are you cooked?
❌ Are you cooking? ❌ What did "haha" actually mean?

**Please communicate with the human.**

## Privacy

All real chat processing happens locally. No external APIs, no cloud, no
telemetry, no uploads. `.gitignore` blocks `*.txt`, `*.csv`, `*.json`, and the
`data/` tree. The public repo ships **synthetic** demo data only.

---

## Repository structure

```text
rizzmatics/
├── app/                    # Streamlit dashboard (8 pages, incl. Research Lab)
│   ├── streamlit_app.py
│   └── components/
├── src/                    # The actual science
│   ├── parser.py  sessions.py  features.py  engagement.py
│   ├── preprocessing.py  models.py  evaluation.py  rizz.py
│   └── research/           # sanity · ablation · prefix · robustness ·
│       │                   #   nulldata · registry · safety · experiment
├── tests/                  # 168 tests, incl. leakage + null-control guards
├── scripts/                # generate_demo_data.py · run_experiment.py · rizzmatics.py
├── experiments/            # reproducible experiment records (git-ignored, regenerable)
├── docs/technical_report.md
├── data/demo/              # synthetic exports only
└── requirements.txt  pyproject.toml  LICENSE
```

## The deeper idea

Machine learning can model **what happened**. It cannot model **why**. The same
message is sent for completely different reasons by different people on different
days. Rizzmatics deliberately pushes quantitative analysis of conversation right
up to the boundary where context and human judgment become unavoidable — and
then stops, because that's where software ends and people begin.

```text
Can we determine what the other person actually feels?

NO.

FINAL RECOMMENDATION:

             TALK TO THE HUMAN.
```
