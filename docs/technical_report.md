# Rizzmatics — Technical Report

*A scientifically conservative account of what the system does, what the
experiments show, and — emphatically — what they do not.*

All numbers below are 5×5 repeated cross-validation on the bundled synthetic
demo data (`data/demo/demo_chat.txt`, seed 42), reported as **mean ± standard
deviation across folds**. Regenerate everything with
`python scripts/run_experiment.py --experiment all`.

> **Headline caveat, stated once and meant throughout:** Performance on synthetic
> data demonstrates that the pipeline can recover structured relationships in the
> generated environment. **It does not establish generalization to real human
> conversations.**

---

## 1. Research Question
Given only the **early portion** of a conversation (its first *N* messages), can
measurable conversational signals predict the **engagement of the later
portion**? Engagement here is a defined behavioral proxy, not a feeling.

## 2. Motivation
The project is a deliberately over-engineered response to interpersonal
uncertainty. The serious core is a genuine question about whether early
conversational behavior carries information about later conversational behavior.
The unserious wrapper is everything else.

## 3. Data Representation
WhatsApp `.txt` exports are parsed into typed messages
(`timestamp, sender, message, is_system_message`). The parser handles iOS and
Android layouts, `DD/MM` vs `MM/DD` (auto-detected), 12/24-hour time, multiline
messages, media placeholders, system lines, and invisible formatting marks.

## 4. Session Detection
Messages are split into sessions by an inactivity gap (default 6 hours,
configurable). System messages are excluded by default.

## 5. Feature Engineering
Thirty features across five families: **volume**, **participation**,
**response-latency**, **linguistic**, **temporal**. Every feature is computable
from an arbitrary message slice — the property that makes leakage-safe prefix
modeling possible. Undefined quantities are `NaN` (median-imputed inside the
model pipeline), never fabricated zeros.

## 6. Engagement Target
The **Conversational Engagement Index** — a configurable weighted combination of
normalized session duration, message volume, back-and-forth exchange,
participation balance, and persistence. It is a research-defined behavioral
proxy, **not** a measure of connection, attraction, or compatibility.

## 7. Prediction Setup
Two tasks: **regression** (predict the continuous engagement index) and
**classification** (predict HIGH engagement, defined by a configurable
percentile, default 75th).

## 8. Leakage Prevention
Features come from the **first *N* messages**; the target comes from the
**entire** session. Sessions without more than *N* messages are dropped (no
future to predict). Preprocessing is refit inside every CV fold. Enforced by
`tests/test_leakage.py` and audited in Phase 15.

## 9. Models
Baselines (mean / majority), Linear/Logistic Regression, Random Forest,
Gradient Boosting. No neural networks — the question is whether *simple,
interpretable* signals carry information. Headline results use Random Forest.

## 10. Evaluation Methodology
Repeated K-fold / stratified K-fold, metrics computed **per fold** and reported
as mean ± std. With ~31 sessions these are fragile point estimates; the standard
deviations are wide and are the honest part of the result.

## 11. Synthetic Data Generation
Sessions are generated from conversational **archetypes** (balanced marathon,
one-sided, rapid-fire, delayed slow-burn, transactional, technical deep-dive,
uneven participation). See Phase 16 for the circularity audit. A second,
**noisy** generator weakens the latent→feature link as a graceful-degradation
control.

## 12. Sanity Checks
| Condition | R² | ROC-AUC |
|---|---|---|
| **Normal** (real targets) | **0.532 ± 0.349** | **0.889 ± 0.126** |
| **Shuffled target** | −0.686 ± 1.388 | 0.305 ± 0.232 |
| **Null control** (n=120, random features & target) | −0.142 ± 0.122 | 0.435 ± 0.101 |

Both controls collapse to (or below) chance, exactly as a valid pipeline must.
This is the primary evidence that the benchmark is trustworthy enough to discuss.

## 13. Ablation Results
| Feature family | R² | ROC-AUC |
|---|---|---|
| all_features (30) | 0.532 ± 0.349 | 0.889 ± 0.126 |
| **participation (6)** | **0.625 ± 0.237** | **0.891 ± 0.086** |
| response_latency (4) | 0.393 ± 0.450 | 0.886 ± 0.122 |
| volume (5) | −0.525 ± 0.617 | 0.722 ± 0.236 |
| linguistic (9) | −0.778 ± 0.720 | 0.411 ± 0.249 |
| temporal (6) | −0.413 ± 0.624 | 0.355 ± 0.183 |
| weak_minimal (2) | −0.756 ± 0.753 | 0.322 ± 0.196 |

Predictive information is concentrated in **participation** and
**response-latency**. Linguistic and temporal families perform at or below
chance — correctly, because the generator encodes no archetype signal into
message content or calendar time. The model finds signal where signal exists and
not where it doesn't.

## 14. Prefix-Length Results
| Window | n | R² | ROC-AUC |
|---|---|---|---|
| first 3 | 36 | 0.045 ± 0.538 | 0.753 ± 0.198 |
| first 5 | 34 | 0.522 ± 0.438 | 0.848 ± 0.136 |
| first 10 | 31 | 0.532 ± 0.349 | 0.889 ± 0.126 |
| first 20 | 25 | 0.190 ± 0.641 | 0.850 ± 0.241 |
| first 30 | 18 | 0.137 ± 0.616 | 0.767 ± 0.250 |
| first 50% | 38 | 0.657 ± 0.318 | 0.958 ± 0.106 |

Classification signal appears **very early** — above chance by the first 3
messages, strong by the first 5. **Critical confound:** larger fixed prefixes
retain fewer, longer sessions (n falls 36→18), so the apparent dip at 20–30
messages conflates window length with a shrinking, changing cohort. `n` must be
read alongside every point. The first-50% window keeps all 38 sessions.

## 15. Robustness Results
Leave-one-group-out (remove one family, keep the rest):

| Removed | R² | ROC-AUC |
|---|---|---|
| nothing (all) | 0.532 ± 0.349 | 0.889 ± 0.126 |
| − participation | **−0.307 ± 0.632** | **0.612 ± 0.272** |
| − response_latency | 0.549 ± 0.347 | 0.898 ± 0.118 |
| − volume | 0.539 ± 0.365 | 0.863 ± 0.138 |
| − linguistic | 0.564 ± 0.268 | 0.928 ± 0.106 |
| − temporal | 0.519 ± 0.398 | 0.898 ± 0.126 |

**One family is doing most of the work.** Removing participation collapses the
model (AUC 0.89→0.61, R² 0.53→−0.31); removing any other family leaves it
essentially unchanged (sometimes marginally better, by dropping noise). Results
are stable across 3h/6h inactivity gaps and degrade at 12h; they shift with the
engagement-weight definition and the classification percentile (see the
robustness experiment record).

## 16. Limitations
- **Synthetic data only.** Every number above describes a generated environment.
- **Tiny sample** (~31 sessions). Error bars are wide; nothing is definitive.
- **Circularity.** Features and target share a latent cause (the archetype), so
  the task is closer to latent-label recovery than open-world prediction.
- **Single dominant family.** The result largely rests on participation features.
- **Content is uninformative here** by construction — a real chat might differ,
  and that is untested.

## 17. Privacy
All processing is local. No external APIs, cloud, or telemetry. `.gitignore`
blocks `*.txt`, `*.csv`, `*.json`, and `/data/*`. Experiment records store only
aggregate numbers (audited: `tests/test_research_safety.py`). A safety gate warns
when input looks like a real, non-demo conversation.

## 18. What the Results Actually Mean
The pipeline is implemented correctly: it recovers real structured relationships
when they exist, collapses to chance when they don't (shuffled target, null
control), uses the intended signal families, and reports honest uncertainty. As
an *engineering* artifact, it works.

## 19. What They Absolutely Do NOT Mean
They do **not** show that Rizzmatics can predict real human conversational
engagement. They do **not** measure attraction, feelings, interest, or
compatibility. They do **not** justify any statement about a specific person.
Fast replies are not affection; long chats are not romance; a signal is not a
meaning. **The system cannot tell you whether someone likes you.**

## 20. Future Work
- Validate on **real, consented** conversational data (behind the safety gate).
- Break the circularity: generators (or datasets) where features and target do
  not share an obvious latent label.
- Larger samples for stable estimates and honest confidence intervals.
- Per-participant and cross-conversation generalization tests.

---

*Rizzmatics is intentionally funny. The science under the joke is treated
seriously. Never manufacture significance; never hide poor results; never turn a
behavioral signal into a claim about a person. Talk to the human.*
