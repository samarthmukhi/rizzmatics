"""Rizzmatics — the Executive Dashboard™.

A Streamlit front-end for an ML experiment that absolutely did not need one.
Run it with:

    streamlit run app/streamlit_app.py

All processing is local. Uploaded chats are parsed in memory and never saved.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app.components.pipeline import (  # noqa: E402
    RizzBundle,
    load_demo_text,
    run_pipeline,
    session_full_features,
    session_prefix_features,
)
from app.components.theme import disclaimer, hero, inject_theme  # noqa: E402
from src.engagement import ENGAGEMENT_DISCLAIMER  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402
from src.rizz import (  # noqa: E402
    RIZZ_DISCLAIMER,
    boot_screen,
    confidence_translation,
    dataset_health,
    final_moment,
    human_override,
    is_relationship_question,
    model_outcome_verdict,
    oracle_card,
    rizz_coefficient,
    rizz_engine_readout,
    things_rizzmatics_cannot_tell_you,
)

st.set_page_config(page_title="Rizzmatics", page_icon="💘", layout="wide")

PAGES = [
    "🏢 Executive Dashboard™",
    "🔎 Conversation Explorer",
    "📈 Signal Analytics",
    "🔮 Rizzmatics Oracle™",
    "🧪 Model Lab",
    "📐 Methodology",
    "🔒 Privacy Center",
]


# --------------------------------------------------------------------------- #
# Sidebar: data source + knobs
# --------------------------------------------------------------------------- #
def sidebar() -> tuple[str, dict, str]:
    st.sidebar.markdown("### RIZZMATICS")
    st.sidebar.caption("Applied mathematics for completely unnecessary flirting.")

    source = st.sidebar.radio(
        "Data source", ["Demo data (safe)", "Upload your own .txt"],
        help="Uploaded chats are processed locally and never saved.",
    )
    if source == "Demo data (safe)":
        text = load_demo_text()
    else:
        up = st.sidebar.file_uploader("WhatsApp .txt export", type=["txt"])
        text = up.read().decode("utf-8", errors="ignore") if up else ""
        if not text:
            st.sidebar.info("Waiting for a .txt export. Nothing leaves your machine.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Parameters")
    inactivity = st.sidebar.slider("Session inactivity gap (hours)", 1.0, 24.0, 6.0, 0.5)
    prefix = st.sidebar.slider("Early-portion prefix (messages)", 3, 40, 10, 1)
    high_pct = st.sidebar.slider("HIGH-engagement percentile", 50, 95, 75, 5)

    st.sidebar.markdown("#### Engagement weights")
    w_dur = st.sidebar.slider("Duration", 0.0, 1.0, 0.30, 0.05)
    w_vol = st.sidebar.slider("Volume", 0.0, 1.0, 0.25, 0.05)
    w_bi = st.sidebar.slider("Bidirectional", 0.0, 1.0, 0.20, 0.05)
    w_bal = st.sidebar.slider("Balance", 0.0, 1.0, 0.15, 0.05)
    w_per = st.sidebar.slider("Persistence", 0.0, 1.0, 0.10, 0.05)

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", PAGES)

    params = dict(
        inactivity_hours=inactivity, prefix=prefix, high_percentile=float(high_pct),
        weights=(w_dur, w_vol, w_bi, w_bal, w_per), dayfirst=None,
    )
    return text, params, page


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sessions_df(bundle: RizzBundle) -> pd.DataFrame:
    rows = [{
        "session_id": s.session_id, "start_time": s.start_time,
        "duration_min": round(s.duration_minutes, 1), "messages": s.message_count,
        "participants": ", ".join(s.participants),
    } for s in bundle.sessions]
    df = pd.DataFrame(rows)
    if not bundle.engagement.empty:
        df = df.merge(
            bundle.engagement[["engagement_index"]].reset_index(),
            on="session_id", how="left",
        )
    return df


def _overall_rizz(bundle: RizzBundle) -> float:
    if bundle.features_full.empty:
        return 0.0
    scores = [rizz_coefficient(row._asdict() if hasattr(row, "_asdict") else dict(row))
              for _, row in bundle.features_full.iterrows()]
    return round(sum(scores) / len(scores), 1) if scores else 0.0


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_executive(bundle: RizzBundle) -> None:
    hero("The Executive Dashboard™ — for stakeholders who overthink at scale.")
    if bundle.n_messages == 0:
        st.warning("No data loaded. Pick the demo or upload a .txt in the sidebar.")
        return

    sdf = _sessions_df(bundle)
    avg_dur = sdf["duration_min"].mean() if not sdf.empty else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Messages analyzed", f"{bundle.n_messages:,}")
    c2.metric("Sessions detected", f"{bundle.n_sessions:,}")
    c3.metric("Avg session (min)", f"{avg_dur:.1f}")
    c4.metric("Features engineered", len(FEATURE_NAMES) + 3)
    c5.metric("Rizz Coefficient™", f"{_overall_rizz(bundle):.0f}")

    status, msg = dataset_health(bundle.n_sessions)
    (st.success if status in ("JUICY",) else st.info)(f"**DATASET HEALTH: {status}** — {msg}")

    if not sdf.empty and "engagement_index" in sdf:
        st.markdown("#### Engagement Index over time")
        fig = px.area(
            sdf.sort_values("start_time"), x="start_time", y="engagement_index",
            markers=True,
        )
        fig.update_traces(line_color="#ff4d8d", fillcolor="rgba(255,77,141,0.15)")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    disclaimer(ENGAGEMENT_DISCLAIMER)
    disclaimer(RIZZ_DISCLAIMER)


def page_explorer(bundle: RizzBundle) -> None:
    hero("Conversation Explorer — inspect a single session, message by message.")
    if not bundle.sessions:
        st.warning("No sessions to explore.")
        return

    sdf = _sessions_df(bundle)
    sid = st.selectbox("Session", sdf["session_id"].tolist())
    session = next(s for s in bundle.sessions if s.session_id == sid)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Messages", session.message_count)
    c2.metric("Duration (min)", f"{session.duration_minutes:.1f}")
    c3.metric("Participants", len(session.participants))
    if not bundle.engagement.empty and sid in bundle.engagement.index:
        c4.metric("Engagement", f"{bundle.engagement.loc[sid, 'engagement_index']:.2f}")

    msg_df = pd.DataFrame([{
        "time": m.timestamp.strftime("%Y-%m-%d %H:%M"),
        "sender": m.sender or "· system ·", "message": m.message,
    } for m in session.messages])
    st.dataframe(msg_df, width="stretch", height=360)

    with st.expander("Rizz Engine™ readout for this session"):
        readout = rizz_engine_readout(session_full_features(bundle, sid))
        st.json(readout)


def page_analytics(bundle: RizzBundle) -> None:
    hero("Signal Analytics — distributions and relationships between features.")
    ff = bundle.features_full
    if ff.empty:
        st.warning("No features to analyze.")
        return

    numeric = [c for c in FEATURE_NAMES if c in ff.columns and ff[c].notna().any()]
    col1, col2 = st.columns(2)
    with col1:
        feat = st.selectbox("Feature distribution", numeric,
                            index=numeric.index("n_messages") if "n_messages" in numeric else 0)
        fig = px.histogram(ff, x=feat, nbins=20, color_discrete_sequence=["#7b61ff"])
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with col2:
        if "engagement_index" in ff.columns:
            st.markdown("**Correlation with Engagement Index**")
            corr = (ff[numeric + ["engagement_index"]].corr()["engagement_index"]
                    .drop("engagement_index").dropna().sort_values())
            cdf = corr.reset_index()
            cdf.columns = ["feature", "correlation"]
            fig2 = px.bar(cdf, x="correlation", y="feature", orientation="h",
                          color="correlation", color_continuous_scale="RdBu",
                          range_color=[-1, 1])
            fig2.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, width="stretch")

    st.caption("Correlation is descriptive, not causal, and certainly not psychological.")


def page_oracle(bundle: RizzBundle) -> None:
    hero("The Rizzmatics Oracle™ — a Random Forest in a trench coat.")

    st.markdown("#### Ask the Oracle")
    q = st.text_input("Your question for the Oracle", placeholder="does she like me?")
    if q:
        if is_relationship_question(q):
            st.error("**THE HUMAN OVERRIDE™ has been triggered.**")
            st.code(human_override(), language=None)
        else:
            st.info("The Oracle only forecasts *observable conversational engagement*. "
                    "For that, pick a session below.")

    st.markdown("---")
    if bundle.error or not bundle.dataset:
        st.warning(bundle.error or "Not enough data to forecast. Feed the machine more.")
        return

    ds = bundle.dataset
    clf = bundle.clf_report
    best = clf.best_model
    scores = clf.results[best].scores
    id_to_score = dict(zip(ds.session_ids, scores)) if scores is not None else {}

    sid = st.selectbox("Forecast a session", ds.session_ids)
    actual_high = bool(ds.y_classification.loc[sid])
    prob = float(id_to_score.get(sid, 0.5))
    label = "HIGH CONVERSATIONAL ENGAGEMENT" if prob >= 0.5 else "LOW CONVERSATIONAL ENGAGEMENT"

    col1, col2 = st.columns([3, 2])
    with col1:
        st.code(oracle_card(label, prob, "Something statistically interesting."), language=None)
    with col2:
        st.metric("Rizz Coefficient™ (this session)",
                  f"{rizz_coefficient(session_full_features(bundle, sid)):.0f}")
        head, sub = model_outcome_verdict(prob >= 0.5, actual_high)
        st.markdown(f"### {head}")
        st.caption(sub)

    st.markdown("#### How Sure Are We, Bro?™")
    st.code(confidence_translation(prob), language=None)
    disclaimer(f"Prediction uses ONLY the first {ds.prefix} messages of this session. "
               "The Oracle has not seen how it ends.")


def page_model_lab(bundle: RizzBundle) -> None:
    hero("Model Lab — where we check whether the machine actually learned anything.")
    if bundle.error or not bundle.reg_report:
        st.warning(bundle.error or "Not enough data to train models.")
        return

    reg, clf = bundle.reg_report, bundle.clf_report
    for note in reg.notes + clf.notes:
        st.warning(note)
    st.caption(f"Cross-validation: regression folds={reg.n_splits}, "
               f"classification folds={clf.n_splits}. Samples: {reg.n_samples}.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Regression — predict the Engagement Index**")
        rdf = pd.DataFrame({n: r.metrics for n, r in reg.results.items()}).T[["MAE", "RMSE", "R2"]]
        st.dataframe(rdf.style.format("{:.3f}").highlight_max(subset=["R2"], color="#2e7d32"),
                     width="stretch")
        st.success(f"Best regressor: **{reg.best_model}**")
    with c2:
        st.markdown("**Classification — predict HIGH engagement**")
        cdf = pd.DataFrame({n: {k: v for k, v in r.metrics.items() if k != "confusion_matrix"}
                            for n, r in clf.results.items()}).T
        st.dataframe(cdf.style.format("{:.3f}").highlight_max(subset=["f1"], color="#2e7d32"),
                     width="stretch")
        st.success(f"Best classifier: **{clf.best_model}**")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Confusion matrix — best classifier**")
        cm = clf.results[clf.best_model].metrics["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Pred LOW", "Pred HIGH"],
                        y=["True LOW", "True HIGH"], color_continuous_scale="Purples")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with c4:
        st.markdown("**Prediction drivers — permutation importance**")
        ddf = pd.DataFrame([{
            "feature": d.feature, "importance": d.importance,
            "direction": {"up": "↑", "down": "↓", "flat": "·"}[d.direction],
        } for d in bundle.drivers])
        fig2 = px.bar(ddf, x="importance", y="feature", orientation="h",
                      color_discrete_sequence=["#ff4d8d"])
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, width="stretch")
        st.caption("↑ moves with engagement, ↓ moves against it. Descriptive only.")


def page_methodology(bundle: RizzBundle) -> None:
    hero("Methodology — the part that is, annoyingly, real science.")
    st.markdown(f"""
### The research question
> Given only the **early portion** of a conversation, can measurable
> conversational signals predict the **engagement of the later portion**?

We do **not** study attraction, feelings, or intentions. We study *observable
conversational behavior* and predict *observable future conversational behavior*.

### Leakage prevention (non-negotiable)
Features are extracted from the **first N messages** of each session. The target
(the Engagement Index) is computed from the **entire** session. The model
therefore only ever sees the beginning and is asked to predict the rest. Sessions
that don't have more than N messages are dropped — there's no future to forecast.
This is enforced by `tests/test_leakage.py`, which fails loudly if the future
ever leaks into the features.

### The Conversational Engagement Index™
A configurable, behavior-only proxy combining normalized session duration,
message volume, back-and-forth exchange, participation balance, and persistence.

> {ENGAGEMENT_DISCLAIMER}

### Models
Baseline (mean / majority), Linear/Logistic Regression, Random Forest,
Gradient Boosting. No neural networks — the question is whether *simple,
interpretable* signals carry predictive information at all.

### Hard scientific boundaries
Fast replies are **not** attraction. Long conversations are **not** romance.
Initiation is **not** affection. A signal is not a meaning. The same message
can be sent for completely different reasons. That boundary is the entire point.
""")


def page_privacy(bundle: RizzBundle) -> None:
    hero("Privacy Center — your chats never leave this machine.")
    st.markdown("""
### Local by construction
```text
YOUR COMPUTER
  Raw chat → Parser → Feature extraction → Local dataset → ML → Dashboard
```
No external AI APIs. No cloud databases. No automatic uploads. No telemetry.
No third-party chat processing. Uploaded `.txt` files are parsed **in memory**
and never written to disk. The public repo ships **synthetic** demo data only,
and `.gitignore` blocks `*.txt`, `*.csv`, `*.json`, and the `data/` tree.
""")
    st.markdown("### Things Rizzmatics Cannot Tell You")
    for item in things_rizzmatics_cannot_tell_you():
        st.markdown(f"- ❌ {item}")
    st.error("**Please communicate with the human.**")
    with st.expander("The Human Override™"):
        st.code(human_override(), language=None)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
_RENDERERS = {
    PAGES[0]: page_executive, PAGES[1]: page_explorer, PAGES[2]: page_analytics,
    PAGES[3]: page_oracle, PAGES[4]: page_model_lab, PAGES[5]: page_methodology,
    PAGES[6]: page_privacy,
}


def main() -> None:
    inject_theme()
    text, params, page = sidebar()

    if not text:
        hero("Applied mathematics for completely unnecessary flirting.")
        st.code(boot_screen(), language=None)
        st.info("Load the demo data or upload a WhatsApp `.txt` in the sidebar to begin.")
        return

    bundle = run_pipeline(text, **params)
    _RENDERERS[page](bundle)

    if page == PAGES[0]:
        with st.expander("The Final Rizzmatics Moment™"):
            st.code(final_moment({
                "messages": bundle.n_messages, "sessions": bundle.n_sessions,
                "features": len(FEATURE_NAMES) + 3,
                "models": 4,
                "predictions": len(bundle.dataset) if bundle.dataset else 0,
            }), language=None)


if __name__ == "__main__":
    main()
