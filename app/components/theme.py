"""Visual identity for the dashboard: cocky, enterprise, faintly ridiculous."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
  --rizz-accent: #ff4d8d;
  --rizz-accent2: #7b61ff;
  --rizz-ink: #e9e9f2;
}
.stApp {
  background: radial-gradient(1200px 600px at 15% -10%, rgba(123,97,255,0.16), transparent),
              radial-gradient(1000px 500px at 110% 10%, rgba(255,77,141,0.14), transparent);
}
.rizz-hero {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 26px 30px;
  background: linear-gradient(135deg, rgba(123,97,255,0.20), rgba(255,77,141,0.12));
  margin-bottom: 8px;
}
.rizz-hero h1 {
  font-size: 2.5rem; margin: 0; letter-spacing: 0.14em; font-weight: 800;
  background: linear-gradient(90deg, var(--rizz-accent), var(--rizz-accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.rizz-hero p { margin: 6px 0 0; opacity: 0.85; font-size: 1.02rem; }
.rizz-badge {
  display:inline-block; padding: 3px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18); font-size: 0.72rem;
  letter-spacing: 0.12em; text-transform: uppercase; margin-right: 6px; opacity:0.8;
}
.rizz-disclaimer {
  border-left: 3px solid var(--rizz-accent);
  background: rgba(255,77,141,0.06);
  padding: 10px 14px; border-radius: 6px; font-size: 0.86rem; opacity: 0.9;
}
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
</style>
"""


def inject_theme() -> None:
    """Inject the Rizzmatics CSS once per page render."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(subtitle: str) -> None:
    """Render the gradient page hero."""
    st.markdown(
        f"""
        <div class="rizz-hero">
          <span class="rizz-badge">Enterprise-Grade</span>
          <span class="rizz-badge">Actual Necessity: 0%</span>
          <h1>RIZZMATICS</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer(text: str) -> None:
    """Render a small disclaimer block."""
    st.markdown(f'<div class="rizz-disclaimer">⚠ {text}</div>', unsafe_allow_html=True)
