"""Visual identity for the public dashboard — premium minimal.

Same black / neon-pink / white language as the After-Hours build, but the
*public* register: clean, product-grade, no signature layer (no sigil, no
starfield, no serif finale — those belong to her build only). Flat black,
hairline borders, one pink accent used sparingly, refined type.
"""

from __future__ import annotations

import streamlit as st

_FONTS = ("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700"
          "&family=JetBrains+Mono:wght@400;500;600&display=swap")

_CSS = f"""
<style>
@import url('{_FONTS}');
:root {{
  --pink:#ff2e88; --ink:#eceef2; --mut:rgba(236,238,242,0.5);
  --line:rgba(255,255,255,0.09); --panel:#070708;
}}
.stApp {{ background:#000; }}
.stApp, .stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"] {{
  font-family:'Space Grotesk', system-ui, sans-serif; color:var(--ink);
}}
h1,h2,h3,h4 {{ font-family:'Space Grotesk',sans-serif !important; letter-spacing:.01em; color:#fff; }}
code, pre, [data-testid="stMetricValue"], .stDataFrame {{ font-family:'JetBrains Mono',monospace !important; }}
::selection {{ background:var(--pink); color:#000; }}

/* hero */
.rizz-hero {{ border-bottom:1px solid var(--line); padding:8px 2px 22px; margin-bottom:22px; }}
.rizz-hero h1 {{
  font-size:2rem; margin:0; letter-spacing:.22em; font-weight:700; color:#fff; }}
.rizz-hero p {{ margin:12px 0 0; color:var(--mut); font-size:.92rem; letter-spacing:.02em; }}
.rizz-badge {{
  display:inline-block; padding:3px 11px; border:1px solid var(--line); border-radius:999px;
  font-family:'JetBrains Mono',monospace; font-size:.6rem; letter-spacing:.22em;
  text-transform:uppercase; margin-right:8px; color:var(--pink); }}
.rizz-disclaimer {{
  border:1px solid var(--line); border-left:2px solid var(--pink); background:transparent;
  padding:11px 14px; border-radius:6px; font-size:.82rem; color:var(--mut); margin:6px 0; }}

/* sidebar */
[data-testid="stSidebar"] {{ background:#050505; border-right:1px solid var(--line); }}
[data-testid="stSidebar"] * {{ font-family:'Space Grotesk',sans-serif; }}

/* metrics as clean stat tiles */
[data-testid="stMetric"] {{
  background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }}
[data-testid="stMetricLabel"] {{ color:var(--mut); text-transform:uppercase; letter-spacing:.12em; font-size:.62rem; }}
[data-testid="stMetricValue"] {{ color:#fff; font-variant-numeric:tabular-nums; }}

/* controls -> pink accent */
[data-baseweb="slider"] div[role="slider"] {{ background:var(--pink) !important; }}
.stSlider [data-baseweb="slider"] > div > div {{ background:var(--pink) !important; }}
.stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"] button {{
  background:#000; color:var(--pink); border:1px solid var(--pink); border-radius:9px;
  font-family:'Space Grotesk',sans-serif; font-weight:600; letter-spacing:.06em;
  transition:background .15s,color .15s; }}
.stButton>button:hover {{ background:var(--pink); color:#000; }}
[data-baseweb="tab-list"] {{ border-bottom:1px solid var(--line); }}
[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {{ letter-spacing:.04em; }}
[aria-selected="true"][data-baseweb="tab"] {{ color:var(--pink) !important; }}

/* inputs */
input, textarea, [data-baseweb="select"] > div {{
  background:#000 !important; border-color:var(--line) !important; color:#fff !important; }}
input:focus {{ border-color:var(--pink) !important; box-shadow:0 0 0 1px var(--pink) !important; }}

/* tables / expanders */
.stDataFrame, [data-testid="stExpander"] {{ border:1px solid var(--line); border-radius:10px; }}
[data-testid="stExpander"] summary {{ color:var(--ink); }}
hr {{ border-color:var(--line); }}

/* alerts (info/success/warning/error) -> uniform minimal, pink edge */
[data-testid="stAlert"], [data-testid="stNotification"] {{
  background:var(--panel) !important; border:1px solid var(--line) !important;
  border-left:2px solid var(--pink) !important; border-radius:8px !important;
  box-shadow:none !important; }}
[data-testid="stAlert"] *, [data-testid="stNotification"] * {{ color:var(--ink) !important; }}
</style>
"""


def inject_theme() -> None:
    """Inject the minimal public theme once per page render."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(subtitle: str) -> None:
    """Render the minimal page header (no gradients)."""
    st.markdown(
        f"""
        <div class="rizz-hero">
          <span class="rizz-badge">Enterprise-Grade</span>
          <span class="rizz-badge">Actual Necessity 0%</span>
          <h1>RIZZMATICS</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer(text: str) -> None:
    """Render a small disclaimer block."""
    st.markdown(f'<div class="rizz-disclaimer">{text}</div>', unsafe_allow_html=True)
