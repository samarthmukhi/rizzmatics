"""THE security test: unauthenticated users cannot receive private lore.

Two complementary checks:
  * AppTest: an unauthenticated visit to ?after_hours=1 renders only the door
    and clue — no lore in the output.
  * Direct gating test: render_after_hours only ever hands the rich component
    (which carries the node data) to Streamlit when the session is authenticated.
    This is independent of Streamlit's component introspection, so it stays
    meaningful even as the UI evolves.
"""

import hashlib
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest

from app.components import gate, private_view
from tests.fixtures.lore_fixture import CLUE, MARKER, REVEAL_MARKER, build_registry

FIXTURE = "tests.fixtures.lore_fixture"
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
KEY = "let me in"
KEY_HASH = hashlib.sha256(KEY.encode()).hexdigest()


def _rendered_text(at) -> str:
    parts = []
    for kind in ("markdown", "code", "info", "warning", "error", "success",
                 "text", "title", "header", "subheader", "caption"):
        for el in getattr(at, kind, []):
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


def _app(monkeypatch):
    monkeypatch.setenv("RIZZMATICS_LORE_MODULE", FIXTURE)
    monkeypatch.setenv("RIZZMATICS_PRIVATE_HASH", KEY_HASH)
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["after_hours"] = "1"
    return at


# --------------------------------------------------------------------------- #
# A tiny fake `st` so we can test the gating without Streamlit internals.
# --------------------------------------------------------------------------- #
class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeSt:
    def __init__(self, authed: bool):
        self.session_state = {"_after_hours_authed": authed}
        self.captured: list[str] = []

    def markdown(self, s, **k):
        self.captured.append(str(s))

    error = info = warning = success = markdown

    def columns(self, spec):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_Ctx() for _ in range(n)]

    def form(self, *a, **k):
        return _Ctx()

    def text_input(self, *a, **k):
        return ""

    def form_submit_button(self, *a, **k):
        return False

    def rerun(self):
        pass


# --------------------------------------------------------------------------- #
# THE core guarantee (rendering-independent).
# --------------------------------------------------------------------------- #
def test_component_is_gated_behind_authentication(monkeypatch):
    monkeypatch.setenv("RIZZMATICS_LORE_MODULE", FIXTURE)
    monkeypatch.setenv("RIZZMATICS_PRIVATE_HASH", KEY_HASH)
    calls: list[str] = []
    monkeypatch.setattr(private_view, "components_html",
                        lambda html, **k: calls.append(html))

    # Unauthenticated: the lore component is NEVER produced, and no marker leaks.
    st_un = _FakeSt(authed=False)
    private_view.render_after_hours(st_un)
    assert calls == [], "lore component rendered for an unauthenticated session!"
    assert all(MARKER not in c for c in st_un.captured)
    assert all(REVEAL_MARKER not in c for c in st_un.captured)

    # Authenticated: the component is produced, and it carries the lore.
    st_ok = _FakeSt(authed=True)
    private_view.render_after_hours(st_ok)
    assert len(calls) == 1
    assert MARKER in calls[0] and REVEAL_MARKER in calls[0]


def test_authenticated_html_contains_lore():
    html = private_view.authenticated_component_html(build_registry())
    assert MARKER in html
    assert REVEAL_MARKER in html
    assert "RIZZMATICS" in html


# --------------------------------------------------------------------------- #
# AppTest: unauthenticated visit shows only the door + clue.
# --------------------------------------------------------------------------- #
def test_unauthenticated_visit_shows_door_not_lore(monkeypatch):
    at = _app(monkeypatch).run()
    text = _rendered_text(at)
    assert "unmodeled" in text.lower()     # the door shell (RIZZMATICS // The Unmodeled)
    assert CLUE in text                    # the cryptic hint (by design)
    assert MARKER not in text              # ...but no lore
    assert REVEAL_MARKER not in text


def test_public_build_route_is_inert(monkeypatch):
    monkeypatch.setenv("RIZZMATICS_LORE_MODULE", "nonexistent.module.at.all")
    monkeypatch.delenv("RIZZMATICS_PRIVATE_HASH", raising=False)
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["after_hours"] = "1"
    at.run()
    text = _rendered_text(at)
    assert MARKER not in text
    assert "does not include the private module" in text
