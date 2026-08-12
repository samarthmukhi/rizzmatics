"""Tests for the authentication gate and optional lore loader."""

import hashlib

import pytest

from app.components import gate

FIXTURE = "tests.fixtures.lore_fixture"


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def test_password_not_configured_by_default(monkeypatch):
    monkeypatch.delenv(gate.HASH_ENV, raising=False)
    assert gate.password_configured() is False
    assert gate.check_password("anything") is False


def test_check_password_matches_only_correct_key(monkeypatch):
    monkeypatch.setenv(gate.HASH_ENV, _hash("open sesame"))
    assert gate.check_password("open sesame") is True
    assert gate.check_password("wrong") is False
    assert gate.check_password("") is False


def test_hash_comparison_is_case_insensitive_on_stored_hex(monkeypatch):
    monkeypatch.setenv(gate.HASH_ENV, _hash("Key123").upper())  # stored uppercased
    assert gate.check_password("Key123") is True


def test_lore_available_and_load_with_fixture(monkeypatch):
    monkeypatch.setenv(gate.LORE_MODULE_ENV, FIXTURE)
    assert gate.lore_available() is True
    reg = gate.load_registry()
    assert reg is not None
    assert reg.home_id() == "home"


def test_lore_absent_returns_none(monkeypatch):
    monkeypatch.setenv(gate.LORE_MODULE_ENV, "nonexistent.module.definitely")
    assert gate.lore_available() is False
    assert gate.load_registry() is None


def test_session_auth_flags():
    session = {}
    assert gate.is_authenticated(session) is False
    gate.mark_authenticated(session)
    assert gate.is_authenticated(session) is True
    gate.mark_authenticated(session, False)
    assert gate.is_authenticated(session) is False
