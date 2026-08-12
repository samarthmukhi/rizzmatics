"""Authentication gate and optional lore loader.

Server-side only. This module decides whether a private lore package exists and
whether the current session has authenticated. It contains **no lore content and
no secret** — only a comparison against a hash held in an environment variable.

Security model (honest): Streamlit executes the whole script server-side and
sends only the output of code paths that actually run. The private render path
does not execute until :func:`is_authenticated` is true, so an unauthenticated
browser never receives lore bytes. This is genuine gating plus content-absence
in the public build — not a hardened cryptographic system. Appropriate for a
personal, manually-shared artifact.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import os

__all__ = [
    "LORE_MODULE_ENV",
    "HASH_ENV",
    "default_lore_module",
    "lore_available",
    "load_registry",
    "password_configured",
    "check_password",
    "is_authenticated",
    "mark_authenticated",
]

LORE_MODULE_ENV = "RIZZMATICS_LORE_MODULE"   # overridable for tests/fixtures
HASH_ENV = "RIZZMATICS_PRIVATE_HASH"          # sha256 hex of the passphrase
_DEFAULT_LORE_MODULE = "private.lore"
_SESSION_KEY = "_after_hours_authed"


def default_lore_module() -> str:
    """The module imported to obtain the lore registry (env-overridable)."""
    return os.environ.get(LORE_MODULE_ENV, _DEFAULT_LORE_MODULE)


def lore_available() -> bool:
    """True if a private lore package is importable in this build."""
    module = default_lore_module()
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def load_registry():
    """Import the lore package and build its registry, or return ``None``.

    Returns ``None`` in the public build (no private package present), which is
    what keeps the After-Hours route inert for strangers.
    """
    module = default_lore_module()
    try:
        mod = importlib.import_module(module)
        importlib.reload(mod)  # pick up fixture swaps within a test session
        return mod.build_registry()
    except Exception:
        return None


def password_configured() -> bool:
    """True if a private-build password hash is set in the environment."""
    return bool(os.environ.get(HASH_ENV))


def check_password(raw: str) -> bool:
    """Constant-time compare of sha256(raw) against the configured hash."""
    expected = os.environ.get(HASH_ENV, "")
    if not expected or raw is None:
        return False
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, expected.strip().lower())


def is_authenticated(session_state) -> bool:
    """Whether this session has cleared the gate."""
    return bool(session_state.get(_SESSION_KEY, False))


def mark_authenticated(session_state, value: bool = True) -> None:
    session_state[_SESSION_KEY] = value
