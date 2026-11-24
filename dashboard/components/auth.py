"""Dashboard sign-in (single shared password, standard library only).

Configuration (environment variable or Streamlit secret of the same name):

    MARKETPULSE_DASHBOARD_PASSWORD_HASH = pbkdf2_sha256$<iterations>$<salt b64>$<hash b64>

Generate the value with:

    python -m dashboard.components.auth           # prompts for the password

Behaviour
---------
* Hash set: every page requires sign-in. Passwords are verified with
  PBKDF2-HMAC-SHA256 and a constant-time comparison; the password itself is
  never stored anywhere.
* After 5 failed attempts a browser session is locked for 60 seconds
  (per-session throttle; put an IP-level limit on the reverse proxy for
  internet-facing deployments).
* No hash + MARKETPULSE_ENV=production: the dashboard refuses to serve and
  explains how to configure it (never an unauthenticated "production" app).
* No hash + development: open, as before, for local use.

This is shared-password access control for a small team. For per-user
accounts, SSO or MFA, put the dashboard behind an authenticating proxy (see
deploy/README.md).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Optional

ENV_NAME = "MARKETPULSE_DASHBOARD_PASSWORD_HASH"
ITERATIONS = 600_000
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
_SESSION_OK = "mp_auth_ok"
_SESSION_FAILS = "mp_auth_failures"
_SESSION_LOCKED_UNTIL = "mp_auth_locked_until"


def hash_password(password: str, *, iterations: int = ITERATIONS, salt: Optional[bytes] = None) -> str:
    if not password:
        raise ValueError("Password must not be empty.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(iterations, base64.b64encode(salt).decode(),
                                           base64.b64encode(digest).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = stored.strip().split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(hash_b64)
        digest = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, int(iters))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


def configured_hash() -> Optional[str]:
    value = os.environ.get(ENV_NAME)
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(ENV_NAME)  # Streamlit Community Cloud secrets
        except Exception:
            value = None
    return value.strip() if isinstance(value, str) and value.strip() else None


def require_login() -> bool:
    """Gate the app. Returns True when the page may render."""
    import streamlit as st
    from config import settings

    stored = configured_hash()
    if not stored:
        if settings.is_production():
            st.error("Dashboard sign-in is not configured. In production MarketPulse AI does not serve data "
                     "without authentication.")
            st.markdown(f"Set `{ENV_NAME}` (environment variable or Streamlit secret). Generate it with "
                        "`python -m dashboard.components.auth`, or run the dashboard behind an authenticating "
                        "proxy — see `deploy/README.md`.")
            return False
        return True

    if st.session_state.get(_SESSION_OK):
        return True

    locked_until = float(st.session_state.get(_SESSION_LOCKED_UNTIL, 0))
    now = time.time()
    st.markdown('<div class="mp-page-header"><span class="mp-header-badge">Sign in</span>'
                '<div class="mp-page-title">MarketPulse AI</div>'
                '<div class="mp-page-subtitle">This workspace is private. Enter the team password to continue.'
                '</div></div>', unsafe_allow_html=True)
    if locked_until > now:
        st.error(f"Too many failed attempts. Try again in {int(locked_until - now) + 1} seconds.")
        return False
    with st.form("mp_login", clear_on_submit=True):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if verify_password(password, stored):
            st.session_state[_SESSION_OK] = True
            st.session_state[_SESSION_FAILS] = 0
            st.rerun()
        fails = int(st.session_state.get(_SESSION_FAILS, 0)) + 1
        st.session_state[_SESSION_FAILS] = fails
        if fails >= MAX_ATTEMPTS:
            st.session_state[_SESSION_LOCKED_UNTIL] = now + LOCKOUT_SECONDS
            st.session_state[_SESSION_FAILS] = 0
            st.error(f"Too many failed attempts. Sign-in is locked for {LOCKOUT_SECONDS} seconds.")
        else:
            st.error("Incorrect password.")
    return False


def render_logout() -> None:
    import streamlit as st
    if configured_hash() and st.session_state.get(_SESSION_OK):
        if st.button("Sign out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


if __name__ == "__main__":  # pragma: no cover - interactive helper
    import getpass
    pw = getpass.getpass("Dashboard password: ")
    if pw != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match.")
    if len(pw) < 12:
        raise SystemExit("Use at least 12 characters.")
    print(f"{ENV_NAME}={hash_password(pw)}")
