from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

import streamlit as st


AUTHENTICATED_KEY = "auth_authenticated"
AUTH_USER_KEY = "auth_user"


def init_auth_state() -> None:
    if AUTHENTICATED_KEY not in st.session_state:
        st.session_state[AUTHENTICATED_KEY] = False
    if AUTH_USER_KEY not in st.session_state:
        st.session_state[AUTH_USER_KEY] = None


def load_auth_settings() -> dict[str, Any]:
    settings: dict[str, Any] = {}

    try:
        if "auth" in st.secrets:
            secret_section = st.secrets["auth"]
            settings.update(dict(secret_section))
    except Exception:  # noqa: BLE001
        pass

    settings["username"] = settings.get("username") or os.getenv("STREAMLIT_AUTH_USERNAME")
    settings["password"] = settings.get("password") or os.getenv("STREAMLIT_AUTH_PASSWORD")
    settings["password_hash_sha256"] = settings.get("password_hash_sha256") or os.getenv("STREAMLIT_AUTH_PASSWORD_HASH")
    settings["enabled"] = bool(settings.get("username") and (settings.get("password") or settings.get("password_hash_sha256")))
    return settings


def is_authenticated() -> bool:
    return bool(st.session_state.get(AUTHENTICATED_KEY, False))


def authenticated_user() -> str | None:
    return st.session_state.get(AUTH_USER_KEY)


def logout() -> None:
    st.session_state[AUTHENTICATED_KEY] = False
    st.session_state[AUTH_USER_KEY] = None


def _verify_password(entered_password: str, settings: dict[str, Any]) -> bool:
    password_hash = settings.get("password_hash_sha256")
    if password_hash:
        entered_hash = hashlib.sha256(entered_password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(entered_hash, str(password_hash))

    password = settings.get("password")
    if password is None:
        return False
    return hmac.compare_digest(entered_password, str(password))


def attempt_login(username: str, password: str, settings: dict[str, Any]) -> bool:
    expected_username = settings.get("username")
    if not expected_username:
        return False

    username_ok = hmac.compare_digest(username.strip(), str(expected_username))
    password_ok = _verify_password(password, settings)
    if username_ok and password_ok:
        st.session_state[AUTHENTICATED_KEY] = True
        st.session_state[AUTH_USER_KEY] = username.strip()
        return True
    return False


def render_login_screen(settings: dict[str, Any]) -> None:
    st.title("Financial Direction Dashboard")
    st.caption("Bu arayuz korunmaktadir. Devam etmek icin giris yapin.")

    if not settings.get("enabled"):
        st.error("Login yapilandirmasi bulunamadi. `.streamlit/secrets.toml` veya environment variable ile auth ayari yapin.")
        st.code(
            "[auth]\nusername = \"admin\"\npassword = \"change-me\"",
            language="toml",
        )
        return

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Kullanici adi")
        password = st.text_input("Sifre", type="password")
        submitted = st.form_submit_button("Giris Yap")

    if submitted:
        if attempt_login(username, password, settings):
            st.success("Giris basarili. Dashboard yukleniyor...")
            st.rerun()
        else:
            st.error("Kullanici adi veya sifre hatali.")
