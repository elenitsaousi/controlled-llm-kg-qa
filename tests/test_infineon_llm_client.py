import os

import pytest

from llm.client import InfineonGPTClient


class _Response:
    def __init__(self, status_code, text="", payload=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


def test_client_refreshes_token_after_sso_redirect(monkeypatch):
    monkeypatch.setenv("INFINEON_API_URL", "https://example.invalid")
    monkeypatch.setenv("INFINEON_API_KEY", "stale-token")
    monkeypatch.setenv("USER_LLM", "user")
    monkeypatch.setenv("PASSWORD_LLM", "password")

    posts = []

    def fake_post(url, headers, json, timeout, verify, allow_redirects):
        posts.append(headers["Authorization"])
        if len(posts) == 1:
            return _Response(
                302,
                headers={"location": "https://sso.infineon.com/as/authorization.oauth2"},
            )
        return _Response(
            200,
            payload={"choices": [{"message": {"content": "ok"}}]},
        )

    def fake_get(url, headers, auth, timeout, verify, allow_redirects):
        assert url == "https://example.invalid/auth/token"
        assert auth == ("user", "password")
        return _Response(200, text="fresh-token")

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    client = InfineonGPTClient()

    assert client.generate_text("hello") == "ok"
    assert posts == ["Bearer stale-token", "Bearer fresh-token"]
    assert os.environ["INFINEON_API_KEY"] == "fresh-token"


@pytest.mark.parametrize("auth_status", [401, 403])
def test_client_refreshes_token_after_auth_failure(monkeypatch, auth_status):
    monkeypatch.setenv("INFINEON_API_URL", "https://example.invalid")
    monkeypatch.setenv("INFINEON_API_KEY", "stale-token")
    monkeypatch.setenv("USER_LLM", "user")
    monkeypatch.setenv("PASSWORD_LLM", "password")

    posts = []

    def fake_post(url, headers, json, timeout, verify, allow_redirects):
        posts.append(headers["Authorization"])
        if len(posts) == 1:
            return _Response(auth_status, text="expired token")
        return _Response(
            200,
            payload={"choices": [{"message": {"content": "ok"}}]},
        )

    def fake_get(url, headers, auth, timeout, verify, allow_redirects):
        assert auth == ("user", "password")
        return _Response(200, text="fresh-token")

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    client = InfineonGPTClient()

    assert client.generate_text("hello") == "ok"
    assert posts == ["Bearer stale-token", "Bearer fresh-token"]


def test_client_can_start_without_static_token_when_credentials_exist(monkeypatch):
    monkeypatch.setenv("INFINEON_API_URL", "https://example.invalid")
    monkeypatch.delenv("INFINEON_API_KEY", raising=False)
    monkeypatch.setenv("USER_LLM", "user")
    monkeypatch.setenv("PASSWORD_LLM", "password")

    def fake_get(url, headers, auth, timeout, verify, allow_redirects):
        return _Response(200, text="fresh-token")

    monkeypatch.setattr("requests.get", fake_get)

    client = InfineonGPTClient()

    assert client.api_key == "fresh-token"


def test_client_requires_token_or_refresh_credentials(monkeypatch):
    monkeypatch.setenv("INFINEON_API_URL", "https://example.invalid")
    monkeypatch.delenv("INFINEON_API_KEY", raising=False)
    monkeypatch.delenv("USER_LLM", raising=False)
    monkeypatch.delenv("PASSWORD_LLM", raising=False)

    with pytest.raises(ValueError, match="Missing API key"):
        InfineonGPTClient()
