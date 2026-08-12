# client.py
import json
import os
import re
import time
import requests
from typing import List, Optional, Union
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None
else:
    load_dotenv()


class LLMClientError(RuntimeError):
    pass


class LLMAuthError(LLMClientError):
    pass


class InfineonGPTClient:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        self.backend = _env_first("LLM_BACKEND", "KGQA_LLM_BACKEND", default="infineon").strip().lower()
        litellm_mode = self.backend in {"litellm", "lite_llm"}
        if litellm_mode:
            self.model = model or _env_first("LITELLM_MODEL", "LITE_LLM_MODEL", "INFINEON_MODEL", default="gpt-4o")
            self.base_url = base_url or _env_first("LITELLM_BASE_URL", "LITE_LLM_BASE_URL", "BASE_URL", "INFINEON_API_URL")
            self.api_key = api_key or _env_first("LITELLM_API_KEY", "LITE_LLM_TOKEN", "LITE_LLM_API_KEY", "INFINEON_API_KEY")
            self.chat_endpoint = (
                _env_first("LITELLM_CHAT_ENDPOINT", "LITE_LLM_CHAT_ENDPOINT", "INFINEON_CHAT_ENDPOINT")
                or "/chat/completions"
            )
        else:
            self.model = model or os.environ.get("INFINEON_MODEL", "gpt-4o")
            self.base_url = base_url or os.environ.get("INFINEON_API_URL")
            self.api_key = api_key or os.environ.get("INFINEON_API_KEY")
            self.chat_endpoint = os.environ.get("INFINEON_CHAT_ENDPOINT", "/chat/completions")
        self.auth_endpoint = os.environ.get("INFINEON_AUTH_ENDPOINT", "/auth/token")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = int(os.environ.get("INFINEON_MAX_RETRIES", "4"))
        self.retry_backoff_sec = float(os.environ.get("INFINEON_RETRY_BACKOFF_SEC", "1.0"))
        self.request_timeout_sec = float(os.environ.get("INFINEON_REQUEST_TIMEOUT_SEC", "120"))
        self.auth_timeout_sec = float(os.environ.get("INFINEON_AUTH_TIMEOUT_SEC", "60"))
        self.auto_refresh_token = (not litellm_mode) and _env_bool("INFINEON_AUTO_REFRESH_TOKEN", True)
        self.api_user = os.environ.get("INFINEON_API_USER") or os.environ.get("USER_LLM")
        self.api_password = os.environ.get("INFINEON_API_PASSWORD") or os.environ.get("PASSWORD_LLM")
        self.verify = _requests_verify_setting()
        if not self.base_url:
            raise ValueError("Missing API URL.")
        if not self.api_key and self.auto_refresh_token and self.api_user and self.api_password:
            self.refresh_api_key()
        if not self.api_key:
            raise ValueError(
                "Missing API key. Set LITELLM_API_KEY/LITE_LLM_TOKEN/INFINEON_API_KEY or set "
                "USER_LLM/PASSWORD_LLM for legacy automatic token retrieval."
            )

    def generate(self, prompt: str, k: int = 5) -> List[str]:
        text = self.generate_text(prompt)
        candidates = _parse_candidates(text)
        return candidates[:k]

    def generate_text(self, prompt: str) -> str:
        endpoint = self.chat_endpoint if self.chat_endpoint.startswith("/") else f"/{self.chat_endpoint}"
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        retryable_statuses = {429, 500, 502, 503, 504}
        auth_failure_statuses = {401, 403}
        redirect_statuses = {301, 302, 303, 307, 308}
        last_error = "unknown error"
        token_refresh_attempted = False

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout_sec,
                    verify=self.verify,
                    allow_redirects=False,
                )
                if response.status_code == 200:
                    data = response.json()
                    return str(data["choices"][0]["message"]["content"])
                if response.status_code in auth_failure_statuses | redirect_statuses:
                    if (
                        not token_refresh_attempted
                        and self.auto_refresh_token
                        and self.api_user
                        and self.api_password
                    ):
                        token_refresh_attempted = True
                        self.refresh_api_key()
                        headers["Authorization"] = f"Bearer {self.api_key}"
                        response = requests.post(
                            url,
                            headers=headers,
                            json=payload,
                            timeout=self.request_timeout_sec,
                            verify=self.verify,
                            allow_redirects=False,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            return str(data["choices"][0]["message"]["content"])
                    location = response.headers.get("location", "")
                    body = (response.text or "")[:500]
                    raise LLMAuthError(
                        f"API authentication failed or redirected to SSO/gateway "
                        f"(status={response.status_code}, "
                        f"location={location!r}). Body preview: {body!r}. "
                        "This usually means INFINEON_API_KEY is missing, expired, or not valid "
                        "for machine API calls to INFINEON_API_URL + INFINEON_CHAT_ENDPOINT."
                    )

                body = (response.text or "")[:500]
                last_error = (
                    f"status={response.status_code} body={body!r}"
                )
                if response.status_code not in retryable_statuses or attempt >= self.max_retries:
                    break
            except requests.RequestException as exc:
                last_error = str(exc)
                if attempt >= self.max_retries:
                    break

            sleep_s = self.retry_backoff_sec * (2 ** attempt)
            time.sleep(sleep_s)

        raise LLMClientError(f"API failed: {last_error}")

    def refresh_api_key(self) -> str:
        if not self.api_user or not self.api_password:
            raise LLMAuthError(
                "Cannot refresh Infineon API token: missing USER_LLM/PASSWORD_LLM "
                "or INFINEON_API_USER/INFINEON_API_PASSWORD."
            )
        endpoint = self.auth_endpoint if self.auth_endpoint.startswith("/") else f"/{self.auth_endpoint}"
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        try:
            response = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                auth=(self.api_user, self.api_password),
                timeout=self.auth_timeout_sec,
                verify=self.verify,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise LLMAuthError(f"Failed to refresh Infineon API token: {exc}") from exc

        if 200 <= response.status_code < 300:
            token = response.text.strip()
            if not token:
                raise LLMAuthError("Infineon token endpoint returned an empty token.")
            self.api_key = token
            os.environ["INFINEON_API_KEY"] = token
            return token

        location = response.headers.get("location", "")
        body = (response.text or "")[:500]
        raise LLMAuthError(
            f"Failed to refresh Infineon API token: status={response.status_code}, "
            f"location={location!r}, body={body!r}"
        )

    def check_auth(self) -> None:
        previous_max_tokens = self.max_tokens
        previous_temperature = self.temperature
        try:
            self.max_tokens = 8
            self.temperature = 0.0
            self.generate_text("Return exactly: OK")
        finally:
            self.max_tokens = previous_max_tokens
            self.temperature = previous_temperature


class OpenAIClient:
    """
    Backward-compatibility shim.
    OpenAI backend is intentionally disabled in this repository branch.
    """
    def __init__(self, *args, **kwargs):
        raise LLMClientError(
            "OpenAI backend is disabled. Use InfineonGPTClient / LLM_BACKEND=infineon."
        )


def _clean_query(query: str) -> str:
    import re
    query = query.strip()
    while query and query[0] in '"\'':
        query = query[1:]
    while query and query[-1] in '"\'':
        query = query[:-1]
    query = query.strip()
    query = re.sub(r'\bSELECTT\b', 'SELECT', query)
    return query


def _parse_candidates(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    # Try JSON list
    try:
        parsed = list(json.loads(cleaned))
        return [_clean_query(str(q)) for q in parsed if q]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from text
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        try:
            parsed = list(json.loads(match.group(0)))
            return [_clean_query(str(q)) for q in parsed if q]
        except json.JSONDecodeError:
            pass

    # Fallback: line-based parsing
    lines = [_clean_query(line) for line in cleaned.splitlines() if line.strip()]
    return [l for l in lines if l]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


def _requests_verify_setting() -> Union[bool, str]:
    raw = (os.environ.get("INFINEON_CERT_PATH") or "").strip()
    if not raw:
        return False
    if raw.lower() in {"0", "false", "no", "off"}:
        return False
    return raw
