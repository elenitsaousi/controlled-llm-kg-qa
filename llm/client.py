# client.py
import json
import os
import re
import time
import requests
from typing import List, Optional
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMClientError(RuntimeError):
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
        self.model = model or os.environ.get("INFINEON_MODEL", "gpt-4o")
        self.base_url = base_url or os.environ.get("INFINEON_API_URL")
        self.api_key = api_key or os.environ.get("INFINEON_API_KEY")
        self.chat_endpoint = os.environ.get("INFINEON_CHAT_ENDPOINT", "/chat/completions")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = int(os.environ.get("INFINEON_MAX_RETRIES", "4"))
        self.retry_backoff_sec = float(os.environ.get("INFINEON_RETRY_BACKOFF_SEC", "1.0"))
        if not self.base_url or not self.api_key:
            raise ValueError("Missing API URL or API key.")

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
        last_error = "unknown error"

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                    verify=False,
                    allow_redirects=False,
                )
                if response.status_code == 200:
                    data = response.json()
                    return str(data["choices"][0]["message"]["content"])
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location", "")
                    body = (response.text or "")[:500]
                    raise LLMClientError(
                        f"API redirect to SSO/gateway (status={response.status_code}, "
                        f"location={location!r}). Body preview: {body!r}"
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
