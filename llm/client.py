#client.py
import json
import os
import re
import requests
from typing import List, Optional


class LLMClientError(RuntimeError):
    pass


class InfineonGPTClient:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self.model = model or os.environ.get("INFINEON_MODEL", "gpt-4ifx")
        self.base_url = base_url or os.environ.get("INFINEON_API_URL")
        self.api_key = api_key or os.environ.get("INFINEON_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.base_url or not self.api_key:
            raise ValueError("Missing API URL or API key.")

    def generate(self, prompt: str, k: int = 5) -> List[str]:
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # 🔁 Try Bearer first, fallback to api-key
        headers_options = [
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            {
                "api-key": self.api_key,
                "Content-Type": "application/json",
            },
        ]

        last_error = None

        for headers in headers_options:
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    candidates = _parse_candidates(text)
                    return candidates[:k]
                else:
                    last_error = response.text

            except requests.RequestException as exc:
                last_error = str(exc)

        raise LLMClientError(f"API failed: {last_error}")

from openai import OpenAI


class OpenAIClient:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompt: str, k: int = 5) -> List[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        text = response.choices[0].message.content
        return _parse_candidates(text)[:k]
    

def _parse_candidates(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    # Try JSON list
    try:
        return list(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array from text
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        try:
            return list(json.loads(match.group(0)))
        except json.JSONDecodeError:
            pass

    # Fallback: line-based parsing
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines