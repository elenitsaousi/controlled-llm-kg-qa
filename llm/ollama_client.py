import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional


class OllamaClientError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        model: Optional[str] = None,
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, k: int = 5) -> List[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise OllamaClientError(str(exc)) from exc

        try:
            response = json.loads(body)
            text = response.get("response", "")
        except json.JSONDecodeError as exc:
            raise OllamaClientError("Invalid JSON response from Ollama.") from exc

        candidates = _parse_candidates(text)
        return candidates[:k]


def _parse_candidates(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    try:
        return list(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        try:
            return list(json.loads(match.group(0)))
        except json.JSONDecodeError:
            pass

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines
