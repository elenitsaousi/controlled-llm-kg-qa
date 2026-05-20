#!/usr/bin/env python3
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm.client import InfineonGPTClient, LLMAuthError, LLMClientError


def _mask_secret(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "<present>"
    return f"{value[:4]}...{value[-4:]} ({len(value)} chars)"


def main() -> int:
    print("===== INFINEON LLM AUTH CHECK =====")
    print(f"INFINEON_API_URL: {os.environ.get('INFINEON_API_URL') or '<missing>'}")
    print(f"INFINEON_CHAT_ENDPOINT: {os.environ.get('INFINEON_CHAT_ENDPOINT') or '/chat/completions'}")
    print(f"INFINEON_MODEL: {os.environ.get('INFINEON_MODEL') or 'gpt-4o'}")
    print(f"INFINEON_API_KEY: {_mask_secret(os.environ.get('INFINEON_API_KEY', ''))}")

    try:
        client = InfineonGPTClient()
    except Exception as exc:
        print(f"FAIL: could not build client: {exc}")
        return 2

    try:
        client.check_auth()
    except LLMAuthError as exc:
        print("FAIL: endpoint redirected to SSO/gateway.")
        print(str(exc))
        print(
            "Action: refresh/replace INFINEON_API_KEY for the local terminal process. "
            "A browser login alone may not update this token."
        )
        return 3
    except LLMClientError as exc:
        print("FAIL: LLM API call failed.")
        print(str(exc))
        return 4

    print("OK: LLM endpoint accepted the configured credentials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
