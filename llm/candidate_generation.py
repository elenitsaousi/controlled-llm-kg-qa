# candidate_generation.py
import os
from typing import Dict, List, Optional
from kg.schema import KGSchema
from llm.prompts import build_candidate_prompt


def _get_default_client():
    """Reads LLM_BACKEND from .env and returns the appropriate client."""
    from llm.client import OpenAIClient, InfineonGPTClient

    backend = os.environ.get("LLM_BACKEND", "auto").lower()

    if backend == "openai":
        return OpenAIClient()

    if backend == "infineon":
        return InfineonGPTClient()

    # auto: prefers OpenAI if key exists
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIClient()

    if os.environ.get("INFINEON_API_URL") and os.environ.get("INFINEON_API_KEY"):
        return InfineonGPTClient()

    raise RuntimeError(
        "No LLM client available. Set LLM_BACKEND=openai or LLM_BACKEND=infineon in .env"
    )


def generate_candidate_prompt(
    question: str, schema: KGSchema, k: int = 5
) -> str:
    return build_candidate_prompt(question, schema, k)


def generate_candidates(
    question: str,
    schema: KGSchema,
    k: int = 5,
    llm_client: Optional[object] = None,
) -> Dict[str, List[str]]:
    prompt = build_candidate_prompt(question, schema, k)

    client = llm_client or _get_default_client()

    # Determine backend name for the source field
    client_type = type(client).__name__
    if "Infineon" in client_type:
        backend_name = "infineon"
    elif "OpenAI" in client_type:
        backend_name = "openai"
    else:
        backend_name = client_type.lower()

    try:
        generated = client.generate(prompt, k=k)
        candidates = [{"query": text, "source": backend_name} for text in generated]
        return {
            "prompt": prompt,
            "candidates": candidates,
            "metadata": {"k": k, "backend": backend_name},
        }
    except Exception as exc:
        return {
            "prompt": prompt,
            "candidates": [],
            "metadata": {"k": k, "error": str(exc), "backend": backend_name},
        }