# candidate_generation.py
from typing import Dict, List, Optional
from kg.schema import KGSchema
from llm.client import OpenAIClient
from llm.prompts import build_candidate_prompt
import re


def generate_candidate_prompt(
    question: str, schema: KGSchema, k: int = 5
) -> str:
    return build_candidate_prompt(question, schema, k)


def generate_candidates(
    question: str,
    schema: KGSchema,
    k: int = 5,
    llm_client: Optional[object] = None,
) -> Dict:

    # Build prompt
    prompt = build_candidate_prompt(question, schema, k)

    # Use provided client or default
    import os
    from llm.client import OpenAIClient, InfineonClient

    def get_default_client():
        provider = os.getenv("LLM_PROVIDER", "openai")

        if provider == "infineon":
            return InfineonClient()
        elif provider == "openai":
            return OpenAIClient()
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
        
    client = llm_client or get_default_client()

    print("\n=== PROMPT ===")
    print(prompt)

    try:
        # Call LLM
        generated = client.generate(prompt, k=k)

        print("\n=== RAW LLM OUTPUT ===")
        print(generated)

        # -----------------------------
        # Normalize LLM output
        # -----------------------------
        if generated is None:
            generated = []

        # Case 1: string output
        elif isinstance(generated, str):
            # Try numbered split first (1. ... 2. ...)
            split_candidates = re.split(r"\n\d+\.\s*", generated)

            # fallback: simple newline split
            if len(split_candidates) <= 1:
                split_candidates = generated.split("\n")

            generated = [g.strip() for g in split_candidates if g.strip()]

        # Case 2: already list
        elif isinstance(generated, list):
            generated = [str(g).strip() for g in generated if str(g).strip()]

        else:
            raise ValueError(f"Unexpected LLM output type: {type(generated)}")

        print("\n=== PARSED CANDIDATES ===")
        print(generated)

        # Build structured candidates
        candidates = [
            {"query": text, "source": "openai"}
            for text in generated
        ]

        if not candidates:
            print("⚠️ WARNING: No candidates generated!")

        return {
            "prompt": prompt,
            "candidates": candidates,
            "metadata": {
                "k_requested": k,
                "k_returned": len(candidates),
            },
        }

    except Exception as exc:
        print("\n❌ LLM GENERATION FAILED:")
        print(exc)
        raise