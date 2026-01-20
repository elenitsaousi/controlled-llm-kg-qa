from typing import Dict, List, Optional

from kg.schema import KGSchema
from llm.ollama_client import OllamaClient, OllamaClientError
from llm.prompts import build_candidate_prompt


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
    client = llm_client or OllamaClient()
    try:
        generated = client.generate(prompt, k=k)
        candidates = [{"query": text, "source": "ollama"} for text in generated]
        return {"prompt": prompt, "candidates": candidates, "metadata": {"k": k}}
    except OllamaClientError as exc:
        return {
            "prompt": prompt,
            "candidates": [],
            "metadata": {"k": k, "error": str(exc)},
        }
