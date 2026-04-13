# candidate_generation.py
from typing import Dict, List, Optional
from kg.schema import KGSchema
from llm.client import OpenAIClient


from llm.prompts import build_candidate_prompt


def generate_candidate_prompt(
    question: str, schema: KGSchema, k: int = 5
) -> str:
    from llm.prompts import build_candidate_prompt
    return build_candidate_prompt(question, schema, k)


def generate_candidates(
    question: str,
    schema: KGSchema,
    k: int = 5,
    llm_client: Optional[object] = None,
) -> Dict[str, List[str]]:

    from llm.prompts import build_candidate_prompt

    prompt = build_candidate_prompt(question, schema, k)

    client = llm_client or OpenAIClient()

    try:
        generated = client.generate(prompt, k=k)
        candidates = [{"query": text, "source": "openai"} for text in generated]

        return {
            "prompt": prompt,
            "candidates": candidates,
            "metadata": {"k": k},
        }

    except Exception as exc:
        return {
            "prompt": prompt,
            "candidates": [],
            "metadata": {"k": k, "error": str(exc)},
        }