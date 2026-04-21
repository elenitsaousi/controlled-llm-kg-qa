# candidate_generation.py
import json
from typing import Dict, List, Optional
from kg.schema import KGSchema
from kg.entity_linking import EntityAliasIndex, canonicalize_question_with_index
from llm.prompts import build_candidate_prompt, build_repair_prompt
from llm.client import InfineonGPTClient
import re


def _normalize_candidate_query(text: str) -> str:
    q = (text or "").strip()
    if not q:
        return ""
    q = q.replace("```sparql", "").replace("```sql", "").replace("```", "").strip()
    q = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", q)
    q = q.strip().strip(",")

    # Keep only the SPARQL part if the model prepends explanations.
    sel_idx = q.upper().find("SELECT")
    if sel_idx > 0:
        q = q[sel_idx:].strip()
    return q


def generate_candidate_prompt(
    question: str,
    schema: KGSchema,
    k: int = 5,
    canonical_question: Optional[str] = None,
    entity_mappings: Optional[List[Dict[str, object]]] = None,
) -> str:
    return build_candidate_prompt(
        question=question,
        schema=schema,
        k=k,
        canonical_question=canonical_question,
        entity_mappings=entity_mappings,
    )


def _default_client():
    import os

    provider = (os.getenv("LLM_PROVIDER") or os.getenv("LLM_BACKEND", "infineon")).strip().lower()
    if provider == "infiineon":
        provider = "infineon"

    if provider == "infineon":
        return InfineonGPTClient()
    raise ValueError(
        f"Unknown/unsupported LLM backend '{provider}'. "
        "Supported backend: infineon."
    )


def _normalize_generated_output(generated: object) -> List[str]:
    if generated is None:
        return []
    if isinstance(generated, str):
        cleaned = generated.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        split_candidates = re.split(r"\n\d+\.\s*", generated)
        if len(split_candidates) <= 1:
            split_candidates = generated.split("\n")
        return [g.strip() for g in split_candidates if g.strip()]
    if isinstance(generated, list):
        return [str(g).strip() for g in generated if str(g).strip()]
    raise ValueError(f"Unexpected LLM output type: {type(generated)}")


def repair_candidate_query(
    question: str,
    schema: KGSchema,
    invalid_query: str,
    error_message: str,
    llm_client: Optional[object] = None,
) -> Optional[str]:
    prompt = build_repair_prompt(
        question=question,
        schema=schema,
        invalid_query=invalid_query,
        error_message=error_message,
    )
    client = llm_client or _default_client()
    generated = client.generate(prompt, k=1)
    items = _normalize_generated_output(generated)
    if not items:
        return None
    repaired = _normalize_candidate_query(items[0])
    return repaired or None


def generate_candidates(
    question: str,
    schema: KGSchema,
    k: int = 5,
    llm_client: Optional[object] = None,
    entity_alias_index: Optional[EntityAliasIndex] = None,
    max_entity_links: int = 5,
) -> Dict:
    resolved = canonicalize_question_with_index(
        question,
        index=entity_alias_index,
        max_matches=max_entity_links,
    )
    effective_question = resolved.effective_question
    entity_mappings = resolved.mappings

    # Build prompt
    prompt = build_candidate_prompt(
        question=question,
        schema=schema,
        k=k,
        canonical_question=effective_question,
        entity_mappings=entity_mappings,
    )

    # Use provided client or default
    client = llm_client or _default_client()



    try:
        # Call LLM
        generated = client.generate(prompt, k=k)


        # -----------------------------
        # Normalize LLM output
        # -----------------------------
        generated = _normalize_generated_output(generated)


        seen = set()
        candidates = []
        for text in generated:
            query = _normalize_candidate_query(str(text))
            if not query:
                continue
            key = " ".join(query.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"query": query, "source": "infineon"})
            if len(candidates) >= k:
                break

        if not candidates:
            print("⚠️ WARNING: No candidates generated!")

        return {
            "prompt": prompt,
            "candidates": candidates,
            "metadata": {
                "k_requested": k,
                "k_returned": len(candidates),
                "original_question": (question or "").strip(),
                "effective_question": effective_question,
                "entity_mappings": entity_mappings,
                "entity_linking_applied": bool(entity_mappings),
            },
        }

    except Exception as exc:
        print("\n❌ LLM GENERATION FAILED:")
        print(exc)
        raise
