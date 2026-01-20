from typing import Dict, Optional
import os
import sys

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from kg.executor import execute_query_stub
from kg.schema import KGSchema
from llm.answer_synthesis import synthesize_answer
from llm.candidate_generation import generate_candidates
from ranking.feature_extraction import extract_features
from ranking.features import collect_errors
from ranking.ranker import rank_candidates
from ranking.runtime_ranker import LogisticRanker, XGBRanker


def answer_question(
    question: str,
    schema: KGSchema,
    questions_path: Optional[str] = None,
    llm_client: Optional[object] = None,
) -> Dict[str, object]:
    candidate_bundle = generate_candidates(
        question, schema, k=5, llm_client=llm_client
    )
    candidates = candidate_bundle.get("candidates", [])
    metadata = candidate_bundle.get("metadata", {})
    schema_def = {
        "labels": schema.labels,
        "relationships": schema.relationships,
    }
    feature_dicts = [
        extract_features(question, c.get("query", ""), schema_def)
        for c in candidates
    ]

    best = None
    ranker_name = None
    ranker_error = None

    try:
        ranker = XGBRanker()
        ranker_name = "xgb"
    except Exception as exc:
        ranker = None
        ranker_error = f"xgb load failed: {exc}"

    if ranker is None:
        try:
            ranker = LogisticRanker()
            ranker_name = "logistic"
        except Exception as exc:
            ranker = None
            ranker_error = (
                f"{ranker_error}; logistic load failed: {exc}"
                if ranker_error
                else f"logistic load failed: {exc}"
            )

    if ranker is not None:
        try:
            # Learning-based selection layer for candidate queries.
            scores = ranker.score(feature_dicts)
            if scores.size:
                best_idx = int(np.argmax(scores))
                best = {
                    "query": candidates[best_idx].get("query", ""),
                    "source": candidates[best_idx].get("source", "unknown"),
                    "score": float(scores[best_idx]),
                    "errors": collect_errors(
                        candidates[best_idx].get("query", ""), schema
                    ),
                }
                metadata["ranker"] = ranker_name
            else:
                ranker_error = "Ranker produced no scores."
        except Exception as exc:
            ranker_error = f"{ranker_name} scoring failed: {exc}"
            ranker = None

    if best is None:
        ranked = rank_candidates(candidate_bundle["candidates"], schema)
        best = ranked[0] if ranked else None
        metadata["ranker"] = "schema_only"
        if ranker_error:
            metadata["ranker_error"] = ranker_error
    if best is None:
        error_message = metadata.get("error")
        if error_message:
            error_text = f"No candidates generated. LLM error: {error_message}"
        else:
            error_text = "No candidates generated."
        return {
            "question": question,
            "answer": f"Answer (placeholder): {error_text}",
            "selected_query": None,
            "errors": [{"type": "pipeline", "message": error_text}],
            "candidates": candidates,
            "metadata": metadata,
        }

    errors = best.get("errors", [])
    if errors:
        answer = synthesize_answer(question, best["query"], {}, errors)
        return {
            "question": question,
            "answer": answer,
            "selected_query": best["query"],
            "errors": errors,
            "candidates": candidates,
            "metadata": metadata,
        }

    results = execute_query_stub(
        best["query"], questions_path=questions_path, question=question
    )
    answer = synthesize_answer(question, best["query"], results)
    return {
        "question": question,
        "answer": answer,
        "selected_query": best["query"],
        "errors": [],
        "results": results,
        "candidates": candidates,
        "metadata": metadata,
    }
