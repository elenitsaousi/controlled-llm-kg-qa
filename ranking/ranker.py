from typing import Dict, List, Optional

from kg.schema import KGSchema
from ranking.features import collect_errors, score_candidate


def rank_candidates(
    candidates: List[Dict[str, str]], schema: KGSchema
) -> List[Dict[str, object]]:
    scored: List[Dict[str, object]] = []
    for item in candidates:
        query = str(item.get("query", ""))
        score = score_candidate(query, schema)
        errors = collect_errors(query, schema)
        scored.append(
            {
                "query": query,
                "source": item.get("source", "unknown"),
                "score": score,
                "errors": errors,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def select_best_candidate(
    candidates: List[Dict[str, str]], schema: KGSchema
) -> Optional[Dict[str, object]]:
    ranked = rank_candidates(candidates, schema)
    if not ranked:
        return None
    return ranked[0]
