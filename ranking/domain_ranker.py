from ranking.domain_features import extract_domain_features

WEIGHTS = {
    "expected_path": 2.0,
    "forbidden_shortcut": -2.0,
    "relevant_relations": 1.0,
    "simple_structure": 1.0,
    "entity_constraint": 1.0,
}

def score_query(query_text: str, question_type: str) -> float:
    feats = extract_domain_features(query_text, question_type)
    score = 0.0
    for k, w in WEIGHTS.items():
        score += w * feats.get(k, 0)
    return score


def rank_queries(candidates: list, question_type: str) -> list:
    """
    Rank candidate queries using domain-aware scoring with deterministic tie-breaking.
    No ground-truth labels are used.
    """
    for c in candidates:
        feats = extract_domain_features(c["query_text"], question_type)
        c["score"] = sum(WEIGHTS[k] * feats.get(k, 0) for k in WEIGHTS)
        c["feats"] = feats

    def tie_key(c):
        f = c["feats"]
        return (
            c["score"],                      # primary: domain score
            f.get("relevant_relations", 0),  # prefer domain-relevant relations
            f.get("entity_constraint", 0),   # prefer constrained queries
            f.get("simple_structure", 0),    # prefer simpler structure
            -len(c["query_text"]),           # prefer shorter queries
        )

    return sorted(candidates, key=tie_key, reverse=True)
