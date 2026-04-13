def classify_difficulty(query: str) -> str:
    q = query.lower()

    triple_count = q.count(".")
    has_group = "group by" in q
    has_bind = "bind(" in q
    has_nested = "if(" in q

    complexity_score = 0

    if triple_count >= 3:
        complexity_score += 1
    if triple_count >= 6:
        complexity_score += 1
    if has_group:
        complexity_score += 1
    if has_bind or has_nested:
        complexity_score += 1

    if complexity_score <= 1:
        return "easy"
    elif complexity_score == 2:
        return "medium"
    else:
        return "hard"