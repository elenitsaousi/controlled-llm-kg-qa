import json
import os
from typing import Dict, List, Optional, Set, Tuple


def _load_questions(questions_path: str) -> List[Dict[str, object]]:
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_results(results_path: str) -> Dict[str, List[Dict[str, object]]]:
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _tokenize(text: str) -> Set[str]:
    return {token for token in _normalize(text).split() if token}


def _intent_keywords() -> Dict[str, List[str]]:
    return {
        "cross_domain_dependency": ["affect", "impact", "yield", "supplier"],
        "aggregate_metric_by_entity": ["average", "avg", "mean", "by"],
        "defect_traceability": ["defect", "tool", "trace", "linked"],
        "material_supply_for_process": ["material", "process", "lithography"],
        "capacity_constraint_impact": ["capacity", "constraint", "fab"],
        "logistics_delay_impact": ["shipment", "delayed", "order", "impact"],
        "risk_assessment": ["risk", "inventory", "disrupted"],
        "supplier_alternatives": ["alternative", "suppliers", "options"],
    }


def _score_intent_match(
    question: str, item: Dict[str, object]
) -> Tuple[float, str]:
    question_tokens = _tokenize(question)
    gold_tokens = _tokenize(str(item.get("question", "")))
    if not question_tokens or not gold_tokens:
        return 0.0, str(item.get("intent", ""))

    intersection = question_tokens & gold_tokens
    union = question_tokens | gold_tokens
    jaccard = len(intersection) / max(len(union), 1)

    intent = str(item.get("intent", ""))
    keywords = _intent_keywords().get(intent, [])
    if keywords:
        hits = sum(1 for kw in keywords if kw in question_tokens)
        keyword_score = hits / len(keywords)
    else:
        keyword_score = 0.0

    return jaccard + (0.25 * keyword_score), intent


def execute_query_stub(
    query: str,
    questions_path: Optional[str] = None,
    question: Optional[str] = None,
) -> Dict[str, object]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_path = os.path.join(
        base_dir, "data", "toy_kg", "questions", "questions.json"
    )
    results_path = os.path.join(
        base_dir, "data", "toy_kg", "results.json"
    )
    questions_file = questions_path or default_path

    if not os.path.exists(questions_file) or not os.path.exists(results_path):
        return {"rows": [], "matched_question_id": None, "error": None}

    questions = _load_questions(questions_file)
    results = _load_results(results_path)
    normalized_query = _normalize(query)
    for item in questions:
        gold = str(item.get("gold_query", "")).strip()
        if _normalize(gold) == normalized_query:
            qid = item.get("id")
            return {
                "rows": results.get(qid, []),
                "matched_question_id": qid,
                "error": None,
            }

    if question:
        best_score = 0.0
        best_item: Optional[Dict[str, object]] = None
        for item in questions:
            score, _ = _score_intent_match(question, item)
            if score > best_score:
                best_score = score
                best_item = item
        if best_item and best_score >= 0.15:
            qid = best_item.get("id")
            intent = best_item.get("intent")
            return {
                "rows": results.get(qid, []),
                "matched_question_id": qid,
                "matched_intent": intent,
                "error": None,
            }

    return {"rows": [], "matched_question_id": None, "error": None}
