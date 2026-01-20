import json
import os
from typing import Dict, List, Optional

from kg.schema import load_default_schema, load_schema
from pipeline.qa import answer_question


def _normalize(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _load_questions(questions_path: str) -> List[Dict[str, object]]:
    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(
    questions_path: str,
    schema_path: Optional[str] = None,
) -> Dict[str, object]:
    schema = load_schema(schema_path) if schema_path else load_default_schema()
    questions = _load_questions(questions_path)

    totals = {
        "total": 0,
        "no_candidates": 0,
        "validation_failed": 0,
        "exact_query_match": 0,
    }

    for item in questions:
        totals["total"] += 1
        question = str(item.get("question", ""))
        gold_query = str(item.get("gold_query", ""))
        result = answer_question(
            question, schema, questions_path=questions_path
        )
        selected = result.get("selected_query")
        errors = result.get("errors", [])
        if selected is None:
            totals["no_candidates"] += 1
            continue
        if errors:
            totals["validation_failed"] += 1
            continue
        if _normalize(selected) == _normalize(gold_query):
            totals["exact_query_match"] += 1

    totals["exact_match_rate"] = (
        totals["exact_query_match"] / totals["total"]
        if totals["total"]
        else 0.0
    )
    return totals


def default_questions_path() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(
        base_dir, "data", "toy_kg", "questions", "questions.json"
    )
