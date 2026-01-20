import json
from pathlib import Path

QUESTIONS_PATH = Path("data/toy_kg/questions/questions.json")
CANDIDATES_DIR = Path("data/toy_kg/experiments/candidates")


def infer_domain(question: str):
    q = question.lower()
    if "supplier" in q and "fab" in q:
        return ["supply_chain", "manufacturing"]
    if "supplier" in q or "shipment" in q or "inventory" in q:
        return ["supply_chain"]
    if "tool" in q or "defect" in q or "lot" in q or "fab" in q:
        return ["manufacturing"]
    return ["cross_domain"]


def infer_intent(question: str):
    q = question.lower()
    if "delay" in q or "delayed" in q:
        return "delay_analysis"
    if "defect" in q:
        return "defect_traceability"
    if "yield" in q:
        return "yield_analysis"
    if "inventory" in q:
        return "inventory_status"
    if "shipment" in q:
        return "shipment_dependency"
    if "supplier" in q:
        return "supplier_analysis"
    return "entity_retrieval"


def infer_difficulty(domains):
    if len(domains) > 1:
        return "hard"
    return "medium"


def main():
    with open(QUESTIONS_PATH) as f:
        existing = json.load(f)

    existing_ids = {q["id"] for q in existing}

    new_questions = []

    for cand_file in sorted(CANDIDATES_DIR.glob("Q*_candidates.json")):
        with open(cand_file) as f:
            data = json.load(f)

        qid = data["question_id"]
        question = data["question"]

        if qid in existing_ids:
            continue

        domains = infer_domain(question)

        new_questions.append({
            "id": qid,
            "question": question,
            "intent": infer_intent(question),
            "entities": [],  # optional / unknown
            "domain": domains,
            "difficulty": infer_difficulty(domains),
            "expected_answer_type": "list_of_entities"
        })

    if not new_questions:
        print("No new questions to add.")
        return

    merged = existing + new_questions

    with open(QUESTIONS_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Added {len(new_questions)} questions to questions.json")


if __name__ == "__main__":
    main()
