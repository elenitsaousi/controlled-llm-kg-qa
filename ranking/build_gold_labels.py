import csv
import json
from pathlib import Path


CANDIDATES_DIR = Path("data/toy_kg/experiments/candidates")
ANNOTATIONS_PATH = Path("data/annotations_llm.csv")
OUT_PATH = Path("ranking/data/gold_labels.json")


def normalize_question(text: str) -> str:
    return " ".join(text.split())


def load_candidates():
    candidates = {}
    for cand_file in sorted(CANDIDATES_DIR.glob("Q*_candidates.json")):
        data = json.loads(cand_file.read_text())
        qid = data["question_id"]
        candidates[qid] = {
            "question": normalize_question(data["question"]),
            "candidate_ids": {c["id"] for c in data["candidates"]},
        }
    return candidates


def main() -> None:
    candidates = load_candidates()

    correct_by_qid = {}
    mismatched_questions = set()
    missing_candidates = set()
    invalid_candidate_ids = set()

    with ANNOTATIONS_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("question_id") or "").strip()
            if not qid or qid not in candidates:
                continue

            question_text = normalize_question(row.get("question_text", ""))
            if question_text != candidates[qid]["question"]:
                mismatched_questions.add(qid)
                continue

            cid = (row.get("candidate_id") or "").strip()
            if cid not in candidates[qid]["candidate_ids"]:
                invalid_candidate_ids.add((qid, cid))
                continue

            label = (row.get("label") or "").strip().strip('"')
            if label != "CORRECT":
                continue

            if qid in correct_by_qid:
                correct_by_qid[qid].append(cid)
            else:
                correct_by_qid[qid] = [cid]

    gold = {}
    multiple_correct = []
    for qid, cids in correct_by_qid.items():
        if len(cids) == 1:
            gold[qid] = cids[0]
        else:
            multiple_correct.append((qid, cids))

    missing_correct = sorted(
        [qid for qid in candidates if qid not in correct_by_qid],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else x,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(gold)} gold labels to {OUT_PATH}")
    if mismatched_questions:
        print(f"Skipped {len(mismatched_questions)} due to question mismatch.")
    if invalid_candidate_ids:
        print(f"Skipped {len(invalid_candidate_ids)} rows with unknown candidate IDs.")
    if multiple_correct:
        print(f"Skipped {len(multiple_correct)} questions with multiple CORRECT labels.")
    if missing_correct:
        print(f"No CORRECT labels for {len(missing_correct)} questions.")


if __name__ == "__main__":
    main()
