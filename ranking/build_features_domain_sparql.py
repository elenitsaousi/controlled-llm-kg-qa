import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ranking.feature_extraction import extract_features, load_schema

SCHEMA_PATH = Path("data/toy_kg/schema.json")
CANDIDATES_DIR = Path("data/toy_kg/experiments/sparql_candidates")
OUT_PATH = Path("ranking/features_domain_sparql.json")

QUESTIONS_PATH = Path("data/toy_kg/questions/questions.json")

with open(QUESTIONS_PATH, "r") as f:
    QUESTIONS = {q["id"]: q["gold_query"] for q in json.load(f)}


def qid_key(path: Path) -> int:
    name = path.stem.split("_")[0]
    if name.startswith("Q") and name[1:].isdigit():
        return int(name[1:])
    return 0


def main() -> None:
    schema = load_schema(str(SCHEMA_PATH))
    all_data = {}

    for cand_file in sorted(CANDIDATES_DIR.glob("Q*_candidates.json"), key=qid_key):
        with open(cand_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        qid = data["question_id"]
        question = data["question"]

        gold_query = QUESTIONS[qid]
        schema["current_gold_query"] = gold_query

        items = []
        for cand in data["candidates"]:
            feats = extract_features(question, cand["query"], schema)
            items.append({"query_id": cand["id"], "features": feats})

        all_data[qid] = items

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(all_data)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
