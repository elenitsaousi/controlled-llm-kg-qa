import json
from pathlib import Path
from kg.schema import load_default_schema
from ranking.features import score_candidate

BASE = Path(__file__).resolve().parents[1]

FEATURES_FILE = BASE / "ranking" / "features_domain.json"
CANDIDATES_DIR = BASE / "data" / "toy_kg" / "experiments" / "candidates"
OUTPUT_FILE = BASE / "results" / "schema_predictions.json"

OUTPUT_FILE.parent.mkdir(exist_ok=True)

with open(FEATURES_FILE) as f:
    feature_entries = json.load(f)

def load_queries():
    queries = {}
    for fpath in CANDIDATES_DIR.glob("Q*_candidates.json"):
        with open(fpath) as f:
            data = json.load(f)
        for c in data.get("candidates", []):
            queries[c["id"]] = c["query"]
    return queries

queries_by_id = load_queries()
schema = load_default_schema()

schema_predictions = {}

for qid, candidates in feature_entries.items():
    best_cid = None
    best_score = float("-inf")

    for c in candidates:
        cid = c.get("candidate_id") or c.get("query_id")
        if not cid:
            continue
        query = queries_by_id.get(cid)
        if not query:
            continue

        score = score_candidate(query, schema)
        if score > best_score:
            best_score = score
            best_cid = cid

    if best_cid is not None:
        schema_predictions[qid] = best_cid

with open(OUTPUT_FILE, "w") as f:
    json.dump(schema_predictions, f, indent=2)

print(f"Saved schema predictions to {OUTPUT_FILE}")
