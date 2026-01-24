import json
import numpy as np
from pathlib import Path
from ranking.runtime_ranker import LogisticRanker

BASE = Path(__file__).resolve().parents[1]

FEATURES_FILE = BASE / "ranking" / "features_domain.json"
MODEL_FILE = BASE / "ranking" / "models" / "logistic_ranker.joblib"
OUTPUT_FILE = BASE / "results" / "learning_predictions.json"

OUTPUT_FILE.parent.mkdir(exist_ok=True)

with open(FEATURES_FILE) as f:
    feature_entries = json.load(f)

ranker = LogisticRanker(str(MODEL_FILE))

learning_predictions = {}

for qid, candidates in feature_entries.items():
    features = [c["features"] for c in candidates]
    scores = ranker.score(features)
    best_idx = int(np.argmax(scores))
    best = candidates[best_idx]
    learning_predictions[qid] = best.get("candidate_id") or best.get("query_id")

with open(OUTPUT_FILE, "w") as f:
    json.dump(learning_predictions, f, indent=2)

print(f"Saved learning predictions to {OUTPUT_FILE}")
