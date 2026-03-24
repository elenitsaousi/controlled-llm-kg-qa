import sys
from pathlib import Path

# allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
from visualization.ambiguity_metrics import ambiguity_entropy

# -------------------------
# paths
# -------------------------

features_path = "ranking/features_domain.json"
schema_path = "results/schema_predictions.json"
learning_path = "results/learning_predictions.json"
gold_path = "ranking/data/gold_labels.json"

# -------------------------
# load files
# -------------------------

features = json.load(open(features_path))
schema = json.load(open(schema_path))
learning = json.load(open(learning_path))
gold = json.load(open(gold_path))

# -------------------------
# compute entropy per question
# -------------------------

entropy_per_q = {}

for qid, candidates in features.items():

    scores = []

    for c in candidates:

        error_count = c.get("error_count", 0)
        length = c.get("length", 0)

        # same scoring function used in schema ranker
        score = -(10.0 * error_count + 0.001 * length)

        scores.append(score)

    scores = np.array(scores, dtype=float)

    entropy = float(ambiguity_entropy(scores))

    entropy_per_q[qid] = entropy


# save entropy file (optional)
Path("analysis_outputs").mkdir(exist_ok=True)

with open("analysis_outputs/entropy_per_question.json", "w") as f:
    json.dump(entropy_per_q, f, indent=2)

print("Wrote entropy_per_question.json")


# -------------------------
# ambiguity bins
# -------------------------

bins = {
    "low": [],
    "medium": [],
    "high": []
}

for qid in gold:

    if qid not in entropy_per_q:
        continue

    h = entropy_per_q[qid]

    if h < 0.5:
        level = "low"
    elif h < 1.0:
        level = "medium"
    else:
        level = "high"

    schema_correct = schema.get(qid) == gold[qid]
    learning_correct = learning.get(qid) == gold[qid]

    bins[level].append((schema_correct, learning_correct))


# -------------------------
# print results
# -------------------------

print("\nAccuracy per ambiguity level:\n")

for level, data in bins.items():

    if len(data) == 0:
        continue

    schema_acc = sum(x[0] for x in data) / len(data)
    learning_acc = sum(x[1] for x in data) / len(data)

    print(
        level,
        "| questions:", len(data),
        "| schema:", round(schema_acc, 3),
        "| ML:", round(learning_acc, 3)
    )