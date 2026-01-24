# visualization/run_visualization.py

import json
import glob
import numpy as np
from collections import defaultdict
import os


from ambiguity_metrics import ambiguity_entropy
from selection_outcomes import selected_rank
from plot_candidate_ambiguity import plot_candidate_ambiguity

# -------------------------
# Paths
# -------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANDIDATES_DIR = os.path.join(
    BASE_DIR,
    "data",
    "toy_kg",
    "experiments",
    "candidates"
)

FEATURES_FILE = os.path.join(BASE_DIR, "ranking", "features_domain.json")
QUESTIONS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "toy_kg",
    "questions",
    "questions.json"
)




# -------------------------
# Load features_domain.json
# -------------------------
with open(FEATURES_FILE) as f:
    feature_entries = json.load(f)

features_by_candidate = {}

for qid, candidate_list in feature_entries.items():
    for entry in candidate_list:
        cid = entry["query_id"]          # π.χ. Q21_C2
        features_by_candidate[cid] = entry["features"]


# -------------------------
# Load questions.json (for gold_query)
# -------------------------
with open(QUESTIONS_FILE) as f:
    questions = json.load(f)

gold_by_qid = {
    q["id"]: q["gold_query"]
    for q in questions
}

# -------------------------
# Feature-based scoring (SYSTEM-DERIVED)
# -------------------------
def compute_score(features):
    """
    Deterministic relevance score derived ONLY from system features.
    """
    return (
        0.4 * features["entity_coverage"] +
        0.4 * features["relation_coverage"] +
        0.2 * features["entity_precision"]
    )

# -------------------------
# Containers for visualization
# -------------------------
ambiguity_scores = []
baseline_ranks = []
improved_ranks = []
ground_truth_ranks = []

baseline_loss = []
improved_loss = []
constraint_loss = []

# -------------------------
# Iterate over candidate files
# -------------------------
candidate_files = sorted(glob.glob(f"{CANDIDATES_DIR}/Q*_candidates.json"))
print("Found candidate files:", candidate_files)

for cand_file in candidate_files:
    with open(cand_file) as f:
        cand_data = json.load(f)

    # --- Resolve query id ---
    qid = os.path.basename(cand_file).split("_")[0]
    print("Processing query:", qid)

    candidates = cand_data["candidates"]

    scores = []
    penalties = []

    for idx, cand in enumerate(candidates):
        cid = f"{qid}_C{idx + 1}"
        feats = features_by_candidate[cid]

        scores.append(compute_score(feats))
        penalties.append(
            (1.0 - feats["relation_precision"]) +
            feats["unexpected_label_ratio"]
        )

    scores = np.array(scores)
    penalties = np.array(penalties)

    # --- Ground-truth proxy (explicitly documented later!) ---
    gt_index = int(np.argmax([
        features_by_candidate[f"{qid}_C{i+1}"]["entity_coverage"] +
        features_by_candidate[f"{qid}_C{i+1}"]["relation_coverage"]
        for i in range(len(candidates))
    ]))

    print("Scores:", scores)
    print("GT index:", gt_index)

    # --- Ambiguity ---
    ambiguity_scores.append(ambiguity_entropy(scores))

    # --- Baseline ---
    baseline_ranks.append(selected_rank(scores, gt_index))
    baseline_loss.append(1.0 - scores[gt_index])

    # --- Constraint-aware ---
    improved_scores = scores - penalties
    improved_ranks.append(selected_rank(improved_scores, gt_index))
    improved_loss.append(1.0 - improved_scores[gt_index])

    # --- Diagnostic ---
    constraint_loss.append(np.mean(penalties))


    ground_truth_ranks.append(1)

# -------------------------
# Visualization (UNCHANGED)
# -------------------------
plot_candidate_ambiguity(
    ambiguity_scores,
    baseline_ranks,
    improved_ranks,
    ground_truth_ranks,
    baseline_loss,
    improved_loss,
    constraint_loss
)
