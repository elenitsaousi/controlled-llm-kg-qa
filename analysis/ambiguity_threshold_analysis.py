import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import entropy as shannon_entropy

# ===============================
# Paths
# ===============================

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

OUTPUT_DIR = BASE / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = BASE / "ranking" / "features_domain.json"
GOLD_FILE = BASE / "ranking" / "data" / "gold_labels.json"
ENTROPY_FILE = OUTPUT_DIR / "entropy_per_question.json"
CANDIDATES_DIR = BASE / "data" / "toy_kg" / "experiments" / "candidates"
LOGISTIC_MODEL = BASE / "ranking" / "models" / "logistic_ranker.joblib"

from kg.schema import load_default_schema
from ranking.features import score_candidate

# ===============================
# Load data
# ===============================

if not FEATURES_FILE.exists():
    raise FileNotFoundError(f"Missing features file: {FEATURES_FILE}")

if not GOLD_FILE.exists():
    raise FileNotFoundError(f"Missing gold labels file: {GOLD_FILE}")

with open(FEATURES_FILE) as f:
    feature_entries = json.load(f)

with open(GOLD_FILE) as f:
    gold = json.load(f)

# ===============================
# Entropy computation
# ===============================

def compute_candidate_score(features: dict) -> float:
    """
    Coverage-based relevance score used ONLY for ambiguity estimation.
    This score is NOT used for ranking or evaluation.
    """
    return (
        0.4 * features.get("entity_coverage", 0.0)
        + 0.4 * features.get("relation_coverage", 0.0)
        + 0.2 * features.get("entity_precision", 0.0)
    )


def compute_entropy_per_question(entries: dict) -> dict:
    entropy_by_qid = {}

    for qid, candidate_list in entries.items():
        scores = np.array(
            [compute_candidate_score(c["features"]) for c in candidate_list],
            dtype=float,
        )

        if scores.size == 0 or np.all(scores == 0):
            entropy_by_qid[qid] = 0.0
            continue

        # softmax normalization
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)

        entropy_by_qid[qid] = float(shannon_entropy(probs))

    return entropy_by_qid


def load_candidate_queries(candidates_dir: Path) -> dict:
    if not candidates_dir.exists():
        raise FileNotFoundError(f"Missing candidates directory: {candidates_dir}")

    queries_by_id = {}
    for cand_file in sorted(candidates_dir.glob("Q*_candidates.json")):
        with open(cand_file) as f:
            data = json.load(f)
        for cand in data.get("candidates", []):
            cid = cand.get("id")
            query = cand.get("query")
            if cid and query:
                queries_by_id[cid] = query
    return queries_by_id


def load_learning_ranker(model_path: Path):
    try:
        from ranking.runtime_ranker import LogisticRanker
    except Exception as exc:
        print(f"WARNING: learning ranker import failed: {exc}")
        return None

    if not model_path.exists():
        print(f"WARNING: learning model not found: {model_path}")
        return None

    try:
        return LogisticRanker(str(model_path))
    except Exception as exc:
        print(f"WARNING: learning model load failed: {exc}")
        return None


# compute entropy if not already stored
if not ENTROPY_FILE.exists():
    entropy = compute_entropy_per_question(feature_entries)
    with open(ENTROPY_FILE, "w") as f:
        json.dump(entropy, f, indent=2, sort_keys=True)
else:
    with open(ENTROPY_FILE) as f:
        entropy = json.load(f)

# ===============================
# Ranking functions
# ===============================

def candidate_id(candidate: dict) -> str:
    return candidate.get("candidate_id") or candidate.get("query_id") or ""


def schema_score(candidate: dict, queries_by_id: dict, schema) -> float:
    """
    Schema-based baseline using validation penalties.
    Higher score is better (fewer errors and shorter queries).
    """
    cid = candidate_id(candidate)
    query = queries_by_id.get(cid)
    if not query:
        f = candidate.get("features", {})
        return (
            f.get("relation_precision", 0.0)
            + f.get("entity_precision", 0.0)
            - f.get("unexpected_label_ratio", 0.0)
        )
    return score_candidate(query, schema)


def learning_scores(candidate_list: list, ranker) -> np.ndarray:
    if not candidate_list:
        return np.array([])

    if all("ml_score" in c.get("features", {}) for c in candidate_list):
        return np.array(
            [c["features"]["ml_score"] for c in candidate_list],
            dtype=float,
        )

    if ranker is not None:
        try:
            return np.array(
                ranker.score([c["features"] for c in candidate_list]),
                dtype=float,
            )
        except Exception as exc:
            print(f"WARNING: learning ranker scoring failed: {exc}")

    return np.array(
        [compute_candidate_score(c["features"]) for c in candidate_list],
        dtype=float,
    )


# ===============================
# Core experiment
# ===============================

rows = []
queries_by_id = load_candidate_queries(CANDIDATES_DIR)
schema = load_default_schema()
ranker = load_learning_ranker(LOGISTIC_MODEL)

for qid, candidate_list in feature_entries.items():
    if qid not in gold or qid not in entropy:
        continue

    # schema-based selection
    schema_best = max(
        candidate_list,
        key=lambda c: schema_score(c, queries_by_id, schema),
    )

    # learning-based selection
    scores = learning_scores(candidate_list, ranker)
    if scores.size == 0:
        continue
    learning_best = candidate_list[int(np.argmax(scores))]

    rows.append({
        "question_id": qid,
        "entropy": entropy[qid],
        "schema_correct": int(candidate_id(schema_best) == gold[qid]),
        "learning_correct": int(candidate_id(learning_best) == gold[qid]),
    })

df = pd.DataFrame(rows)
df["delta"] = df["learning_correct"] - df["schema_correct"]

# ===============================
# Ambiguity regimes
# ===============================

df["entropy_bin"] = pd.qcut(
    df["entropy"],
    q=3,
    labels=["low", "mid", "high"],
    duplicates="drop"
)

summary = df.groupby("entropy_bin").agg(
    n=("delta", "count"),
    schema_acc=("schema_correct", "mean"),
    learning_acc=("learning_correct", "mean"),
    mean_delta=("delta", "mean")
)

print("\n=== AMBIGUITY REGIME SUMMARY ===")
print(summary)

summary.to_csv(OUTPUT_DIR / "ambiguity_summary.csv")

# ===============================
# Plots
# ===============================

plt.figure(figsize=(6, 4))
summary["mean_delta"].plot(kind="bar")
plt.axhline(0, color="black", linestyle="--")
plt.ylabel("Mean Learning Gain (Δ)")
plt.title("Learning Gain vs Ambiguity Regime")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_gain_vs_entropy.png")
plt.close()

plt.figure(figsize=(6, 4))
summary["schema_acc"].plot(marker="o", label="Schema")
summary["learning_acc"].plot(marker="o", label="Learning")
plt.ylabel("Top-1 Accuracy")
plt.title("Accuracy vs Ambiguity Regime")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_vs_entropy.png")
plt.close()

print("\nDONE: results written to analysis_outputs/")
