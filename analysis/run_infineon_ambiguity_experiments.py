# analysis/run_infineon_ambiguity_experiments.py
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from ranking.feature_config import FEATURE_NAMES
from visualization.ambiguity_metrics import ambiguity_entropy

# Infineon-specific paths
TRAINING_DATA_FILE = BASE / "ranking" / "infineon_training_data.json"
EVAL_FILE = BASE / "results" / "infineon_eval.json"
OUTPUT_DIR = BASE / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_training_data():
    with open(TRAINING_DATA_FILE) as f:
        return json.load(f)


def load_eval_results():
    with open(EVAL_FILE) as f:
        return json.load(f)


def entropy_from_scores(scores: List[float]) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    return float(ambiguity_entropy(arr))


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    reg: float = 0.01,
    epochs: int = 2000,
) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pos_weight = (neg / pos) if pos > 0 else 1.0
    for _ in range(epochs):
        logits = X @ w + b
        probs = 1.0 / (1.0 + np.exp(-logits))
        weights = np.where(y == 1, pos_weight, 1.0)
        error = (probs - y) * weights
        grad_w = (X.T @ error) / n + reg * w
        grad_b = float(np.mean(error))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_scores(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    logits = X @ w + b
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    # Load training data
    training_data = load_training_data()
    eval_data = load_eval_results()

    # Build X, y, meta
    X, y, meta = [], [], []
    for qid, candidates in training_data.items():
        for c in candidates:
            features = c.get("features", {})
            if not features:
                continue
            row = [float(features.get(name, 0)) for name in FEATURE_NAMES]
            X.append(row)
            y.append(float(c["is_correct"]))
            meta.append((qid, c["query_id"]))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    print(f"Training samples: {len(X)}")
    print(f"Correct: {int(sum(y))}")

    if len(X) == 0:
        print("No training data!")
        return

    # Train
    w, b = train_logistic_regression(X, y)
    ml_scores = predict_scores(X, w, b)
    ml_by_cid = {cid: float(score) for (_, cid), score in zip(meta, ml_scores)}

    # Load eval results for baseline comparison
    eval_details = {d["id"]: d for d in eval_data["details"]}

    rows = []

    for qid, candidates in training_data.items():
        if not candidates:
            continue

        # Get candidate scores
        candidate_scores = []
        best_schema = None
        best_schema_score = float("-inf")
        best_ml = None
        best_ml_score = float("-inf")

        for c in candidates:
            cid = c["query_id"]
            features = c.get("features", {})
            if not features:
                continue

            # Use ML score for entropy calculation
            l_score = ml_by_cid.get(cid, 0.0)
            candidate_scores.append(l_score)

            # Schema score: feature-based heuristic
            if features:
                s_score = (
                    features.get("has_aggregation", 0) * 0.3 +
                    features.get("has_where", 0) * 0.2 +
                    features.get("has_type", 0) * 0.2 +
                    max(0, 1 - features.get("invalid_predicate_count", 0) * 0.1) +
                    max(0, 1 - features.get("unused_select_vars", 0) * 0.1)
                )
            else:
                s_score = float(c.get("is_valid", 0))

            if s_score > best_schema_score:
                best_schema_score = s_score
                best_schema = c

            # ML score
            l_score = ml_by_cid.get(cid, float("-inf"))
            if l_score > best_ml_score:
                best_ml_score = l_score
                best_ml = c

        # Entropy from candidate scores
        entropy = entropy_from_scores(candidate_scores)

        # Correctness
        schema_correct = int(
            best_schema is not None and
            best_schema.get("is_correct", 0) == 1
        )
        learning_correct = int(
            best_ml is not None and
            best_ml.get("is_correct", 0) == 1
        )
        any_correct = int(
            any(c.get("is_correct", 0) == 1 for c in candidates)
        )

        rows.append({
            "question_id": qid,
            "entropy": entropy,
            "schema_correct": schema_correct,
            "learning_correct": learning_correct,
            "any_correct": any_correct,
        })

    # Entropy thresholds
    entropies = np.array([r["entropy"] for r in rows], dtype=float)
    H1 = float(np.quantile(entropies, 0.33)) if entropies.size else 0.0
    H2 = float(np.quantile(entropies, 0.66)) if entropies.size else 0.0

    entropy_stats = {
        "mean": float(np.mean(entropies)),
        "std": float(np.std(entropies)),
        "min": float(np.min(entropies)),
        "max": float(np.max(entropies)),
    }

    # Categorize
    for row in rows:
        h = row["entropy"]
        row["entropy_bin"] = (
            "low" if h <= H1 else
            "mid" if h <= H2 else
            "high"
        )

    # Summary per bin
    summary = {}
    for row in rows:
        b = row["entropy_bin"]
        entry = summary.setdefault(
            b, {"n": 0, "schema": 0, "learning": 0, "delta": 0}
        )
        entry["n"] += 1
        entry["schema"] += row["schema_correct"]
        entry["learning"] += row["learning_correct"]
        entry["delta"] += row["learning_correct"] - row["schema_correct"]

    for b, entry in summary.items():
        n = max(entry["n"], 1)
        entry["schema_acc"] = entry["schema"] / n
        entry["learning_acc"] = entry["learning"] / n
        entry["mean_delta"] = entry["delta"] / n

    # Overall
    schema_acc = sum(r["schema_correct"] for r in rows) / max(len(rows), 1)
    learning_acc = sum(r["learning_correct"] for r in rows) / max(len(rows), 1)

    print(f"\n=== INFINEON AMBIGUITY ANALYSIS ===")
    print(f"Entropy thresholds: H1={H1:.3f}, H2={H2:.3f}")
    print(f"Entropy stats: {entropy_stats}")
    print(f"\nOverall Schema acc: {schema_acc:.3f}")
    print(f"Overall Learning acc: {learning_acc:.3f}")
    print(f"\nPer-ambiguity summary:")
    for b in ["low", "mid", "high"]:
        if b in summary:
            e = summary[b]
            print(f"  {b}: n={e['n']}, "
                  f"schema={e['schema_acc']:.2%}, "
                  f"learning={e['learning_acc']:.2%}, "
                  f"delta={e['mean_delta']:+.2%}")

    # Save outputs
    with open(OUTPUT_DIR / "infineon_entropy_per_question.json", "w") as f:
        json.dump({r["question_id"]: r["entropy"] for r in rows}, f, indent=2)

    with open(OUTPUT_DIR / "infineon_entropy_stats.json", "w") as f:
        json.dump(entropy_stats, f, indent=2)

    with open(OUTPUT_DIR / "infineon_ambiguity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "infineon_ambiguity_details.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved to analysis_outputs/")


if __name__ == "__main__":
    main()