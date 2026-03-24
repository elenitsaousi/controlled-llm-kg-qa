import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from kg.schema import load_default_schema
from kg.sparql_matching import is_relaxed_correct
from ranking.feature_extraction import extract_features
from ranking.feature_config import FEATURE_NAMES
from ranking.features import score_candidate
from visualization.ambiguity_metrics import ambiguity_entropy


FEATURES_FILE = BASE / "ranking" / "features_domain_sparql.json"
CANDIDATES_DIR = BASE / "data" / "toy_kg" / "experiments" / "sparql_candidates"
QUESTIONS_FILE = BASE / "data" / "toy_kg" / "questions" / "questions.json"
OUTPUT_DIR = BASE / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_queries() -> Dict[str, str]:
    queries = {}
    for fpath in CANDIDATES_DIR.glob("Q*_candidates.json"):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cand in data.get("candidates", []):
            cid = cand.get("id")
            query = cand.get("query")
            if cid and query:
                queries[cid] = query
    return queries


def load_features():
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions() -> Dict[str, str]:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item.get("gold_query", "") for item in data}


def build_dataset(
    features: Dict[str, List[Dict[str, object]]],
    gold_queries: Dict[str, str],
    queries_by_id: Dict[str, str],
):
    X, y, meta = [], [], []
    for qid, items in features.items():
        if qid not in gold_queries:
            continue
        for item in items:
            ftrs = item["features"]
            cid = item["query_id"]
            cand_query = queries_by_id.get(cid, "")
            if not cand_query:
                continue
            row = [float(ftrs.get(name, 0.0)) for name in FEATURE_NAMES]
            label = 1 if is_relaxed_correct(cand_query, gold_queries[qid]) else 0
            X.append(row)
            y.append(label)
            meta.append((qid, cid))
    return np.array(X, dtype=float), np.array(y, dtype=float), meta


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


def entropy_from_scores(scores: List[float]) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores, dtype=float)
    return float(ambiguity_entropy(arr))


def main() -> None:
    schema = load_default_schema()
    features = load_features()
    gold_queries = load_questions()
    queries_by_id = load_queries()

    # Train lightweight logistic regression (numpy only).
    X, y, meta = build_dataset(features, gold_queries, queries_by_id)
    w, b = train_logistic_regression(X, y)
    ml_scores = predict_scores(X, w, b)

    # Index ML scores by candidate id.
    ml_by_cid = {cid: float(score) for (_, cid), score in zip(meta, ml_scores)}

    entropy_by_qid = {}
    schema_pred = {}
    learning_pred = {}
    rows = []

    for qid, items in features.items():
        if qid not in gold_queries:
            continue

        candidate_scores = []
        best_schema = None
        best_schema_score = float("-inf")

        best_ml = None
        best_ml_score = float("-inf")

        for item in items:
            cid = item["query_id"]
            query = queries_by_id.get(cid, "")
            if not query:
                continue

            s_score = score_candidate(query, schema)
            candidate_scores.append(s_score)
            if s_score > best_schema_score:
                best_schema_score = s_score
                best_schema = cid

            l_score = ml_by_cid.get(cid, float("-inf"))
            if l_score > best_ml_score:
                best_ml_score = l_score
                best_ml = cid

        entropy = entropy_from_scores(candidate_scores)
        entropy_by_qid[qid] = entropy
        if best_schema:
            schema_pred[qid] = best_schema
        if best_ml:
            learning_pred[qid] = best_ml

        schema_query = queries_by_id.get(schema_pred.get(qid, ""), "")
        learning_query = queries_by_id.get(learning_pred.get(qid, ""), "")
        schema_correct = int(
            bool(schema_query)
            and is_relaxed_correct(schema_query, gold_queries[qid])
        )
        learning_correct = int(
            bool(learning_query)
            and is_relaxed_correct(learning_query, gold_queries[qid])
        )

        rows.append(
            {
                "question_id": qid,
                "entropy": entropy,
                "schema_correct": schema_correct,
                "learning_correct": learning_correct,
            }
        )

    # thresholds
    entropies = np.array([r["entropy"] for r in rows], dtype=float)
    H1 = float(np.quantile(entropies, 0.33)) if entropies.size else 0.0
    H2 = float(np.quantile(entropies, 0.66)) if entropies.size else 0.0
    entropy_stats = {
        "mean": float(np.mean(entropies)) if entropies.size else 0.0,
        "std": float(np.std(entropies)) if entropies.size else 0.0,
        "min": float(np.min(entropies)) if entropies.size else 0.0,
        "max": float(np.max(entropies)) if entropies.size else 0.0,
    }

    for row in rows:
        h = row["entropy"]
        use_learning = H1 <= h <= H2
        row["gated_correct"] = (
            row["learning_correct"] if use_learning else row["schema_correct"]
        )
        row["entropy_bin"] = (
            "low" if h <= H1 else "mid" if h <= H2 else "high"
        )

    # summaries
    summary = {}
    for row in rows:
        b = row["entropy_bin"]
        entry = summary.setdefault(b, {"n": 0, "schema": 0, "learning": 0, "delta": 0})
        entry["n"] += 1
        entry["schema"] += row["schema_correct"]
        entry["learning"] += row["learning_correct"]
        entry["delta"] += row["learning_correct"] - row["schema_correct"]

    for b, entry in summary.items():
        n = max(entry["n"], 1)
        entry["schema_acc"] = entry["schema"] / n
        entry["learning_acc"] = entry["learning"] / n
        entry["mean_delta"] = entry["delta"] / n

    # overall accuracy
    schema_acc = sum(r["schema_correct"] for r in rows) / max(len(rows), 1)
    learning_acc = sum(r["learning_correct"] for r in rows) / max(len(rows), 1)
    gated_acc = sum(r["gated_correct"] for r in rows) / max(len(rows), 1)

    # write outputs
    with open(OUTPUT_DIR / "sparql_entropy_per_question.json", "w", encoding="utf-8") as f:
        json.dump(entropy_by_qid, f, indent=2, sort_keys=True)
        f.write("\n")

    with open(OUTPUT_DIR / "sparql_entropy_stats.json", "w", encoding="utf-8") as f:
        json.dump(entropy_stats, f, indent=2)
        f.write("\n")

    with open(OUTPUT_DIR / "sparql_gated_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"H1": H1, "H2": H2}, f, indent=2)
        f.write("\n")

    with open(OUTPUT_DIR / "sparql_gated_policy_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")

    with open(OUTPUT_DIR / "sparql_ambiguity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print("Entropy thresholds:", H1, H2)
    print("Entropy stats:", entropy_stats)
    print("Schema acc:", round(schema_acc, 3))
    print("Learning acc:", round(learning_acc, 3))
    print("Gated acc:", round(gated_acc, 3))
    print("Regime summary:", summary)


if __name__ == "__main__":
    main()
