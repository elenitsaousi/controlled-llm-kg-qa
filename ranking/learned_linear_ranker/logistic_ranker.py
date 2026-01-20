import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from collections import defaultdict
import joblib

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from ranking.feature_config import FEATURE_NAMES

if __name__ == "__main__":

    # -----------------------------
    # Load feature data
    # -----------------------------
    with open("ranking/features_domain.json") as f:
        data = json.load(f)

    # -----------------------------
    # Gold labels (per question)
    # -----------------------------
    with open("ranking/data/gold_labels.json") as f:
        gold = json.load(f)


    # -----------------------------
    # Build dataset (grouped by question)
    # -----------------------------
    X = []
    y = []
    groups = []
    meta = []  # (question_id, candidate_id)

    for qid, items in data.items():
        if qid not in gold:
            continue
        for item in items:
            ftrs = item["features"]

            row = [ftrs[name] for name in FEATURE_NAMES]
            label = 1 if item["query_id"] == gold[qid] else 0

            X.append(row)
            y.append(label)
            groups.append(qid)
            meta.append((qid, item["query_id"]))

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    def top1_accuracy(scores, meta_subset, gold_labels):
        ranked = defaultdict(list)
        for (qid, cid), score in zip(meta_subset, scores):
            ranked[qid].append((cid, score))

        correct = 0
        for qid, lst in ranked.items():
            selected = max(lst, key=lambda x: x[1])[0]
            if selected == gold_labels[qid]:
                correct += 1
        return correct / max(len(ranked), 1), ranked

    # -----------------------------
    # -----------------------------
    # Hold-out evaluation (group split by question)
    # -----------------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    model = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
    model.fit(X[train_idx], y[train_idx])

    scores = model.predict_proba(X[test_idx])[:, 1]
    meta_test = [meta[i] for i in test_idx]
    acc, ranked = top1_accuracy(scores, meta_test, gold)

    test_qids = sorted(set(groups[test_idx]), key=lambda q: int(q[1:]) if q[1:].isdigit() else q)
    print(f"\nHold-out: train questions={len(set(groups[train_idx]))} | test questions={len(test_qids)}")
    print(f"Top-1 Accuracy (Hold-out): {acc:.4f}")

    # -----------------------------
    # Cross-validation (grouped by question)
    # -----------------------------
    cv = GroupKFold(n_splits=5)
    cv_accs = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups), start=1):
        fold_model = LogisticRegression(
            solver="liblinear", class_weight="balanced", random_state=42
        )
        fold_model.fit(X[tr_idx], y[tr_idx])
        fold_scores = fold_model.predict_proba(X[te_idx])[:, 1]
        fold_meta = [meta[i] for i in te_idx]
        fold_acc, _ = top1_accuracy(fold_scores, fold_meta, gold)
        cv_accs.append(fold_acc)
        print(f"Fold {fold} Top-1 Accuracy: {fold_acc:.4f}")

    print(
        f"\nCV Top-1 Accuracy: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}"
    )

    # -----------------------------
    # Model interpretability
    # -----------------------------
    print("\nLearned feature weights:")
    print(model.coef_)

    model_dir = os.path.join("ranking", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "logistic_ranker.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")
