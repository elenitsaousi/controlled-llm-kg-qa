import json
import os
import sys
import numpy as np
import xgboost as xgb
from collections import defaultdict
from sklearn.model_selection import GroupKFold

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import ranking
from ranking.feature_config import FEATURE_NAMES


def load_dataset(features_path, gold_path):
    with open(features_path) as f:
        data = json.load(f)

    with open(gold_path) as f:
        gold = json.load(f)

    X, y, groups, meta = [], [], [], []

    for qid, items in data.items():
        if qid not in gold:
            continue
        for item in items:
            ftrs = item["features"]
            X.append([ftrs[n] for n in FEATURE_NAMES])
            y.append(1 if item["query_id"] == gold[qid] else 0)
            groups.append(qid)
            meta.append((qid, item["query_id"]))

    return np.array(X), np.array(y), np.array(groups), meta


def build_ranker():
    return {
        "objective": "rank:pairwise",
        "eta": 0.05,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "map",
        "seed": 42,
        "verbosity": 0,
        "lambda": 2.0
    }


def top1_accuracy(scores, meta, gold):
    ranked = defaultdict(list)
    for (qid, cid), score in zip(meta, scores):
        ranked[qid].append((cid, score))

    correct = 0
    for qid, lst in ranked.items():
        selected = max(lst, key=lambda x: x[1])[0]
        if selected == gold[qid]:
            correct += 1

    return correct / len(ranked)


if __name__ == "__main__":

    X, y, groups, meta = load_dataset(
        "ranking/features_domain.json",
        "ranking/data/gold_labels.json",
    )

    with open("ranking/data/gold_labels.json") as f:
        gold = json.load(f)

    cv = GroupKFold(n_splits=5)
    accs = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):

        # group sizes per fold
        train_groups = [np.sum(groups[tr] == g) for g in np.unique(groups[tr])]
        test_groups = [np.sum(groups[te] == g) for g in np.unique(groups[te])]

        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        dtrain.set_group(train_groups)

        dtest = xgb.DMatrix(X[te], label=y[te])
        dtest.set_group(test_groups)

        model = xgb.train(
            build_ranker(),
            dtrain,
            num_boost_round=100,
        )

        scores = model.predict(dtest)
        meta_test = [meta[i] for i in te]

        acc = top1_accuracy(scores, meta_test, gold)
        accs.append(acc)

        print(f"Fold {fold} Top-1 Accuracy: {acc:.4f}")

    print(f"\nXGBoost CV Top-1 Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
np.save("ranking/ml_learning_ranker/train_xgb.npy", model)