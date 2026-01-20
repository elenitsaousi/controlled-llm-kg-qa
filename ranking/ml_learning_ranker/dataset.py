import json
import os
import sys
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from ranking.feature_config import FEATURE_NAMES


def load_ranking_dataset(features_path: str, gold_path: str):
    """
    Builds dataset for learning-to-rank.

    features_path:
        JSON with structure:
        {
          "Q1": [
            {
              "query_id": "Q1_C1",
              "features": { ... }
            },
            ...
          ],
          ...
        }

    gold_path:
        JSON with structure:
        {
          "Q1": "Q1_C2",
          "Q2": "Q2_C1",
          ...
        }

    Returns:
        X      : np.ndarray (n_samples, n_features)
        y      : np.ndarray (n_samples,)
        group  : list[int]   (#candidates per question)
        meta   : list[(question_id, candidate_id)]
    """

    # Load data
    with open(features_path) as f:
        data = json.load(f)

    with open(gold_path) as f:
        gold = json.load(f)

    X = []
    y = []
    group = []
    meta = []

    # Build ranking dataset
    for qid, candidates in data.items():
        if qid not in gold:
            continue
        group.append(len(candidates))

        for cand in candidates:
            ftrs = cand["features"]

            X.append([ftrs[name] for name in FEATURE_NAMES])

            y.append(1 if cand["query_id"] == gold[qid] else 0)
            meta.append((qid, cand["query_id"]))

    return (
        np.array(X, dtype=float),
        np.array(y, dtype=int),
        group,
        meta,
    )


import json

with open("ranking/features_domain.json") as f:
    data = json.load(f)

print("Questions:", len(data))
