import sys
import os
import pandas as pd

# make project root visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ranking.domain_ranker import rank_queries

# load annotations
df = pd.read_csv("data/annotations_llm.csv")

# simple mapping from question_id to question_type
QUESTION_TYPE = {
    "Q1": "SUPPLIER_WAFER_PRODUCT",
    "Q2": "FACTORY_PRODUCT",
    "Q3": "SUPPLIER_WAFER_PRODUCT",
    "Q4": "FACTORY_PRODUCT",
    "Q5": "SUPPLIER_WAFER",
    "Q6": "SUPPLIER_WAFER_FACTORY",
}

correct = 0
total = df["question_id"].nunique()

for qid, group in df.groupby("question_id"):
    question_type = QUESTION_TYPE.get(qid, "UNKNOWN")
    candidates = group.to_dict("records")
    ranked = rank_queries(candidates, question_type)

    print(f"\nQuestion {qid}")
    for c in ranked:
        print(c["label"], c["score"], c["query_text"][:80])

    top1 = ranked[0]
    if top1["label"] == "CORRECT":
        correct += 1


accuracy = correct / total
print(f"Schema + Domain Top-1 Accuracy: {accuracy:.2f}")
