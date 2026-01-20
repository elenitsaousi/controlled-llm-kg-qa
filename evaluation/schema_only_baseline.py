import pandas as pd
import re

# Load data
df = pd.read_csv("data/annotations_llm.csv")
df.columns = df.columns.str.strip()
df["label"] = df["label"].str.strip()

# Define schema
VALID_LABELS = {
    "Supplier", "Material", "Wafer", "Product", "Fab", "Factory",
    "Lot", "ProcessStep", "Tool", "Defect"
}

VALID_RELATIONS = {
    "SUPPLIES", "USED_IN", "PRODUCES", "PRODUCED_FOR",
    "REQUIRES", "HAS_DEFECT", "HAS_CONSTRAINT", "HAS_YIELD",
    "PROCESSED_WITH", "AFFECTS"
}

def extract_labels(query):
    return re.findall(r":([A-Za-z_]+)", query)

def extract_relations(query):
    return re.findall(r"\[:([A-Za-z_]+)\]", query)

def schema_score(query):
    labels = extract_labels(query)
    relations = extract_relations(query)

    # Hard exclusion
    for l in labels:
        if l not in VALID_LABELS:
            return -100
    for r in relations:
        if r not in VALID_RELATIONS:
            return -100

    # Soft score
    score = 0
    score += len(labels)
    score += len(relations)
    return score

# Evaluation
correct = 0
total = df["question_id"].nunique()

for qid, group in df.groupby("question_id"):
    group = group.copy()
    group["score"] = group["query_text"].apply(schema_score)

    best = group.sort_values("score", ascending=False).iloc[0]
    if best["label"] == "CORRECT":
        correct += 1

accuracy = correct / total
print(f"Schema-only Top-1 Accuracy: {accuracy:.2f}")
