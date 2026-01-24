import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# Paths
# ===============================

BASE = Path(__file__).resolve().parents[1]

ENTROPY_FILE = BASE / "analysis_outputs" / "entropy_per_question.json"
SCHEMA_FILE = BASE / "results" / "schema_predictions.json"
LEARNING_FILE = BASE / "results" / "learning_predictions.json"
GOLD_FILE = BASE / "ranking" / "data" / "gold_labels.json"

OUTPUT_DIR = BASE / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ===============================
# Load data
# ===============================

with open(ENTROPY_FILE) as f:
    entropy = json.load(f)

with open(SCHEMA_FILE) as f:
    schema_pred = json.load(f)

with open(LEARNING_FILE) as f:
    learning_pred = json.load(f)

with open(GOLD_FILE) as f:
    gold = json.load(f)

# ===============================
# Build evaluation table
# ===============================

rows = []

for qid in gold.keys():
    if qid not in entropy or qid not in schema_pred or qid not in learning_pred:
        continue
    rows.append({
        "question_id": qid,
        "entropy": entropy[qid],
        "schema_correct": int(schema_pred[qid] == gold[qid]),
        "learning_correct": int(learning_pred[qid] == gold[qid]),
    })

df = pd.DataFrame(rows)

# ===============================
# Define entropy thresholds (OBJECTIVE)
# ===============================

H1 = np.quantile(df["entropy"], 0.33)
H2 = np.quantile(df["entropy"], 0.66)

print(f"Entropy thresholds: H1={H1:.4f}, H2={H2:.4f}")

# ===============================
# Gated policy
# ===============================

def gated_policy(row):
    """
    Use learning only in medium ambiguity regime.
    """
    if H1 <= row["entropy"] <= H2:
        return row["learning_correct"]
    else:
        return row["schema_correct"]

df["gated_correct"] = df.apply(gated_policy, axis=1)

df["entropy_bin"] = pd.qcut(
    df["entropy"],
    q=3,
    labels=["low", "mid", "high"],
    duplicates="drop"
)

regime_summary = df.groupby("entropy_bin").agg(
    n=("question_id", "count"),
    schema_acc=("schema_correct", "mean"),
    learning_acc=("learning_correct", "mean"),
    gated_acc=("gated_correct", "mean"),
)

print("\n=== Regime-wise Accuracy ===")
print(regime_summary)


# ===============================
# Accuracy comparison
# ===============================

schema_acc = df["schema_correct"].mean()
learning_acc = df["learning_correct"].mean()
gated_acc = df["gated_correct"].mean()


improved = df[
    (df["gated_correct"] == 1) &
    ((df["schema_correct"] == 0) | (df["learning_correct"] == 0))
]

print(improved[["question_id", "entropy",
                "schema_correct", "learning_correct"]])


print("\nQuestions where learning hurts but gated avoids it:")
print(df[
    (df["learning_correct"] == 0) &
    (df["schema_correct"] == 1) &
    (df["gated_correct"] == 1)
][["question_id", "entropy"]])


for shift in [-0.002, 0.0, 0.002]:
    H1p = H1 + shift
    H2p = H2 + shift
    acc = df.apply(
        lambda r: r["learning_correct"]
        if H1p <= r["entropy"] <= H2p
        else r["schema_correct"],
        axis=1
    ).mean()
    print(shift, acc)


with open(OUTPUT_DIR / "gated_thresholds.json", "w") as f:
    json.dump({"H1": float(H1), "H2": float(H2)}, f, indent=2)


print("\n=== Accuracy Comparison ===")
print(f"Schema-only   : {schema_acc:.3f}")
print(f"Learning-only : {learning_acc:.3f}")
print(f"Gated policy  : {gated_acc:.3f}")

# ===============================
# Save results
# ===============================

df.to_csv(OUTPUT_DIR / "gated_policy_results.csv", index=False)

# ===============================
# Plot comparison
# ===============================

plt.figure(figsize=(6,4))
plt.bar(
    ["Schema", "Learning", "Gated"],
    [schema_acc, learning_acc, gated_acc]
)
plt.ylabel("Top-1 Accuracy")
plt.title("Accuracy Comparison Across Policies")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gated_policy_comparison.png")
plt.close()

print("\nDONE: gated policy evaluation completed.")


