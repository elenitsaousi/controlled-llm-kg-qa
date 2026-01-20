import pandas as pd
import random

# Load data
df = pd.read_csv("data/annotations_llm.csv")
#df = pd.read_csv("data/annotations_llm.csv")

# Clean columns and labels (CRITICAL FIX)
df.columns = df.columns.str.strip()
df["label"] = df["label"].str.strip().str.replace('"', '', regex=False)

# Sanity check (optional)
print(df.groupby("question_id")["label"].value_counts())

# Random baseline experiment
runs = 100
accs = []

num_questions = df["question_id"].nunique()

for _ in range(runs):
    correct = 0
    for qid, group in df.groupby("question_id"):
        chosen = random.choice(group.to_dict("records"))
        if chosen["label"] == "CORRECT":
            correct += 1
    accs.append(correct / num_questions)

print(f"Random Top-1 Accuracy (avg over {runs} runs): {sum(accs) / runs:.2f}")
