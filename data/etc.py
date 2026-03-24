import csv
from collections import Counter

path = "data/annotations_llm.csv"

counts = Counter()

with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row["label"].strip()
        counts[label] += 1

total = sum(counts.values())

print("Annotation statistics:\n")
for label, count in counts.items():
    print(f"{label}: {count}")

print(f"\nTotal candidates: {total}")