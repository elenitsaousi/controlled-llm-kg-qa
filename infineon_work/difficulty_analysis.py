import json
from infineon_work.classify import classify_difficulty

def run_difficulty_analysis():
    # Load evaluation results
    with open("results/infineon_eval.json") as f:
        data = json.load(f)

    # Load dataset (gold queries)
    with open("data/infineon/infineon_dataset_30.json") as f:
        dataset = {d["id"]: d for d in json.load(f)}

    stats = {
        "easy": {"total": 0, "correct": 0},
        "medium": {"total": 0, "correct": 0},
        "hard": {"total": 0, "correct": 0},
    }

    for d in data["details"]:
        qid = d["id"]
        gold_query = dataset[qid]["query"]

        difficulty = classify_difficulty(gold_query)

        stats[difficulty]["total"] += 1
        if d["top1_correct"]:
            stats[difficulty]["correct"] += 1

    print("\n===== DIFFICULTY ANALYSIS =====")
    for k, v in stats.items():
        acc = v["correct"] / v["total"] if v["total"] else 0
        print(f"{k}: {acc:.2%} ({v['correct']}/{v['total']})")


if __name__ == "__main__":
    run_difficulty_analysis()