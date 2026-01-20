import json

def score(features):
    return (
        2.0 * features["entity_coverage"]
        + 1.5 * features["relation_coverage"]
        + 1.0 * features["expected_intermediate_coverage"]
        - 1.0 * features["unexpected_label_ratio"]
        - 0.01 * features["rel_count"]
    )

if __name__ == "__main__":
    with open("ranking/features_domain.json") as f:
        data = json.load(f)

    for qid, items in data.items():
        ranked = sorted(
            items,
            key=lambda x: score(x["features"]),
            reverse=True
        )
        print(f"{qid} best:", ranked[0]["query_id"])
