import json
from pathlib import Path

RAW_PATH = Path("data/toy_kg/raw_llm_output.txt")
OUT_DIR = Path("data/toy_kg/experiments/candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_blocks(text):
    blocks = []

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect question
        if line.startswith("Which "):
            question = line
            queries = []
            i += 1

            while i < len(lines):
                if lines[i].startswith("MATCH"):
                    queries.append(lines[i])

                if len(queries) == 5:
                    blocks.append((question, queries))
                    break

                i += 1
        else:
            i += 1

    return blocks


def get_next_q_index():
    existing = list(OUT_DIR.glob("Q*_candidates.json"))
    if not existing:
        return 1

    ids = []
    for f in existing:
        try:
            ids.append(int(f.stem.split("_")[0][1:]))
        except:
            pass

    return max(ids) + 1


def main():
    with open(RAW_PATH) as f:
        raw_text = f.read()

    blocks = parse_blocks(raw_text)
    start_idx = get_next_q_index()

    print(f"Found {len(blocks)} new questions")
    print(f"Starting from Q{start_idx}")

    for offset, (question, queries) in enumerate(blocks):
        qid = f"Q{start_idx + offset}"

        data = {
            "question_id": qid,
            "question": question,
            "candidates": []
        }

        for j, q in enumerate(queries, start=1):
            data["candidates"].append({
                "id": f"{qid}_C{j}",
                "query": q
            })

        out_path = OUT_DIR / f"{qid}_candidates.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Wrote {len(blocks)} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
