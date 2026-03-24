import json
import sys
import random
from pathlib import Path
from typing import List, Tuple

BASE = Path(__file__).resolve().parents[3]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from kg.schema import load_default_schema
from kg.sparql_matching import parse_sparql, is_relaxed_correct
from llm.candidate_generation import generate_candidates
from llm.ollama_client import OllamaClient, OllamaClientError
from validation.semantic import validate_query_semantic
from validation.syntax import validate_query_syntax

QUESTIONS_PATH = BASE / "data" / "toy_kg" / "questions" / "questions.json"
OUT_DIR = BASE / "data" / "toy_kg" / "experiments" / "sparql_candidates"

K = 5
MAX_ROUNDS = 3
POOL_TARGET = 15


def _load_questions() -> List[dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return list(json.load(f))


def _is_basic_sparql(query: str) -> bool:
    q = query.strip()
    if not q:
        return False
    upper = q.upper()
    return upper.startswith("SELECT") and "WHERE" in upper and "{" in q and "}" in q


def _is_syntax_ok(query: str) -> bool:
    return not validate_query_syntax(query)


def _is_semantic_ok(query: str) -> bool:
    return not validate_query_semantic(query)


def _collect_candidates(question: str, schema, client: OllamaClient) -> List[str]:
    seen = set()
    collected: List[str] = []

    rounds = 0
    while rounds < MAX_ROUNDS and len(collected) < POOL_TARGET:
        print(f"[DEBUG] {question[:30]}... round={rounds}, collected={len(collected)}")

        result = generate_candidates(
            question=question,
            schema=schema,
            k=5,
            llm_client=client,
        )

        error = result.get("metadata", {}).get("error")
        if error:
            raise OllamaClientError(error)

        for cand in result.get("candidates", []):
            query = str(cand.get("query", "")).strip()

            if not query or query in seen:
                continue
            if not _is_basic_sparql(query):
                continue
            # if not _is_syntax_ok(query):
            #     continue
            # if not _is_semantic_ok(query):
            #     continue

            collected.append(query)
            seen.add(query)

            if len(collected) >= POOL_TARGET:
                break

        rounds += 1

    return collected


def _signature(query: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], int]:
    types, triples = parse_sparql(query)

    classes = []
    for cls_set in types.values():
        classes.extend(list(cls_set))

    preds = [p for _, p, _ in triples]

    return (
        tuple(sorted(set(classes))),
        tuple(sorted(set(preds))),
        len(triples),
    )


def main() -> None:
    schema = load_default_schema()
    questions = _load_questions()

    client = OllamaClient(temperature=0.5, max_tokens=512)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for item in questions:
        qid = item["id"]
        question_text = item["question"]
        gold_query = item["gold_query"]

        try:
            pool = _collect_candidates(question_text, schema, client)
        except OllamaClientError as exc:
            raise RuntimeError(f"Ollama error for {qid}: {exc}") from exc

        # 🔍 DEBUG
        correct_candidates = [
            q for q in pool if is_relaxed_correct(q, gold_query)
        ]

        if len(correct_candidates) == 0:
            correct_candidates = [gold_query]

        # --- NEW: split partial / incorrect ---
        partial_candidates = []
        incorrect_candidates = []

        gold_types, gold_triples = parse_sparql(gold_query)
        gold_preds = {p for _, p, _ in gold_triples}

        for q in pool:
            if is_relaxed_correct(q, gold_query):
                continue

            _, triples = parse_sparql(q)
            preds = {p for _, p, _ in triples}

            overlap = len(preds & gold_preds)

            schema_valid = len(preds) > 0  # ήδη περνάς syntax/semantic

            if overlap >= len(gold_preds) - 1:
                partial_candidates.append(q)
            elif overlap > 0:
                incorrect_candidates.append(q)
            else:
                incorrect_candidates.append(q)

        # --- SELECTION ---
        selected = []
        used = set()

        # 1 correct
        correct_query = random.choice(correct_candidates)
        selected.append(correct_query)
        used.add(_signature(correct_query))

        # --- HARD NEGATIVE (IMPORTANT) ---
        hard_negatives = []

        for q in pool:
            if is_relaxed_correct(q, gold_query):
                continue

            _, triples = parse_sparql(q)
            preds = {p for _, p, _ in triples}

            if preds == gold_preds:   # same predicates → tricky
                hard_negatives.append(q)

        random.shuffle(hard_negatives)

        for q in hard_negatives:
            sig = _signature(q)
            if sig not in used:
                selected.append(q)
                used.add(sig)
                break  # μόνο 1

        # --- partial ---
        random.shuffle(partial_candidates)
        for q in partial_candidates:
            sig = _signature(q)
            if sig not in used:
                selected.append(q)
                used.add(sig)
            if len(selected) == 3:
                break

        # --- incorrect ---
        random.shuffle(incorrect_candidates)
        for q in incorrect_candidates:
            sig = _signature(q)
            if sig not in used:
                selected.append(q)
                used.add(sig)
            if len(selected) == 5:
                break

        # fallback αν δεν φτάνουν
        if len(selected) < K:
            for q in pool:
                sig = _signature(q)
                if sig not in used:
                    selected.append(q)
                    used.add(sig)
                if len(selected) == K:
                    break

        if len(selected) < K:
            raise RuntimeError(f"Not enough candidates for {qid}")

        data = {
            "question_id": qid,
            "question": question_text,
            "candidates": [
                {"id": f"{qid}_C{i+1}", "query": q}
                for i, q in enumerate(selected)
            ],
        }

        out_path = OUT_DIR / f"{qid}_candidates.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    print(f"Wrote {len(questions)} SPARQL candidate files to {OUT_DIR}")


if __name__ == "__main__":
    main()