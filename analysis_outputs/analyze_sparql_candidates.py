import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))
    
import json
from pathlib import Path

from kg.sparql_matching import is_relaxed_correct
from validation.syntax import validate_query_syntax
from validation.semantic import validate_query_semantic

CANDIDATES_DIR = Path("data/toy_kg/experiments/sparql_candidates")
QUESTIONS_PATH = Path("data/toy_kg/questions/questions.json")

with open(QUESTIONS_PATH, "r") as f:
    questions = {q["id"]: q["gold_query"] for q in json.load(f)}

correct = 0
valid_wrong = 0
invalid = 0
total = 0

for file in CANDIDATES_DIR.glob("Q*_candidates.json"):
    with open(file, "r") as f:
        data = json.load(f)

    qid = data["question_id"]
    gold = questions[qid]

    for cand in data["candidates"]:
        q = cand["query"]
        total += 1

        syntax_ok = not validate_query_syntax(q)
        semantic_ok = not validate_query_semantic(q)

        if is_relaxed_correct(q, gold):
            correct += 1
        elif syntax_ok and semantic_ok:
            valid_wrong += 1
        else:
            invalid += 1

print("\n=== SPARQL Candidate Analysis ===")
print(f"CORRECT: {correct}")
print(f"VALID WRONG: {valid_wrong}")
print(f"INVALID: {invalid}")
print(f"TOTAL: {total}")