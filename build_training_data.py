# build_training_data.py
import json
from dotenv import load_dotenv
load_dotenv(".env")
from rdflib import Graph
from kg.schema import load_schema
from ranking.feature_extraction import extract_features
from llm.candidate_generation import generate_candidates


def run_multiple_generations(
    dataset_path,
    graph_path,
    schema_path,
    output_path,
    k=5,
    n_runs=3
):
    """Generate multiple sets of candidates for better training data."""

    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Load schema dict for feature extraction
    with open(schema_path) as f:
        schema_dict = json.load(f)

    # Load graph
    g = Graph()
    g.parse(graph_path, format="turtle")
    print(f"Graph loaded: {len(g)} triples")

    # Load schema object for candidate generation
    schema = load_schema(schema_path)

    PREFIX = (
        "PREFIX survey: "
        "<http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>\n"
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
    )

    training_data = {}
    total_correct = 0
    total_candidates = 0

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        gold_query = item["query"]

        print(f"\nProcessing: {qid}")

        # Get gold results
        full_gold = PREFIX + gold_query if "PREFIX" not in gold_query else gold_query
        try:
            gold_rows = set(
                tuple(str(v) for v in row)
                for row in g.query(full_gold)
            )
        except Exception as e:
            print(f"  Gold query failed: {e}")
            continue

        training_data[qid] = []
        seen_queries = set()

        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...")
            try:
                result = generate_candidates(question, schema, k=k)
                candidates = result.get("candidates", [])
            except Exception as e:
                print(f"  Generation failed: {e}")
                continue

            for c in candidates:
                query = c.get("query", "")

                # Skip duplicates
                if query in seen_queries:
                    continue
                seen_queries.add(query)

                # Check correctness by execution
                full_query = PREFIX + query if "PREFIX" not in query else query
                try:
                    rows = set(
                        tuple(str(v) for v in row)
                        for row in g.query(full_query)
                    )
                    is_correct = 1 if rows == gold_rows and len(rows) > 0 else 0
                    is_valid = 1
                except Exception:
                    is_correct = 0
                    is_valid = 0

                # Extract features
                try:
                    features = extract_features(question, query, schema_dict)
                except Exception:
                    features = {}

                total_candidates += 1
                total_correct += is_correct

                training_data[qid].append({
                    "query_id": f"{qid}_R{run}_C{len(training_data[qid])}",
                    "question": question,
                    "query": query,
                    "is_correct": is_correct,
                    "is_valid": is_valid,
                    "features": features
                })

        correct_in_q = sum(
            1 for c in training_data[qid] if c["is_correct"] == 1
        )
        print(f"  Candidates: {len(training_data[qid])}, Correct: {correct_in_q}")

    # Save
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"Questions: {len(training_data)}")
    print(f"Total candidates: {total_candidates}")
    if total_candidates > 0:
        print(f"Correct: {total_correct} ({total_correct/total_candidates*100:.1f}%)")
    else:
        print("No candidates generated - check token!")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    run_multiple_generations(
        dataset_path="data/infineon/infineon_dataset_30.json",
        graph_path="data/infineon/graph.ttl",
        schema_path="data/infineon/schema.json",
        output_path="ranking/infineon_training_data.json",
        k=5,
        n_runs=3
    )