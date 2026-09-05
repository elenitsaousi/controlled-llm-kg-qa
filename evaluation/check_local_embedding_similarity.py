import argparse

from ranking.local_embedding_similarity import local_embedding_cosine, local_embedding_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Check optional local embedding similarity setup.")
    parser.add_argument(
        "--question",
        default="Show total demand by region.",
        help="Question text to embed.",
    )
    parser.add_argument(
        "--description",
        default="metric total demand aggregation sum dimensions region",
        help="Candidate-query description text to embed.",
    )
    args = parser.parse_args()

    status = local_embedding_status()
    score = local_embedding_cosine(args.question, args.description)

    print("===== LOCAL EMBEDDING SIMILARITY CHECK =====")
    print(f"Enabled: {status['enabled']}")
    print(f"Model: {status['model'] or '(not set)'}")
    print(f"Model path exists: {status['model_path_exists']}")
    print(f"Model loaded: {status['loaded']}")
    print(f"Similarity: {score:.4f}")
    if not status["loaded"]:
        print("")
        print("Local embeddings are inactive. Set:")
        print("  KGQA_LOCAL_EMBEDDINGS=1")
        print("  KGQA_LOCAL_EMBEDDING_MODEL=<local sentence-transformers model folder>")


if __name__ == "__main__":
    main()

