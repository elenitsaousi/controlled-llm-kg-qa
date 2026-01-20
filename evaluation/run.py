import argparse

from evaluation.metrics import default_questions_path, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the toy KG QA pipeline."
    )
    parser.add_argument(
        "--questions",
        help="Path to questions.json",
        default=default_questions_path(),
    )
    parser.add_argument(
        "--schema",
        help="Path to schema.json (optional)",
        default=None,
    )
    args = parser.parse_args()

    metrics = evaluate(args.questions, args.schema)
    for key in sorted(metrics.keys()):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
