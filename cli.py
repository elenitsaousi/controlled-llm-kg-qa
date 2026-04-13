import argparse
from typing import Optional

from kg.schema import load_default_schema, load_schema
from llm.candidate_generation import generate_candidate_prompt
from llm.client import OllamaClient
from pipeline.qa import answer_question


def _load_schema(schema_path: Optional[str]):
    if schema_path:
        return load_schema(schema_path)
    return load_default_schema()


def run_once(
    question: str,
    schema_path: Optional[str],
    show_prompt: bool,
    show_candidates: bool,
    model: Optional[str],
) -> None:
    schema = _load_schema(schema_path)
    if show_prompt:
        prompt = generate_candidate_prompt(question, schema, k=5)
        print("\n--- Candidate Generation Prompt ---")
        print(prompt)
        print("--- End Prompt ---\n")
    client = OllamaClient(model=model)
    result = answer_question(question, schema, llm_client=client)
    metadata = result.get("metadata", {})
    if metadata.get("error"):
        print(f"\nLLM error: {metadata['error']}")
    if show_candidates:
        print("\nCandidates:")
        for item in result.get("candidates", []):
            print(f"- ({item.get('source')}) {item.get('query')}")
    if result.get("selected_query"):
        print("\nSelected Query:")
        print(result["selected_query"])
    print(result["answer"])


def run_interactive(
    schema_path: Optional[str],
    show_prompt: bool,
    show_candidates: bool,
    model: Optional[str],
) -> None:
    print("Enter a natural language question (blank line to exit).")
    while True:
        question = input("> ").strip()
        if not question:
            break
        run_once(question, schema_path, show_prompt, show_candidates, model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal CLI for NL questions over the toy KG."
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Natural language question to answer.",
    )
    parser.add_argument(
        "--schema",
        help="Path to schema.json (defaults to data/toy_kg/schema.json).",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the candidate generation prompt.",
    )
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Print the generated candidate queries.",
    )
    parser.add_argument(
        "--model",
        help="Ollama model name (default: env OLLAMA_MODEL or llama3.1:8b).",
        default=None,
    )
    args = parser.parse_args()

    if args.question:
        run_once(
            args.question,
            args.schema,
            args.show_prompt,
            args.show_candidates,
            args.model,
        )
        return
    run_interactive(
        args.schema, args.show_prompt, args.show_candidates, args.model
    )


if __name__ == "__main__":
    main()
