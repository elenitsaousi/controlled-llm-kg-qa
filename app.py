import os

import streamlit as st

from kg.schema import load_default_schema, load_schema
from llm.candidate_generation import generate_candidate_prompt
from llm.ollama_client import OllamaClient
from pipeline.qa import answer_question


def _load_schema(schema_path: str):
    cleaned = schema_path.strip()
    if not cleaned:
        return load_default_schema()
    if not os.path.exists(cleaned):
        st.warning(
            "Schema path not found. Falling back to data/toy_kg/schema.json."
        )
        return load_default_schema()
    return load_schema(cleaned)


st.set_page_config(page_title="Toy KG QA", layout="centered")

st.title("Toy KG Question Answering")
st.caption("Type a natural language question and view a placeholder answer.")

use_custom_schema = st.checkbox("Use custom schema path", value=False)
schema_path = st.text_input(
    "Schema path",
    value="",
    help="Provide a path to schema.json",
    disabled=not use_custom_schema,
)
show_prompt = st.checkbox("Show candidate generation prompt", value=False)
show_candidates = st.checkbox("Show candidates", value=True)
default_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
model_name = st.text_input("Ollama model", value=default_model)

question = st.text_area(
    "Your question",
    placeholder="e.g., Which suppliers affect the production yield of product X?",
    height=100,
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        schema = _load_schema(schema_path if use_custom_schema else "")
        if show_prompt:
            prompt = generate_candidate_prompt(question, schema, k=5)
            st.subheader("Candidate Generation Prompt")
            st.code(prompt, language="text")
        client = OllamaClient(model=model_name.strip() or default_model)
        result = answer_question(question, schema, llm_client=client)
        st.subheader("Answer")
        st.write(result["answer"])
        metadata = result.get("metadata", {})
        if metadata.get("error"):
            st.error(f"LLM error: {metadata['error']}")
        if show_candidates:
            st.subheader("Candidates")
            candidates = result.get("candidates", [])
            if not candidates:
                st.write("No candidates returned.")
            else:
                for item in candidates:
                    st.code(item.get("query", ""), language="text")
