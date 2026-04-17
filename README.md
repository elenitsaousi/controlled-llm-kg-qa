# controlled-llm-kg-qa (`sparql-version`)

Toy knowledge-graph QA pipeline with:
- schema-constrained SPARQL candidate generation
- schema-based + learning-based ranking
- ambiguity-aware gating analysis
- basic CLI/Streamlit interfaces and visualization scripts

This README documents what you need to run this branch end-to-end.

## 1) Prerequisites

- Python `3.10+` (recommended: `3.11`)
- `pip`
- Optional but recommended: [Ollama](https://ollama.com/) for local LLM generation

## 2) Environment Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn joblib matplotlib networkx streamlit scipy xgboost
```

Optional extras (needed only for some local/custom scripts):

```bash
pip install python-dotenv requests rdflib
```

Optional ForceAtlas2 layout for graph plots (the code already falls back to spring layout if missing):

```bash
pip install fa2-modified
```

## 3) Ollama Setup (for candidate generation)

If you want LLM candidate generation to work locally:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
export OLLAMA_MODEL=qwen2.5:7b-instruct
```

Default model in code is `qwen2.5:7b-instruct`.

## 4) Core Run Commands

### CLI (single question)

```bash
./.venv/bin/python cli.py --question "Which suppliers affect yield?" --show-candidates
```

### CLI (interactive)

```bash
./.venv/bin/python cli.py --show-candidates
```

### Streamlit app

```bash
./.venv/bin/streamlit run app.py
```

### Evaluation on toy questions

Use module mode:

```bash
./.venv/bin/python -m evaluation.run
```

## 5) SPARQL Candidate/Ranking Workflow

### A) Generate SPARQL candidates with LLM

```bash
./.venv/bin/python data/toy_kg/experiments/generate_sparql_candidates_llm.py
```

Outputs candidate files under:
- `data/toy_kg/experiments/sparql_candidates/`

### B) Build feature file for SPARQL candidates

```bash
./.venv/bin/python ranking/build_features_domain_sparql.py
```

Output:
- `ranking/features_domain_sparql.json`

### C) Run ambiguity experiment (schema vs learning vs gated)

```bash
./.venv/bin/python analysis/run_sparql_ambiguity_experiments.py
```

Outputs under:
- `analysis_outputs/sparql_entropy_per_question.json`
- `analysis_outputs/sparql_entropy_stats.json`
- `analysis_outputs/sparql_gated_thresholds.json`
- `analysis_outputs/sparql_gated_policy_results.json`
- `analysis_outputs/sparql_ambiguity_summary.json`

### D) Compare rankers by entropy bin

```bash
./.venv/bin/python results/compare_rankers.py
```

## 6) Visualization

### Toy schema graph (interactive)

```bash
MPLCONFIGDIR=/tmp/mpl_cache ./.venv/bin/python visualization/visualize_toy_schema.py
```

This branch includes compact-layout + larger-label settings for this plot.

## 7) Troubleshooting

- `ModuleNotFoundError: No module named 'evaluation'`
  - Run evaluation as module: `python -m evaluation.run`

- `LLM error: <urlopen error ...>`
  - Ollama is not running or not reachable in the current environment.
  - Start `ollama serve`, pull model, and retry.

- Matplotlib cache/font permission warnings
  - Use `MPLCONFIGDIR=/tmp/mpl_cache` before plot commands.

- `pytest` fails before tests due external plugins
  - Use:
    ```bash
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest -q
    ```

## 8) Notes

- This branch currently has no pinned `requirements.txt`; install commands above are the reference setup.
- Some legacy training scripts under `ranking/ml_learning_ranker/` are placeholders/empty in this branch.
