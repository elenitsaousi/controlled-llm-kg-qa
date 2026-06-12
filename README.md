# True Demand KGQA

Controlled LLM-to-SPARQL question answering over the True Demand knowledge graph.

The system combines graph-backed candidate generation, ML-assisted query selection,
confidence-aware routing, clarification options, Fuseki execution, and WebVOWL
ontology visualization.

## What Is In This Repository

- `app.py` - Streamlit application for natural-language KGQA, clarification, feedback, and graph views.
- `data/infineon/graph.ttl` - full True Demand RDF knowledge graph used for SPARQL execution.
- `data/infineon/schema.json` - schema metadata used by generation, validation, and ranking.
- `data/infineon/true_demand_ontology_extracted.ttl` - readable ontology/schema layer extracted from the full graph.
- `data/infineon/true_demand_webvowl.json` - precomputed WebVOWL export for ontology visualization.
- `pipeline/`, `llm/`, `kg/`, `validation/` - runtime KGQA pipeline components.
- `ranking/` - ranker feature extraction, training scripts, and saved models.
- `evaluation/` - benchmark evaluation, error analysis, confidence routing, and audit scripts.
- `docs/` - detailed setup and evaluation notes.

## Runtime Architecture

```text
User question
  -> Streamlit UI
  -> optional graph-aware guidance / capability resolution
  -> LLM candidate SPARQL generation
  -> validation + semantic/contract features
  -> ML ranking and confidence routing
  -> Fuseki SPARQL execution over the full True Demand KG
  -> answer synthesis + evidence graph / clarification if needed
```

WebVOWL is used only for ontology/schema visualization. The full graph remains in
Fuseki for query execution.

## Main Artifacts

The current graph is survey-grounded. It integrates partner survey responses
across demand, future demand, current demand baselines, inventory, shortage,
vehicle sales, order cancellation, and autonomous-driving related indicators.

Important generated/evaluation artifacts include:

- `results/splits/final1000_within_family/` - train/dev/test benchmark splits.
- `ranking/final1000_wf_train_ranker_data.json` - ranker training data built from generated candidates.
- `ranking/models/final1000_wf_ranker_scope_origin.json` - current logistic ranker used in final tests.
- `results/final1000_wf_test_scope_origin_m010.json` - held-out test results with ML ranking.
- `results/final1000_wf_test_scope_origin_confidence_routing_sorted_v2.json` - confidence-routing report.

Some local or historical artifacts may not be committed if they are too large,
machine-specific, or generated during experiments.

## Required Services

For the full UI, keep three terminals open.

### 1. Fuseki

Start Apache Jena Fuseki with the full graph:

```powershell
cd C:\Users\tsaousieleni\Downloads\apache-jena-fuseki-6.1.0\apache-jena-fuseki-6.1.0

.\fuseki-server.bat --file=C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa\data\infineon\graph.ttl /infineon
```

The SPARQL endpoint is:

```text
http://localhost:3030/infineon/sparql
```

If `java` is not recognized, run Fuseki with the full Java path or add the JDK
`bin` directory to the current PowerShell session.

### 2. WebVOWL

If WebVOWL was already built and has a `deploy` folder:

```powershell
cd C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa\WebVOWL
python -m http.server 8080 -d deploy
```

Open:

```text
http://localhost:8080
```

Load this file in WebVOWL:

```text
C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa\data\infineon\true_demand_webvowl.json
```

### 3. Streamlit

```powershell
cd C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa
python -m streamlit run app.py
```

The app usually opens at:

```text
http://localhost:8501
```

## Key Environment Variables

The app can also be configured through the Developer settings panel.

```env
INFINEON_API_URL=https://gpt4ifx.icp.infineon.com
INFINEON_CHAT_ENDPOINT=/chat/completions
INFINEON_MODEL=gpt-4o
INFINEON_API_KEY=...

FUSEKI_QUERY_URL=http://localhost:3030/infineon/sparql
WEBVOWL_URL=http://localhost:8080
TRUE_DEMAND_ONTOLOGY_PATH=data/infineon/true_demand_ontology_extracted.ttl
TRUE_DEMAND_WEBVOWL_JSON_PATH=data/infineon/true_demand_webvowl.json
```

If `FUSEKI_QUERY_URL` is empty, the app falls back to local RDFLib execution over
`data/infineon/graph.ttl`, which is slower.

## Rebuilding The WebVOWL Ontology

When `graph.ttl` changes, regenerate the schema-level ontology:

```powershell
python visualization\extract_webvowl_ontology.py --graph data\infineon\graph.ttl --out data\infineon\true_demand_ontology_extracted.ttl --rdfxml-out data\infineon\true_demand_ontology_extracted.owl
```

Then convert `true_demand_ontology_extracted.owl` with OWL2VOWL and replace:

```text
data/infineon/true_demand_webvowl.json
```

The extracted ontology intentionally contains only schema-level classes and
relationships. Do not load the raw 1M+ triple graph into WebVOWL unless you want
a noisy instance-level visualization.

## Common Evaluation Commands

Run held-out evaluation:

```powershell
python evaluation\run_infineon_holdout_eval.py --dataset results\splits\final1000_within_family\test.json --k 8 --progress --query-timeout 10 --use-ml-ranking --use-semantic-selection --ml-model ranking\models\final1000_wf_ranker_scope_origin.json --resume --out results\final1000_wf_test_eval.json
```

Apply guarded reranking to an existing result file:

```powershell
python evaluation\apply_ml_ranker_to_results.py --results results\final1000_wf_test_eval.json --model ranking\models\final1000_wf_ranker_scope_origin.json --schema data\infineon\schema.json --out results\final1000_wf_test_scope_origin_m010.json --guarded --min-margin 0.10 --min-score 0.45 --max-rank 4
```

Analyze KGQA results:

```powershell
python evaluation\analyze_infineon_results.py --results results\final1000_wf_test_scope_origin_m010.json --dataset results\splits\final1000_within_family\test.json --out-md results\final1000_wf_test_scope_origin_m010_error_analysis.md
```

Analyze confidence-aware routing:

```powershell
python evaluation\analyze_confidence_routing.py --results results\final1000_wf_test_scope_origin_m010.json --score-key ml_score --sort-by-score --out-json results\final1000_wf_test_scope_origin_confidence_routing_sorted_v2.json --out-md results\final1000_wf_test_scope_origin_confidence_routing_sorted_v2.md
```

Analyze entropy-based ambiguity regimes:

```powershell
python evaluation\analyze_entropy_ambiguity.py --results results\final1000_wf_test_scope_origin_m010.json --dataset results\splits\final1000_within_family\test.json --score-key ml_score --sort-by-score --normalization auto --bucket-mode quantiles --out-json results\final1000_wf_test_scope_origin_entropy_ambiguity.json --out-md results\final1000_wf_test_scope_origin_entropy_ambiguity.md
```

## Current Evaluation Framing

Report these metrics separately:

- forced Top-1 selection accuracy
- Any-Correct candidate recall
- selection failures where the correct candidate was present
- generation failures where no correct candidate was present
- confidence-aware auto-answer accuracy and coverage
- clarification rate and clarification option quality
- empty-result rate
- response-time distribution
- entropy-regime accuracy: low / medium / high candidate-set ambiguity

The final system should not be described only by forced Top-1 accuracy. The
confidence-routing mode is part of the reliability design: high-confidence
questions can be answered automatically, while ambiguous questions are routed to
clarification.

## More Documentation

- `docs/fuseki_windows_setup.md`
- `docs/webvowl_windows_setup.md`
- `docs/final_evaluation_protocol.md`
- `docs/family_schema_routing.md`
- `docs/math_section.md`
- `docs/project_artifact_cleanup.md`
