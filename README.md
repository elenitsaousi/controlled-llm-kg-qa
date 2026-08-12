# True Demand KGQA

Controlled LLM-to-SPARQL question answering over the True Demand knowledge graph.

The system combines graph-backed candidate generation, ML-assisted query selection,
confidence-aware routing, clarification options, deterministic ontology/advisory
routing, Fuseki execution, evidence visualization, and user testing audit logs.

## What Is In This Repository

- `app.py` - main Streamlit application for natural-language KGQA, clarification, feedback, DR ontology browsing, audit logging, and graph views.
- `True Demand-lovable/` - archived/prototype React frontend. The current managed-machine UI is the Python-only Streamlit app.
- `api/` - optional FastAPI adapter for the archived React frontend.
- `data/infineon/graph.ttl` - full True Demand RDF knowledge graph used for SPARQL execution.
- `data/infineon/schema.json` - schema metadata used by generation, validation, and ranking.
- `data/infineon/true_demand_ontology_extracted.ttl` - readable ontology/schema layer extracted from the full graph.
- `pipeline/`, `llm/`, `kg/`, `validation/` - runtime KGQA pipeline components.
- `ranking/` - ranker feature extraction, training scripts, and saved models.
- `evaluation/` - benchmark evaluation, error analysis, confidence routing, and audit scripts.
- `docs/` - detailed setup and evaluation notes.

## Runtime Architecture

```text
User question
  -> Streamlit UI
  -> request routing: KG analytics, DR definition, advisory, unsupported, or fallback
  -> deterministic route when one checked graph/ontology/advisory path is available
  -> LLM candidate SPARQL generation only for unresolved or genuinely ambiguous questions
  -> validation + semantic/contract features
  -> ML ranking and confidence routing
  -> Fuseki SPARQL execution over the full True Demand KG
  -> answer synthesis + evidence graph / clarification or controlled no-answer if needed
  -> user audit logging
```

The full graph remains in Fuseki for query execution. Ontology/model questions
are routed to the Digital Reference ontology when `TRUE_DEMAND_DR_ONTOLOGY_PATH`
or `DR_ONTOLOGY_PATH` is configured.

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

## Recommended Shared Deployment For Internal Testing

For Philipp, Hans, and other testers, do not ask each user to run the repository
locally. The recommended setup is one shared Streamlit deployment managed by the
deployment owner.

The shared deployment should provide:

- one Streamlit app URL for all users;
- one shared Fuseki endpoint loaded with `data/infineon/graph.ttl`;
- `DigitalReference.ttl` available on the server and configured through an
  environment variable;
- persistent storage for the `logs/` folder;
- LLM credentials configured server-side if fallback LLM answering is enabled;
- developer mode disabled for normal users.

The app writes every submitted question to:

- `logs/kgqa_user_audit.jsonl`
- `logs/kgqa_user_audit.sqlite3`

These logs include the question, route, confidence, selected/top query, answer,
row count, timing, and metadata. If everyone uses the same deployed app, the
testing data is collected centrally. If each user runs a local copy, the logs
stay on each user's machine.

Minimal server environment:

```env
FUSEKI_QUERY_URL=http://<server>:3030/infineon/sparql
TRUE_DEMAND_DR_ONTOLOGY_PATH=/path/to/DigitalReference.ttl
LLM_BACKEND=litellm
LITELLM_BASE_URL=https://<litellm-gateway>
LITELLM_CHAT_ENDPOINT=/chat/completions
LITELLM_MODEL=<approved-model-name>
LITELLM_API_KEY=<server-side-key>
INFINEON_ENABLE_LLM_CACHE=1
TRUE_DEMAND_ENABLE_DEVELOPER_MODE=0
```

For the Infineon LiteLLM deployment, the application also accepts the aliases
used by the platform team:

```env
LLM_BACKEND=litellm
BASE_URL=https://litellm.icp.infineon.com
LITE_LLM_TOKEN=<server-side-key>
LITELLM_MODEL=<approved-model-name>
LITELLM_CHAT_ENDPOINT=/chat/completions
```

Use `TRUE_DEMAND_ENABLE_DEVELOPER_MODE=1` only for debugging by the deployment
owner.

For deployed Streamlit, bind the server to all interfaces and make sure the
reverse proxy supports Streamlit websocket traffic:

```bash
python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

If Fuseki runs in a different container or service, `FUSEKI_QUERY_URL` must use
that internal service URL, not `localhost`.

## Required Services For Local Development

For local development, keep Fuseki and Streamlit in separate terminals.

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

### 2. Python-only Streamlit UI

This is the recommended UI on managed Windows machines where Node.js/npm is not
available. It uses the same KGQA runtime and keeps the direct routing, ML
selection, clarification, evidence graph, examples, guided builder, DR ontology
browser, audit logging, and dashboard in one Python app.

From the repository root:

```powershell
python -m pip install -r requirements-ui.txt
$env:FUSEKI_QUERY_URL="http://localhost:3030/infineon/sparql"
$env:TRUE_DEMAND_DR_ONTOLOGY_PATH="C:\path\to\DigitalReference.ttl"
$env:INFINEON_ENABLE_LLM_CACHE="1"
python -m streamlit run app.py --server.port 8501
```

Open `http://localhost:8501`.

If PowerShell blocks local scripts, use:

```powershell
.\run_streamlit_ui.bat
```

or:

```powershell
.\run_streamlit_ui.ps1
```

### 3. FastAPI adapter for the archived React frontend

```powershell
python -m pip install -r requirements-api.txt
$env:FUSEKI_QUERY_URL="http://localhost:3030/infineon/sparql"
python -m uvicorn api.main:app --reload --port 8000
```

The API health endpoint is `http://localhost:8000/api/health`.

### 4. React frontend (archived/optional)

The React/Lovable UI is optional and requires Node.js/npm.

```powershell
cd "C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa\True Demand-lovable"
npm install
npm run dev
```

Open the URL printed by Vite, currently `http://localhost:8080`. The frontend uses
`http://localhost:8000` by default and can be configured with
`VITE_API_BASE_URL` or from its Settings page.

The Streamlit app is the current recommended interface. The React integration
does not replace or delete `app.py`; both interfaces call the same KGQA pipeline
and use the same Fuseki dataset.

## Key Environment Variables

The app can also be configured through the Developer settings panel.

```env
INFINEON_API_URL=https://gpt4ifx.icp.infineon.com
INFINEON_CHAT_ENDPOINT=/chat/completions
INFINEON_MODEL=gpt-4o
INFINEON_API_KEY=...

FUSEKI_QUERY_URL=http://localhost:3030/infineon/sparql
TRUE_DEMAND_ONTOLOGY_PATH=data/infineon/true_demand_ontology_extracted.ttl
TRUE_DEMAND_DR_ONTOLOGY_PATH=/path/to/DigitalReference.ttl
DR_ONTOLOGY_PATH=/path/to/DigitalReference.ttl
TRUE_DEMAND_ENABLE_DEVELOPER_MODE=0

# Optional cost/latency controls used by the Streamlit demo
INFINEON_ENABLE_LLM_CACHE=1
INFINEON_LLM_CACHE_DIR=.cache/kgqa_llm
```

If `FUSEKI_QUERY_URL` is empty, the app falls back to local RDFLib execution over
`data/infineon/graph.ttl`, which is slower.

The Streamlit app can skip the LLM entirely when the capability inventory resolves
one graph-supported interpretation and that query returns rows. For repeated
free-text questions, `INFINEON_ENABLE_LLM_CACHE=1` reuses candidate-generation
outputs only when the prompt/model/settings hash is identical. This reduces demo
cost and latency without changing the ranking logic for new or ambiguous
questions.

`TRUE_DEMAND_DR_ONTOLOGY_PATH` or `DR_ONTOLOGY_PATH` is optional. If it points to the Digital Reference
ontology, definition-style questions such as "What is Demand?" or "What is
Product?" are answered directly from DR labels, comments, definitions, domains,
and ranges before the LLM is called. Analytical questions such as demand trends,
monthly changes, or grouped totals still use the True Demand KGQA pipeline.

The API and Streamlit app also include a deterministic graph-grounded advisory
layer for a small set of insight questions, for example "which region should be
monitored more closely?" or "which vehicle type shows the strongest future
demand signal?". These are mapped to fixed SPARQL templates and summarized as
analytical signals, not autonomous business recommendations.

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

Classify remaining selection failures into action buckets:

```powershell
python evaluation\analyze_remaining_failure_actions.py --results results\final1000_wf_test_scope_origin_m010.json --out-json results\final1000_wf_test_remaining_failure_actions.json --out-md results\final1000_wf_test_remaining_failure_actions.md
```

Analyze Streamlit runtime efficiency and estimated LLM cost:

```powershell
python evaluation\analyze_system_efficiency.py --log logs\kgqa_sessions.jsonl --cost-per-call 0.20 --out-json results\system_efficiency_report.json --out-md results\system_efficiency_report.md
```

The UI first attempts direct graph-supported capability routing for common
answerable families before calling the LLM: current/regional demand,
future demand, vehicle-sales time breakdowns, shortage counts, autonomous
driving averages, inventory distributions, and order-cancellation summaries.
If no exact graph-supported capability path survives, the system falls back to
LLM candidates, ML/semantic reranking, execution-aware selection checks, and
clarification for genuinely competing interpretations.

Build and run the controlled 500-question efficiency set. The default run is a
cost estimate: graph-supported direct routes are executed, while unresolved
questions are counted as one LLM call without actually spending those calls.

```powershell
python evaluation\build_efficiency_question_set.py --out evaluation\question_sets\true_demand_efficiency_500.json --target 500
python evaluation\run_efficiency_question_set.py --questions evaluation\question_sets\true_demand_efficiency_500.json --out-log logs\kgqa_efficiency_500.jsonl --fuseki-query-url http://localhost:3030/infineon/sparql
python evaluation\analyze_system_efficiency.py --log logs\kgqa_efficiency_500.jsonl --cost-per-call 0.20 --out-json results\kgqa_efficiency_500_report.json --out-md results\kgqa_efficiency_500_report.md
```

For a smaller real LLM run, add `--call-llm --limit 50`. A full 500-question
LLM run can cost up to `500 * €0.20 = €100` before any routing savings.

Build a manual system-accuracy audit sheet from the same log. Fill the
`correctness` column with `correct`, `incorrect`, or `unclear`, then summarize
the filled sheet. This is the engineering system-level accuracy view and should
be reported separately from LLM-needed selection accuracy.

```powershell
python evaluation\build_system_accuracy_audit.py --log logs\kgqa_efficiency_500.jsonl --questions evaluation\question_sets\true_demand_efficiency_500.json --out-csv results\kgqa_system_accuracy_audit_500.csv
python evaluation\build_system_accuracy_audit.py --labeled-csv results\kgqa_system_accuracy_audit_500.csv --out-json results\kgqa_system_accuracy_audit_500.json --out-md results\kgqa_system_accuracy_audit_500.md --unclear-as-incorrect
```

Analyze entropy-based ambiguity regimes:

```powershell
python evaluation\analyze_entropy_ambiguity.py --results results\final1000_wf_test_scope_origin_m010.json --dataset results\splits\final1000_within_family\test.json --score-key ml_score --sort-by-score --normalization auto --bucket-mode quantiles --out-json results\final1000_wf_test_scope_origin_entropy_ambiguity.json --out-md results\final1000_wf_test_scope_origin_entropy_ambiguity.md
```

Compare baseline vs ML selection within entropy regimes:

```powershell
python evaluation\compare_entropy_regime_selection.py --baseline-results results\final1000_wf_test_eval_no_ml.json --ml-results results\final1000_wf_test_scope_origin_m010.json --dataset results\splits\final1000_within_family\test.json --entropy-source ml --score-key ml_score --sort-by-score --normalization softmax --temperature 0.10 --bucket-mode quantiles --out-json results\final1000_wf_test_entropy_regime_baseline_vs_ml.json --out-md results\final1000_wf_test_entropy_regime_baseline_vs_ml.md
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
