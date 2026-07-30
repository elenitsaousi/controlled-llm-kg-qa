# Final 1000-question evaluation using the repaired/validated KG gold benchmark.
# Run from:
#   C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa
#
# Assumptions:
# - Fuseki is running at http://localhost:3030/infineon/sparql
# - Infineon GPT token/session is valid
# - DigitalReference.ttl exists at C:\Users\tsaousieleni\Downloads\DigitalReference.ttl

$ErrorActionPreference = "Stop"

$env:INFINEON_REQUEST_TIMEOUT_SEC = "90"
$env:INFINEON_AUTH_TIMEOUT_SEC = "30"
$env:INFINEON_MAX_RETRIES = "1"
$env:INFINEON_RETRY_BACKOFF_SEC = "1"
$env:FUSEKI_QUERY_URL = "http://localhost:3030/infineon/sparql"
$env:INFINEON_ENABLE_LLM_CACHE = "1"

# ---------------------------------------------------------------------------
# 0. Validate the repaired gold benchmark against Fuseki and semantic checks.
# ---------------------------------------------------------------------------
python evaluation\semantic_validate_gold_queries.py `
  --dataset evaluation\question_sets\true_demand_final_1000_gold_repaired.json `
  --audit results\true_demand_final_1000_gold_semantic_audit_working.csv `
  --execute `
  --fuseki-query-url http://localhost:3030/infineon/sparql `
  --out-csv results\gold_semantic_validation_repaired_auto.csv `
  --out-review-csv results\gold_semantic_manual_review_repaired_only.csv `
  --out-json results\gold_semantic_validation_repaired_auto.json `
  --out-md results\gold_semantic_validation_repaired_summary.md

# ---------------------------------------------------------------------------
# 1. LLM-only candidate generation and raw baseline selection.
#    No deterministic routing. This is resumable and is the expensive step.
# ---------------------------------------------------------------------------
python evaluation\run_infineon_holdout_eval.py `
  --dataset evaluation\question_sets\true_demand_final_1000_gold_repaired.json `
  --graph data\infineon\graph.ttl `
  --schema data\infineon\schema.json `
  --out results\final1000_repaired_llm_raw_candidates.json `
  --k 8 `
  --llm infineon `
  --temperature 0.2 `
  --query-timeout 10 `
  --resume `
  --progress

python evaluation\analyze_infineon_results.py `
  --results results\final1000_repaired_llm_raw_candidates.json `
  --dataset evaluation\question_sets\true_demand_final_1000_gold_repaired.json `
  --schema data\infineon\schema.json `
  --out-json results\final1000_repaired_llm_raw_analysis.json `
  --out-md results\final1000_repaired_llm_raw_analysis.md

# ---------------------------------------------------------------------------
# 2. Offline schema/semantic selector on the same generated candidates.
#    No new LLM calls.
# ---------------------------------------------------------------------------
python evaluation\apply_selection_to_results.py `
  --results results\final1000_repaired_llm_raw_candidates.json `
  --out results\final1000_repaired_schema_semantic_selection.json

python evaluation\analyze_infineon_results.py `
  --results results\final1000_repaired_schema_semantic_selection.json `
  --dataset evaluation\question_sets\true_demand_final_1000_gold_repaired.json `
  --schema data\infineon\schema.json `
  --out-json results\final1000_repaired_schema_semantic_analysis.json `
  --out-md results\final1000_repaired_schema_semantic_analysis.md

# ---------------------------------------------------------------------------
# 3. Offline guarded ML reranker on the same candidates.
#    No new LLM calls.
# ---------------------------------------------------------------------------
python evaluation\apply_ml_ranker_to_results.py `
  --results results\final1000_repaired_llm_raw_candidates.json `
  --model ranking\models\final1000_wf_ranker_current.json `
  --schema data\infineon\schema.json `
  --out results\final1000_repaired_guarded_ml_selection.json `
  --guarded `
  --min-margin 0.10 `
  --min-score 0.45 `
  --max-rank 4 `
  --structured-guard

python evaluation\analyze_infineon_results.py `
  --results results\final1000_repaired_guarded_ml_selection.json `
  --dataset evaluation\question_sets\true_demand_final_1000_gold_repaired.json `
  --schema data\infineon\schema.json `
  --out-json results\final1000_repaired_guarded_ml_analysis.json `
  --out-md results\final1000_repaired_guarded_ml_analysis.md

# ---------------------------------------------------------------------------
# 4. Build the mixed full-system benchmark from repaired KG + DR + advisory.
# ---------------------------------------------------------------------------
python evaluation\build_dr_ontology_benchmark.py `
  --dr-ontology "C:\Users\tsaousieleni\Downloads\DigitalReference.ttl" `
  --limit 300 `
  --out evaluation\question_sets\dr_ontology_benchmark_current.json

python evaluation\build_full_system_question_set.py `
  --kg-questions evaluation\question_sets\true_demand_final_1000_gold_repaired.json `
  --dr-questions evaluation\question_sets\dr_ontology_benchmark_current.json `
  --target-total 1000 `
  --kg-count 800 `
  --ontology-count 150 `
  --advisory-count 50 `
  --out evaluation\question_sets\true_demand_full_system_1000_repaired.json

# ---------------------------------------------------------------------------
# 5. Full system: deterministic first, LLM only when needed.
#    This is resumable through the JSONL log.
# ---------------------------------------------------------------------------
python evaluation\run_efficiency_question_set.py `
  --questions evaluation\question_sets\true_demand_full_system_1000_repaired.json `
  --out-log logs\kgqa_system_accuracy_1000_repaired.jsonl `
  --graph data\infineon\graph.ttl `
  --schema data\infineon\schema.json `
  --fuseki-query-url http://localhost:3030/infineon/sparql `
  --call-llm `
  --enable-llm-cache `
  --resume

python evaluation\analyze_system_efficiency.py `
  --log logs\kgqa_system_accuracy_1000_repaired.jsonl `
  --cost-per-call 0.20 `
  --out-json results\kgqa_efficiency_1000_repaired_report.json `
  --out-md results\kgqa_efficiency_1000_repaired_report.md

python evaluation\build_system_accuracy_audit.py `
  --log logs\kgqa_system_accuracy_1000_repaired.jsonl `
  --questions evaluation\question_sets\true_demand_full_system_1000_repaired.json `
  --out-csv results\kgqa_system_accuracy_audit_1000_repaired.csv

# ---------------------------------------------------------------------------
# 6. Label and summarize final answer-level accuracy.
#    If a previous labeled repaired audit exists, unchanged rows reuse those
#    labels; changed KG rows are checked against the repaired gold result
#    signature.
# ---------------------------------------------------------------------------
if (Test-Path results\kgqa_system_accuracy_audit_1000_repaired_labeled.csv) {
  python evaluation\label_full_system_audit.py `
    --audit-csv results\kgqa_system_accuracy_audit_1000_repaired.csv `
    --gold evaluation\question_sets\true_demand_final_1000_gold_repaired.json `
    --graph data\infineon\graph.ttl `
    --previous-labeled-csv results\kgqa_system_accuracy_audit_1000_repaired_labeled.csv `
    --out-csv results\kgqa_system_accuracy_audit_1000_repaired_labeled.csv `
    --out-json results\kgqa_system_accuracy_audit_1000_repaired_labeled_summary.json `
    --out-md results\kgqa_system_accuracy_audit_1000_repaired_labeled_summary.md
} else {
  python evaluation\label_full_system_audit.py `
    --audit-csv results\kgqa_system_accuracy_audit_1000_repaired.csv `
    --gold evaluation\question_sets\true_demand_final_1000_gold_repaired.json `
    --graph data\infineon\graph.ttl `
    --out-csv results\kgqa_system_accuracy_audit_1000_repaired_labeled.csv `
    --out-json results\kgqa_system_accuracy_audit_1000_repaired_labeled_summary.json `
    --out-md results\kgqa_system_accuracy_audit_1000_repaired_labeled_summary.md
}
