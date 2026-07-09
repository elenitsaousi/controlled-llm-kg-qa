# Final 1000-question KGQA evaluation commands.
# Run from:
#   C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa
#
# Assumptions:
# - Fuseki is running at http://localhost:3030/infineon/sparql
# - Infineon GPT token/session is valid
# - Cost estimate is 0.20 EUR per LLM call

$ErrorActionPreference = "Stop"

# Keep long unattended runs from blocking for many minutes on one LLM request.
# If a request exceeds this, the question is marked as a generation failure and
# the evaluation moves on. Auth failures still stop the final run.
$env:INFINEON_REQUEST_TIMEOUT_SEC = "90"
$env:INFINEON_AUTH_TIMEOUT_SEC = "30"
$env:INFINEON_MAX_RETRIES = "1"
$env:INFINEON_RETRY_BACKOFF_SEC = "1"
$env:FUSEKI_QUERY_URL = "http://localhost:3030/infineon/sparql"
$env:INFINEON_ENABLE_LLM_CACHE = "1"

# ---------------------------------------------------------------------------
# 0. Optional: rebuild the controlled 1000-question plan from the current seed bank
# ---------------------------------------------------------------------------
python evaluation\build_kgqa_seed_bank.py `
  data\infineon\infineon_train.json `
  data\infineon\infineon_dev.json `
  data\infineon\infineon_test_final.json `
  data\infineon\kgqa_seed_expansion_round1.json `
  --out results\current_kgqa_seed_bank.json

python evaluation\build_final_kgqa_benchmark_plan.py `
  --seed-bank results\current_kgqa_seed_bank.json `
  --schema data\infineon\schema.json `
  --target-total 1000 `
  --out results\final_kgqa_benchmark_1000_plan.json

# ---------------------------------------------------------------------------
# 1. Generate the final 1000 natural-language gold benchmark.
#    This uses LLM calls for rewriting only. It is resumable.
# ---------------------------------------------------------------------------
python evaluation\generate_final_kgqa_benchmark_llm.py `
  --plan results\final_kgqa_benchmark_1000_plan.json `
  --out evaluation\question_sets\true_demand_final_1000_gold.json `
  --resume `
  --progress `
  --request-pause-sec 2.1

python evaluation\audit_generated_benchmark_wording.py `
  --dataset evaluation\question_sets\true_demand_final_1000_gold.json `
  --out results\true_demand_final_1000_wording_audit.json

# ---------------------------------------------------------------------------
# 2. LLM-only candidate generation and raw baseline selection.
#    No deterministic routing. This is the expensive KGQA candidate run.
# ---------------------------------------------------------------------------
python evaluation\run_infineon_holdout_eval.py `
  --dataset evaluation\question_sets\true_demand_final_1000_gold.json `
  --graph data\infineon\graph.ttl `
  --schema data\infineon\schema.json `
  --out results\final1000_current_llm_raw_candidates.json `
  --k 8 `
  --llm infineon `
  --temperature 0.2 `
  --query-timeout 10 `
  --resume `
  --progress

python evaluation\analyze_infineon_results.py `
  --results results\final1000_current_llm_raw_candidates.json `
  --dataset evaluation\question_sets\true_demand_final_1000_gold.json `
  --schema data\infineon\schema.json `
  --out-json results\final1000_current_llm_raw_analysis.json `
  --out-md results\final1000_current_llm_raw_analysis.md

# ---------------------------------------------------------------------------
# 3. Offline schema/semantic selector on the same candidates.
#    No new LLM generation calls.
# ---------------------------------------------------------------------------
python evaluation\apply_selection_to_results.py `
  --results results\final1000_current_llm_raw_candidates.json `
  --out results\final1000_current_schema_semantic_selection.json

python evaluation\analyze_infineon_results.py `
  --results results\final1000_current_schema_semantic_selection.json `
  --dataset evaluation\question_sets\true_demand_final_1000_gold.json `
  --schema data\infineon\schema.json `
  --out-json results\final1000_current_schema_semantic_analysis.json `
  --out-md results\final1000_current_schema_semantic_analysis.md

# ---------------------------------------------------------------------------
# 4. Offline guarded ML reranker on the same candidates.
#    No new LLM generation calls.
# ---------------------------------------------------------------------------
python evaluation\apply_ml_ranker_to_results.py `
  --results results\final1000_current_llm_raw_candidates.json `
  --model ranking\models\final1000_wf_ranker_current.json `
  --schema data\infineon\schema.json `
  --out results\final1000_current_guarded_ml_selection.json `
  --guarded `
  --min-margin 0.10 `
  --min-score 0.45 `
  --max-rank 4 `
  --structured-guard

python evaluation\analyze_infineon_results.py `
  --results results\final1000_current_guarded_ml_selection.json `
  --dataset evaluation\question_sets\true_demand_final_1000_gold.json `
  --schema data\infineon\schema.json `
  --out-json results\final1000_current_guarded_ml_analysis.json `
  --out-md results\final1000_current_guarded_ml_analysis.md

# ---------------------------------------------------------------------------
# 5. Build the mixed full-system benchmark.
#    This includes KG/data questions + DR ontology questions + advisory questions.
# ---------------------------------------------------------------------------
python evaluation\build_dr_ontology_benchmark.py `
  --dr-ontology "C:\Users\tsaousieleni\Downloads\DigitalReference.ttl" `
  --limit 300 `
  --out evaluation\question_sets\dr_ontology_benchmark_current.json

python evaluation\build_full_system_question_set.py `
  --kg-questions evaluation\question_sets\true_demand_final_1000_gold.json `
  --dr-questions evaluation\question_sets\dr_ontology_benchmark_current.json `
  --target-total 1000 `
  --kg-count 800 `
  --ontology-count 150 `
  --advisory-count 50 `
  --out evaluation\question_sets\true_demand_full_system_1000.json

# ---------------------------------------------------------------------------
# 6. Full system: deterministic first, LLM only when needed.
# ---------------------------------------------------------------------------
python evaluation\run_efficiency_question_set.py `
  --questions evaluation\question_sets\true_demand_full_system_1000.json `
  --out-log logs\kgqa_system_accuracy_1000_current.jsonl `
  --graph data\infineon\graph.ttl `
  --schema data\infineon\schema.json `
  --fuseki-query-url http://localhost:3030/infineon/sparql `
  --call-llm `
  --enable-llm-cache `
  --resume

python evaluation\analyze_system_efficiency.py `
  --log logs\kgqa_system_accuracy_1000_current.jsonl `
  --cost-per-call 0.20 `
  --out-json results\kgqa_efficiency_1000_current_report.json `
  --out-md results\kgqa_efficiency_1000_current_report.md

python evaluation\build_system_accuracy_audit.py `
  --log logs\kgqa_system_accuracy_1000_current.jsonl `
  --questions evaluation\question_sets\true_demand_full_system_1000.json `
  --out-csv results\kgqa_system_accuracy_audit_1000_current.csv

# Fill the correctness column in:
#   results\kgqa_system_accuracy_audit_1000_current.csv
#
# Then summarize it:
# python evaluation\build_system_accuracy_audit.py `
#   --labeled-csv results\kgqa_system_accuracy_audit_1000_current_labeled.csv `
#   --out-json results\kgqa_system_accuracy_audit_1000_current_labeled.json `
#   --out-md results\kgqa_system_accuracy_audit_1000_current_labeled.md `
#   --unclear-as-incorrect

# ---------------------------------------------------------------------------
# 7. Optional final summary after the labeled audit exists.
# ---------------------------------------------------------------------------
# python evaluation\build_final_evaluation_summary.py `
#   --system-accuracy results\kgqa_system_accuracy_audit_1000_current_labeled.json `
#   --efficiency results\kgqa_efficiency_1000_current_report.json `
#   --selection results\final1000_current_guarded_ml_analysis.json `
#   --baseline-vs-ml results\final1000_current_guarded_ml_analysis.json `
#   --dr-ontology results\dr_ontology_benchmark_current_report.json `
#   --out-json results\final_evaluation_summary_1000_current.json `
#   --out-md results\final_evaluation_summary_1000_current.md
