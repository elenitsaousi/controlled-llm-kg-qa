# Diagnostics for the final thesis feedback items.
# Run from the repository root on the Windows laptop after pulling the branch.

$AuditCsv = "results\kgqa_system_accuracy_audit_1000_current_labeled_codex.csv"
if (-not (Test-Path $AuditCsv)) {
  $AuditCsv = "results\kgqa_system_accuracy_audit_1000_current.csv"
}

$RankerModel = "ranking\models\final1000_wf_plus_calibration200_ranker.json"
if (-not (Test-Path $RankerModel)) {
  $RankerModel = "ranking\models\final1000_wf_ranker_current.json"
}

python evaluation\analyze_schema_ml_signals.py `
  --raw-analysis results\final1000_current_llm_raw_analysis.json `
  --schema-analysis results\final1000_current_schema_semantic_analysis.json `
  --ml-analysis results\final1000_current_guarded_ml_analysis.json `
  --model $RankerModel `
  --out-json results\current_schema_ml_signal_analysis.json `
  --out-md results\current_schema_ml_signal_analysis.md

python evaluation\analyze_execution_signature_entropy.py `
  --results results\final1000_current_guarded_ml_selection.json `
  --score-key ml_score `
  --temperature 0.10 `
  --out-json results\current_execution_signature_entropy.json `
  --out-md results\current_execution_signature_entropy.md

python evaluation\analyze_answer_level_delta.py `
  --selection-results results\final1000_current_guarded_ml_selection.json `
  --audit-csv $AuditCsv `
  --out-json results\current_selection_answer_delta.json `
  --out-md results\current_selection_answer_delta.md

python evaluation\analyze_advisory_route.py `
  --audit-csv $AuditCsv `
  --out-json results\current_advisory_route_audit.json `
  --out-md results\current_advisory_route_audit.md

python evaluation\estimate_manual_sparql_effort.py `
  --audit-csv $AuditCsv `
  --only-incorrect `
  --only-llm `
  --difficulty-counts easy=53 medium=37 hard=23 `
  --easy-minutes 5 `
  --medium-minutes 15 `
  --hard-minutes 30 `
  --out-json results\current_manual_sparql_effort_estimate.json `
  --out-md results\current_manual_sparql_effort_estimate.md
