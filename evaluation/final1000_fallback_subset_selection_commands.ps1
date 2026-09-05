$ErrorActionPreference = "Stop"

$Audit = "results\kgqa_system_accuracy_audit_1000_repaired_labeled.csv"
$Dataset = "evaluation\question_sets\true_demand_final_1000_gold_repaired.json"

if (-not (Test-Path $Audit)) {
    throw "Missing audit CSV: $Audit"
}
if (-not (Test-Path $Dataset)) {
    throw "Missing repaired KG dataset: $Dataset"
}

$Selections = @()

if (Test-Path "results\final1000_repaired_llm_raw_candidates.json") {
    $Selections += "--selection"
    $Selections += "Raw LLM selection=results\final1000_repaired_llm_raw_candidates.json"
}
if (Test-Path "results\final1000_repaired_schema_semantic_selection.json") {
    $Selections += "--selection"
    $Selections += "Schema-only selection=results\final1000_repaired_schema_semantic_selection.json"
}
if (Test-Path "results\final1000_repaired_guarded_ml_selection.json") {
    $Selections += "--selection"
    $Selections += "Guarded ML selection=results\final1000_repaired_guarded_ml_selection.json"
}
if (Test-Path "results\final1000_repaired_family_gated_clean_ml_selection.json") {
    $Selections += "--selection"
    $Selections += "Family-gated clean ML selection=results\final1000_repaired_family_gated_clean_ml_selection.json"
}

if ($Selections.Count -eq 0) {
    throw "No repaired selection result JSON files found in results\. Run the raw/schema/ML selection commands first."
}

python evaluation\analyze_fallback_subset_selection.py `
  --audit-csv $Audit `
  --dataset $Dataset `
  @Selections `
  --out-json results\final1000_repaired_fallback_subset_selection.json `
  --out-md results\final1000_repaired_fallback_subset_selection.md
