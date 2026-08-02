# Clean ML reranker experiment for the repaired final-1000 benchmark.
#
# Run from:
#   C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa
#
# Required existing files:
# - evaluation\question_sets\true_demand_final_1000_gold_repaired.json
# - results\final1000_repaired_llm_raw_candidates.json
#
# This script does not call the LLM. It reuses the saved repaired candidate set.
# It intentionally does NOT merge with historical training data, so the holdout
# score is a clean generalization check.

$ErrorActionPreference = "Stop"

$Dataset = "evaluation\question_sets\true_demand_final_1000_gold_repaired.json"
$RawResults = "results\final1000_repaired_llm_raw_candidates.json"
$Schema = "data\infineon\schema.json"
$OldModel = "ranking\models\final1000_wf_ranker_current.json"
$NewModel = "ranking\models\final1000_repaired_train600_ranker.json"

$TrainSet = "evaluation\question_sets\true_demand_final1000_repaired_train_600.json"
$TuneSet = "evaluation\question_sets\true_demand_final1000_repaired_tune_200.json"
$HoldoutSet = "evaluation\question_sets\true_demand_final1000_repaired_holdout_200.json"

if (-not (Test-Path $Dataset)) {
  throw "Missing repaired dataset: $Dataset"
}
if (-not (Test-Path $RawResults)) {
  throw "Missing repaired raw candidate results: $RawResults. Run the repaired LLM-only candidate generation first."
}
if (-not (Test-Path $Schema)) {
  throw "Missing schema: $Schema"
}

# 1. Non-overlapping split: 600 train, 200 threshold tuning, 200 untouched holdout.
python evaluation\build_final1000_train_tune_holdout_split.py `
  --dataset $Dataset `
  --results $RawResults `
  --train-size 600 `
  --tune-size 200 `
  --seed 20260802 `
  --out-train $TrainSet `
  --out-tune $TuneSet `
  --out-holdout $HoldoutSet `
  --out-manifest results\final1000_repaired_train_tune_holdout_manifest.json

# 2. Filter the same repaired candidate set into train, tune, and holdout files.
python evaluation\filter_results_by_dataset.py `
  --results $RawResults `
  --dataset $TrainSet `
  --out results\final1000_repaired_train600_raw_candidates.json

python evaluation\filter_results_by_dataset.py `
  --results $RawResults `
  --dataset $TuneSet `
  --out results\final1000_repaired_tune200_raw_candidates.json

python evaluation\filter_results_by_dataset.py `
  --results $RawResults `
  --dataset $HoldoutSet `
  --out results\final1000_repaired_holdout200_raw_candidates.json

# 3. Build training rows from train only, then train the clean model.
python evaluation\build_ranker_training_from_results.py `
  --results results\final1000_repaired_train600_raw_candidates.json `
  --schema $Schema `
  --dataset $TrainSet `
  --out ranking\final1000_repaired_train600_ranker_data.json

python ranking\train_infineon_np_tfidf_ranker.py `
  --training-data ranking\final1000_repaired_train600_ranker_data.json `
  --cv-out results\final1000_repaired_train600_ranker_cv.json `
  --model-out $NewModel

# 4. Tune guarded thresholds on tune only.
python evaluation\sweep_guarded_ml_rerank.py `
  --results results\final1000_repaired_tune200_raw_candidates.json `
  --model $NewModel `
  --schema $Schema `
  --out results\final1000_repaired_tune200_guarded_ml_sweep.json `
  --margins "0.00,0.05,0.10,0.15,0.20,0.25" `
  --scores "0.35,0.40,0.45,0.50,0.55,0.60" `
  --max-ranks "1,2,3,4,5,6,8" `
  --structured-guard `
  --enable-rank2-trusted-rescue `
  --trusted-rescue-max-rank 4 `
  --trusted-rescue-min-score 0.20 `
  --trusted-rescue-min-margin -0.25 `
  --trusted-rescue-topics "inventory,order_cancellation,vehicle_sales" `
  --enable-shortage-status-rescue `
  --shortage-status-rescue-max-rank 5 `
  --shortage-status-rescue-min-score 0.20 `
  --shortage-status-rescue-min-margin -0.25 `
  --enable-current-baseline-rescue `
  --current-baseline-rescue-max-rank 6 `
  --current-baseline-rescue-min-score 0.20 `
  --current-baseline-rescue-min-margin -0.25

# 5. Apply the best tune thresholds back to tune, then derive a conservative
# family policy. This policy is learned on tune only, never on holdout.
python evaluation\apply_best_guarded_ml_from_sweep.py `
  --sweep results\final1000_repaired_tune200_guarded_ml_sweep.json `
  --results results\final1000_repaired_tune200_raw_candidates.json `
  --model $NewModel `
  --schema $Schema `
  --out results\final1000_repaired_tune200_clean_guarded_ml_selection.json

python evaluation\derive_family_selection_policy.py `
  --raw-results results\final1000_repaired_tune200_raw_candidates.json `
  --selected-results results\final1000_repaired_tune200_clean_guarded_ml_selection.json `
  --dataset $TuneSet `
  --min-delta-questions 1 `
  --min-gain-loss-ratio 1.0 `
  --out-json results\final1000_repaired_tune200_family_ml_policy.json `
  --out-md results\final1000_repaired_tune200_family_ml_policy.md

# 6. Evaluate raw, schema/semantic, old ML, clean ML, and family-gated clean ML
# on untouched holdout.
python evaluation\analyze_infineon_results.py `
  --results results\final1000_repaired_holdout200_raw_candidates.json `
  --dataset $HoldoutSet `
  --schema $Schema `
  --out-json results\final1000_repaired_holdout200_raw_analysis.json `
  --out-md results\final1000_repaired_holdout200_raw_analysis.md

python evaluation\analyze_selection_equivalence.py `
  --results results\final1000_repaired_holdout200_raw_candidates.json `
  --dataset $HoldoutSet `
  --out-json results\final1000_repaired_holdout200_raw_equivalence.json `
  --out-md results\final1000_repaired_holdout200_raw_equivalence.md

python evaluation\apply_selection_to_results.py `
  --results results\final1000_repaired_holdout200_raw_candidates.json `
  --out results\final1000_repaired_holdout200_schema_semantic_selection.json `
  --use-stored-features

python evaluation\analyze_infineon_results.py `
  --results results\final1000_repaired_holdout200_schema_semantic_selection.json `
  --dataset $HoldoutSet `
  --schema $Schema `
  --out-json results\final1000_repaired_holdout200_schema_semantic_analysis.json `
  --out-md results\final1000_repaired_holdout200_schema_semantic_analysis.md

if (Test-Path $OldModel) {
  python evaluation\apply_ml_ranker_to_results.py `
    --results results\final1000_repaired_holdout200_raw_candidates.json `
    --model $OldModel `
    --schema $Schema `
    --out results\final1000_repaired_holdout200_old_guarded_ml_selection.json `
    --guarded `
    --min-margin 0.10 `
    --min-score 0.45 `
    --max-rank 4 `
    --structured-guard

  python evaluation\analyze_infineon_results.py `
    --results results\final1000_repaired_holdout200_old_guarded_ml_selection.json `
    --dataset $HoldoutSet `
    --schema $Schema `
    --out-json results\final1000_repaired_holdout200_old_guarded_ml_analysis.json `
    --out-md results\final1000_repaired_holdout200_old_guarded_ml_analysis.md
}

python evaluation\apply_best_guarded_ml_from_sweep.py `
  --sweep results\final1000_repaired_tune200_guarded_ml_sweep.json `
  --results results\final1000_repaired_holdout200_raw_candidates.json `
  --model $NewModel `
  --schema $Schema `
  --out results\final1000_repaired_holdout200_clean_guarded_ml_selection.json

python evaluation\analyze_infineon_results.py `
  --results results\final1000_repaired_holdout200_clean_guarded_ml_selection.json `
  --dataset $HoldoutSet `
  --schema $Schema `
  --out-json results\final1000_repaired_holdout200_clean_guarded_ml_analysis.json `
  --out-md results\final1000_repaired_holdout200_clean_guarded_ml_analysis.md

python evaluation\analyze_selection_equivalence.py `
  --results results\final1000_repaired_holdout200_clean_guarded_ml_selection.json `
  --dataset $HoldoutSet `
  --out-json results\final1000_repaired_holdout200_clean_guarded_ml_equivalence.json `
  --out-md results\final1000_repaired_holdout200_clean_guarded_ml_equivalence.md

python evaluation\apply_family_gated_selection.py `
  --raw-results results\final1000_repaired_holdout200_raw_candidates.json `
  --selected-results results\final1000_repaired_holdout200_clean_guarded_ml_selection.json `
  --dataset $HoldoutSet `
  --policy results\final1000_repaired_tune200_family_ml_policy.json `
  --out results\final1000_repaired_holdout200_family_gated_clean_ml_selection.json

python evaluation\analyze_infineon_results.py `
  --results results\final1000_repaired_holdout200_family_gated_clean_ml_selection.json `
  --dataset $HoldoutSet `
  --schema $Schema `
  --out-json results\final1000_repaired_holdout200_family_gated_clean_ml_analysis.json `
  --out-md results\final1000_repaired_holdout200_family_gated_clean_ml_analysis.md

python evaluation\analyze_selection_equivalence.py `
  --results results\final1000_repaired_holdout200_family_gated_clean_ml_selection.json `
  --dataset $HoldoutSet `
  --out-json results\final1000_repaired_holdout200_family_gated_clean_ml_equivalence.json `
  --out-md results\final1000_repaired_holdout200_family_gated_clean_ml_equivalence.md

if (Test-Path "results\final1000_repaired_holdout200_old_guarded_ml_analysis.json") {
  python evaluation\compare_selection_analysis_reports.py `
    --report raw=results\final1000_repaired_holdout200_raw_analysis.json `
    --report schema=results\final1000_repaired_holdout200_schema_semantic_analysis.json `
    --report old_ml=results\final1000_repaired_holdout200_old_guarded_ml_analysis.json `
    --report clean_ml=results\final1000_repaired_holdout200_clean_guarded_ml_analysis.json `
    --report family_gated_clean_ml=results\final1000_repaired_holdout200_family_gated_clean_ml_analysis.json `
    --selection raw=results\final1000_repaired_holdout200_raw_candidates.json `
    --selection schema=results\final1000_repaired_holdout200_schema_semantic_selection.json `
    --selection old_ml=results\final1000_repaired_holdout200_old_guarded_ml_selection.json `
    --selection clean_ml=results\final1000_repaired_holdout200_clean_guarded_ml_selection.json `
    --selection family_gated_clean_ml=results\final1000_repaired_holdout200_family_gated_clean_ml_selection.json `
    --out-json results\final1000_repaired_holdout200_clean_ml_comparison.json `
    --out-md results\final1000_repaired_holdout200_clean_ml_comparison.md
} else {
  python evaluation\compare_selection_analysis_reports.py `
    --report raw=results\final1000_repaired_holdout200_raw_analysis.json `
    --report schema=results\final1000_repaired_holdout200_schema_semantic_analysis.json `
    --report clean_ml=results\final1000_repaired_holdout200_clean_guarded_ml_analysis.json `
    --report family_gated_clean_ml=results\final1000_repaired_holdout200_family_gated_clean_ml_analysis.json `
    --selection raw=results\final1000_repaired_holdout200_raw_candidates.json `
    --selection schema=results\final1000_repaired_holdout200_schema_semantic_selection.json `
    --selection clean_ml=results\final1000_repaired_holdout200_clean_guarded_ml_selection.json `
    --selection family_gated_clean_ml=results\final1000_repaired_holdout200_family_gated_clean_ml_selection.json `
    --out-json results\final1000_repaired_holdout200_clean_ml_comparison.json `
    --out-md results\final1000_repaired_holdout200_clean_ml_comparison.md
}

Write-Host ""
Write-Host "Done. Check:"
Write-Host "  results\final1000_repaired_holdout200_clean_ml_comparison.md"
Write-Host "  results\final1000_repaired_train_tune_holdout_manifest.json"
Write-Host "  results\final1000_repaired_train600_ranker_cv.json"
Write-Host "  results\final1000_repaired_tune200_family_ml_policy.md"
Write-Host "  results\final1000_repaired_holdout200_family_gated_clean_ml_equivalence.md"
Write-Host ""
Write-Host "Do not promote the new model unless family_gated_clean_ml improves the untouched holdout."
