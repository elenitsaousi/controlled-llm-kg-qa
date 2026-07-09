# Calibrate the ML reranker on 200 stratified final-1000 questions and
# evaluate only on the untouched 800-question holdout.
#
# Run from:
#   C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa
#
# Required existing files:
# - evaluation\question_sets\true_demand_final_1000_gold.json
# - results\final1000_current_llm_raw_candidates.json
#
# This script does not call the LLM. It reuses the saved candidate set.

$ErrorActionPreference = "Stop"

$Dataset = "evaluation\question_sets\true_demand_final_1000_gold.json"
$RawResults = "results\final1000_current_llm_raw_candidates.json"
$Schema = "data\infineon\schema.json"
$OldModel = "ranking\models\final1000_wf_ranker_current.json"
$NewModel = "ranking\models\final1000_wf_plus_calibration200_ranker.json"

# 1. Stratified split: 200 calibration questions, 800 untouched holdout.
python evaluation\build_final1000_calibration_split.py `
  --dataset $Dataset `
  --results $RawResults `
  --calibration-size 200 `
  --seed 20260709 `
  --out-calibration evaluation\question_sets\true_demand_final1000_calibration_200.json `
  --out-holdout evaluation\question_sets\true_demand_final1000_holdout_800.json `
  --out-manifest results\final1000_calibration_split_manifest.json

# 2. Filter the same raw candidate set into calibration and holdout files.
python evaluation\filter_results_by_dataset.py `
  --results $RawResults `
  --dataset evaluation\question_sets\true_demand_final1000_calibration_200.json `
  --out results\final1000_calibration200_raw_candidates.json

python evaluation\filter_results_by_dataset.py `
  --results $RawResults `
  --dataset evaluation\question_sets\true_demand_final1000_holdout_800.json `
  --out results\final1000_holdout800_raw_candidates.json

# 3. Build calibration training rows and merge with the historical training set.
python evaluation\build_ranker_training_from_results.py `
  --results results\final1000_calibration200_raw_candidates.json `
  --schema $Schema `
  --dataset evaluation\question_sets\true_demand_final1000_calibration_200.json `
  --out ranking\final1000_calibration200_ranker_data.json

python evaluation\merge_ranker_training_data.py `
  --base ranking\final1000_wf_train_ranker_data.json `
  --extra ranking\final1000_calibration200_ranker_data.json `
  --out ranking\final1000_wf_plus_calibration200_ranker_data.json

# 4. Train the calibrated model. CV is training diagnostics only, not final accuracy.
python ranking\train_infineon_np_tfidf_ranker.py `
  --training-data ranking\final1000_wf_plus_calibration200_ranker_data.json `
  --cv-out results\final1000_wf_plus_calibration200_ranker_cv.json `
  --model-out $NewModel

# 5. Tune guarded thresholds on calibration only.
python evaluation\sweep_guarded_ml_rerank.py `
  --results results\final1000_calibration200_raw_candidates.json `
  --model $NewModel `
  --schema $Schema `
  --out results\final1000_calibration200_guarded_ml_sweep.json `
  --margins "0.00,0.05,0.10,0.15,0.20,0.25" `
  --scores "0.35,0.40,0.45,0.50,0.55,0.60" `
  --max-ranks "1,2,3,4,5,6,8" `
  --structured-guard

# 6. Holdout baselines: raw, old frozen ML, and new calibrated ML.
python evaluation\analyze_infineon_results.py `
  --results results\final1000_holdout800_raw_candidates.json `
  --dataset evaluation\question_sets\true_demand_final1000_holdout_800.json `
  --schema $Schema `
  --out-json results\final1000_holdout800_raw_analysis.json `
  --out-md results\final1000_holdout800_raw_analysis.md

python evaluation\apply_ml_ranker_to_results.py `
  --results results\final1000_holdout800_raw_candidates.json `
  --model $OldModel `
  --schema $Schema `
  --out results\final1000_holdout800_old_guarded_ml_selection.json `
  --guarded `
  --min-margin 0.10 `
  --min-score 0.45 `
  --max-rank 4 `
  --structured-guard

python evaluation\analyze_infineon_results.py `
  --results results\final1000_holdout800_old_guarded_ml_selection.json `
  --dataset evaluation\question_sets\true_demand_final1000_holdout_800.json `
  --schema $Schema `
  --out-json results\final1000_holdout800_old_guarded_ml_analysis.json `
  --out-md results\final1000_holdout800_old_guarded_ml_analysis.md

python evaluation\apply_best_guarded_ml_from_sweep.py `
  --sweep results\final1000_calibration200_guarded_ml_sweep.json `
  --results results\final1000_holdout800_raw_candidates.json `
  --model $NewModel `
  --schema $Schema `
  --out results\final1000_holdout800_calibrated_guarded_ml_selection.json

python evaluation\analyze_infineon_results.py `
  --results results\final1000_holdout800_calibrated_guarded_ml_selection.json `
  --dataset evaluation\question_sets\true_demand_final1000_holdout_800.json `
  --schema $Schema `
  --out-json results\final1000_holdout800_calibrated_guarded_ml_analysis.json `
  --out-md results\final1000_holdout800_calibrated_guarded_ml_analysis.md

# 7. Final comparison table. Use the holdout numbers for thesis claims.
python evaluation\compare_selection_analysis_reports.py `
  --report raw=results\final1000_holdout800_raw_analysis.json `
  --report old_ml=results\final1000_holdout800_old_guarded_ml_analysis.json `
  --report calibrated_ml=results\final1000_holdout800_calibrated_guarded_ml_analysis.json `
  --out-json results\final1000_holdout800_calibrated_ml_comparison.json `
  --out-md results\final1000_holdout800_calibrated_ml_comparison.md

Write-Host ""
Write-Host "Done. Check:"
Write-Host "  results\final1000_holdout800_calibrated_ml_comparison.md"
Write-Host ""
Write-Host "If calibrated_ml is worse than old_ml on holdout, do NOT promote the new model."
