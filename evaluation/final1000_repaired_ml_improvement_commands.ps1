# Feature-rich ML/LTR experiment for the repaired final-1000 benchmark.
#
# Run from:
#   C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa
#
# Required existing files:
# - evaluation\question_sets\true_demand_final_1000_gold_repaired.json
# - results\final1000_repaired_llm_raw_candidates.json
#
# This script does not call the LLM. It reuses the saved repaired candidate set.
# It evaluates improved selection features and XGBoost learning-to-rank on a
# clean train/tune/holdout split. Do not promote a model unless it improves the
# untouched holdout.

$ErrorActionPreference = "Stop"

$Dataset = "evaluation\question_sets\true_demand_final_1000_gold_repaired.json"
$RawResults = "results\final1000_repaired_llm_raw_candidates.json"
$Schema = "data\infineon\schema.json"

$TrainSet = "evaluation\question_sets\true_demand_final1000_repaired_train_600.json"
$TuneSet = "evaluation\question_sets\true_demand_final1000_repaired_tune_200.json"
$HoldoutSet = "evaluation\question_sets\true_demand_final1000_repaired_holdout_200.json"

$TrainResults = "results\final1000_repaired_train600_raw_candidates.json"
$TuneResults = "results\final1000_repaired_tune200_raw_candidates.json"
$HoldoutResults = "results\final1000_repaired_holdout200_raw_candidates.json"
$TrainingRows = "ranking\final1000_repaired_train600_ranker_data.json"

$NpModel = "ranking\models\final1000_repaired_train600_feature_rich_ranker.json"
$XgbPairwiseModel = "ranking\models\final1000_repaired_train600_xgb_pairwise_ltr.pkl"
$XgbNdcgModel = "ranking\models\final1000_repaired_train600_xgb_ndcg_ltr.pkl"

if (-not (Test-Path $Dataset)) {
  throw "Missing repaired dataset: $Dataset"
}
if (-not (Test-Path $RawResults)) {
  throw "Missing repaired raw candidate results: $RawResults. Run repaired LLM-only candidate generation first."
}
if (-not (Test-Path $Schema)) {
  throw "Missing schema: $Schema"
}

Write-Host "===== SPLIT + FILTER REPAIRED CANDIDATES ====="
python evaluation\build_final1000_train_tune_holdout_split.py `
  --dataset $Dataset `
  --results $RawResults `
  --train-size 600 `
  --tune-size 200 `
  --seed 20260802 `
  --out-train $TrainSet `
  --out-tune $TuneSet `
  --out-holdout $HoldoutSet `
  --out-manifest results\final1000_repaired_ml_improvement_split_manifest.json

python evaluation\filter_results_by_dataset.py --results $RawResults --dataset $TrainSet --out $TrainResults
python evaluation\filter_results_by_dataset.py --results $RawResults --dataset $TuneSet --out $TuneResults
python evaluation\filter_results_by_dataset.py --results $RawResults --dataset $HoldoutSet --out $HoldoutResults

Write-Host "===== TRAINING ROWS ====="
python evaluation\build_ranker_training_from_results.py `
  --results $TrainResults `
  --schema $Schema `
  --dataset $TrainSet `
  --out $TrainingRows

Write-Host "===== BASELINE HOLDOUT ANALYSES ====="
python evaluation\analyze_infineon_results.py `
  --results $HoldoutResults `
  --dataset $HoldoutSet `
  --schema $Schema `
  --out-json results\final1000_repaired_holdout200_improved_raw_analysis.json `
  --out-md results\final1000_repaired_holdout200_improved_raw_analysis.md

python evaluation\apply_selection_to_results.py `
  --results $HoldoutResults `
  --out results\final1000_repaired_holdout200_improved_schema_selection.json `
  --use-stored-features

python evaluation\analyze_infineon_results.py `
  --results results\final1000_repaired_holdout200_improved_schema_selection.json `
  --dataset $HoldoutSet `
  --schema $Schema `
  --out-json results\final1000_repaired_holdout200_improved_schema_analysis.json `
  --out-md results\final1000_repaired_holdout200_improved_schema_analysis.md

Write-Host "===== TRAIN FEATURE-RICH NP TF-IDF RANKER ====="
python ranking\train_infineon_np_tfidf_ranker.py `
  --training-data $TrainingRows `
  --cv-out results\final1000_repaired_train600_feature_rich_ranker_cv.json `
  --model-out $NpModel

Write-Host "===== TRAIN XGBOOST LTR RANKERS ====="
python ranking\train_xgboost_ltr_ranker.py `
  --training-data $TrainingRows `
  --cv-out results\final1000_repaired_train600_xgb_pairwise_ltr_cv.json `
  --model-out $XgbPairwiseModel `
  --objective rank:pairwise `
  --n-estimators 180 `
  --max-depth 3 `
  --learning-rate 0.03 `
  --subsample 0.9 `
  --colsample-bytree 0.9 `
  --reg-lambda 8.0

python ranking\train_xgboost_ltr_ranker.py `
  --training-data $TrainingRows `
  --cv-out results\final1000_repaired_train600_xgb_ndcg_ltr_cv.json `
  --model-out $XgbNdcgModel `
  --objective rank:ndcg `
  --n-estimators 180 `
  --max-depth 3 `
  --learning-rate 0.03 `
  --subsample 0.9 `
  --colsample-bytree 0.9 `
  --reg-lambda 8.0

function Apply-SweptModel {
  param(
    [string]$Name,
    [string]$Model
  )

  Write-Host "===== SWEEP $Name ON TUNE ====="
  $Sweep = "results\final1000_repaired_tune200_${Name}_sweep.json"
  $TuneSelected = "results\final1000_repaired_tune200_${Name}_selection.json"
  $HoldoutSelected = "results\final1000_repaired_holdout200_${Name}_selection.json"

  python evaluation\sweep_guarded_ml_rerank.py `
    --results $TuneResults `
    --model $Model `
    --schema $Schema `
    --out $Sweep `
    --margins "-1.00,-0.50,-0.25,-0.10,0.00,0.05,0.10,0.15,0.20,0.25" `
    --scores "-10.00,-2.00,-1.00,-0.50,0.00,0.20,0.35,0.45,0.55,0.65" `
    --max-ranks "1,2,3,4,5,6,8" `
    --structured-guard `
    --enable-rank2-trusted-rescue `
    --trusted-rescue-max-rank 4 `
    --trusted-rescue-min-score -10.0 `
    --trusted-rescue-min-margin -1.0 `
    --trusted-rescue-topics "inventory,order_cancellation,vehicle_sales,future_demand,current_demand_baselines,regional_demand" `
    --enable-shortage-status-rescue `
    --shortage-status-rescue-max-rank 5 `
    --shortage-status-rescue-min-score -10.0 `
    --shortage-status-rescue-min-margin -1.0 `
    --enable-current-baseline-rescue `
    --current-baseline-rescue-max-rank 6 `
    --current-baseline-rescue-min-score -10.0 `
    --current-baseline-rescue-min-margin -1.0

  python evaluation\apply_best_guarded_ml_from_sweep.py `
    --sweep $Sweep `
    --results $TuneResults `
    --model $Model `
    --schema $Schema `
    --out $TuneSelected

  python evaluation\apply_best_guarded_ml_from_sweep.py `
    --sweep $Sweep `
    --results $HoldoutResults `
    --model $Model `
    --schema $Schema `
    --out $HoldoutSelected

  python evaluation\analyze_infineon_results.py `
    --results $HoldoutSelected `
    --dataset $HoldoutSet `
    --schema $Schema `
    --out-json "results\final1000_repaired_holdout200_${Name}_analysis.json" `
    --out-md "results\final1000_repaired_holdout200_${Name}_analysis.md"

  python evaluation\analyze_selection_equivalence.py `
    --results $HoldoutSelected `
    --dataset $HoldoutSet `
    --out-json "results\final1000_repaired_holdout200_${Name}_equivalence.json" `
    --out-md "results\final1000_repaired_holdout200_${Name}_equivalence.md"
}

Apply-SweptModel -Name "feature_rich_np" -Model $NpModel
Apply-SweptModel -Name "xgb_pairwise_ltr" -Model $XgbPairwiseModel
Apply-SweptModel -Name "xgb_ndcg_ltr" -Model $XgbNdcgModel

Write-Host "===== COMPARISON ====="
python evaluation\compare_selection_analysis_reports.py `
  --report raw=results\final1000_repaired_holdout200_improved_raw_analysis.json `
  --report schema=results\final1000_repaired_holdout200_improved_schema_analysis.json `
  --report feature_rich_np=results\final1000_repaired_holdout200_feature_rich_np_analysis.json `
  --report xgb_pairwise_ltr=results\final1000_repaired_holdout200_xgb_pairwise_ltr_analysis.json `
  --report xgb_ndcg_ltr=results\final1000_repaired_holdout200_xgb_ndcg_ltr_analysis.json `
  --selection raw=$HoldoutResults `
  --selection schema=results\final1000_repaired_holdout200_improved_schema_selection.json `
  --selection feature_rich_np=results\final1000_repaired_holdout200_feature_rich_np_selection.json `
  --selection xgb_pairwise_ltr=results\final1000_repaired_holdout200_xgb_pairwise_ltr_selection.json `
  --selection xgb_ndcg_ltr=results\final1000_repaired_holdout200_xgb_ndcg_ltr_selection.json `
  --out-json results\final1000_repaired_holdout200_ml_improvement_comparison.json `
  --out-md results\final1000_repaired_holdout200_ml_improvement_comparison.md

Write-Host ""
Write-Host "Done. Main files to inspect:"
Write-Host "  results\final1000_repaired_holdout200_ml_improvement_comparison.md"
Write-Host "  results\final1000_repaired_train600_feature_rich_ranker_cv.json"
Write-Host "  results\final1000_repaired_train600_xgb_pairwise_ltr_cv.json"
Write-Host "  results\final1000_repaired_train600_xgb_ndcg_ltr_cv.json"
Write-Host ""
Write-Host "Promote only the best untouched-holdout model, not the best tune result."
