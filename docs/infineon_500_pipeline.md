# Infineon 500Q Pipeline (Step-by-Step)

This is the full workflow for:
- 500-question benchmark
- train/dev/test leakage-safe split
- ML ranker training
- ambiguity-gated policy calibration
- KPI + generation-recall reporting

## 0) Environment

Set `.env` (same as your current Infineon setup):

```bash
LLM_BACKEND=infineon
INFINEON_API_URL=https://...
INFINEON_API_KEY=...
INFINEON_MODEL=gpt-4o
INFINEON_CHAT_ENDPOINT=/chat/completions
```

## 1) Generate 500 benchmark questions

```bash
python data/infineon/generate_infineon_dataset_500.py \
  --seed data/infineon/infineon_dataset_100.json \
  --out data/infineon/infineon_dataset_500.json \
  --target 500
```

Output:
- `data/infineon/infineon_dataset_500.json`

What to verify:
- total `500`
- labels roughly preserved (`low/mid/high`)
- `family` field exists

## 2) Create train/dev/test split (grouped by family)

```bash
python data/infineon/split_infineon_dataset.py \
  --dataset data/infineon/infineon_dataset_500.json \
  --out-dir data/infineon/splits/infineon_500 \
  --ratios 0.8,0.1,0.1 \
  --seed 42
```

Outputs:
- `data/infineon/splits/infineon_500/train.json`
- `data/infineon/splits/infineon_500/dev.json`
- `data/infineon/splits/infineon_500/test.json`
- `data/infineon/splits/infineon_500/manifest.json`

## 3) Build candidate pools (generation stage) for each split

Train:
```bash
python ranking/build_infineon_training_data.py \
  --dataset data/infineon/splits/infineon_500/train.json \
  --graph data/infineon/graph.ttl \
  --schema data/infineon/schema.json \
  --out ranking/infineon_train_candidates_500.json \
  --k 5 \
  --n-runs 3
```

Dev:
```bash
python ranking/build_infineon_training_data.py \
  --dataset data/infineon/splits/infineon_500/dev.json \
  --graph data/infineon/graph.ttl \
  --schema data/infineon/schema.json \
  --out ranking/infineon_dev_candidates_500.json \
  --k 5 \
  --n-runs 3
```

Test:
```bash
python ranking/build_infineon_training_data.py \
  --dataset data/infineon/splits/infineon_500/test.json \
  --graph data/infineon/graph.ttl \
  --schema data/infineon/schema.json \
  --out ranking/infineon_test_candidates_500.json \
  --k 5 \
  --n-runs 3
```

What to track at this step:
- `Correct candidates: X (Y%)` = candidate-level generation precision
- question-level generation recall will be measured in step 4 (`generation_recall_rate`)

## 4) Train ranker and get split KPIs (clean comparison on same candidate sets)

```bash
python ranking/train_infineon_np_tfidf_split.py \
  --train-data ranking/infineon_train_candidates_500.json \
  --dev-data ranking/infineon_dev_candidates_500.json \
  --test-data ranking/infineon_test_candidates_500.json \
  --model-out ranking/models/infineon_np_tfidf_ranker_500.json \
  --report-out results/infineon_split_kpi_500.json \
  --ml-regimes mid
```

Outputs:
- model: `ranking/models/infineon_np_tfidf_ranker_500.json`
- KPI report: `results/infineon_split_kpi_500.json`

Main KPI fields (test split):
- `generation_recall_rate` (any-correct upper bound)
- `no_ml_top1_rate`
- `ml_top1_rate`
- `gated_top1_rate`
- `delta_gated_vs_no_ml`

## 5) Fit ambiguity policy (tau1/tau2) on dev split

```bash
python analysis/fit_infineon_ambiguity_policy.py \
  --train-data ranking/infineon_train_candidates_500.json \
  --calib-data ranking/infineon_dev_candidates_500.json \
  --model ranking/models/infineon_np_tfidf_ranker_500.json \
  --schema data/infineon/schema.json \
  --graph data/infineon/graph.ttl \
  --entropy-source agreement \
  --ml-regimes mid \
  --out-config ranking/models/infineon_ambiguity_config_500.json \
  --out-report results/infineon_ambiguity_calibration_500.json
```

Outputs:
- policy config: `ranking/models/infineon_ambiguity_config_500.json`
- calibration report: `results/infineon_ambiguity_calibration_500.json`

## 6) Runtime eval with learned ambiguity gating (optional)

```bash
python evaluation/run_infineon_100_eval.py \
  --dataset data/infineon/infineon_dataset_100.json \
  --use-ml-ranking \
  --ml-model ranking/models/infineon_np_tfidf_ranker_500.json \
  --ambiguity-config ranking/models/infineon_ambiguity_config_500.json \
  --out results/infineon_eval_100_ml_gated_from_config.json \
  --progress
```

## 7) Produce one consolidated KPI summary

```bash
python analysis/infineon_kpi_report.py \
  --split-report results/infineon_split_kpi_500.json \
  --out results/infineon_kpi_summary_500.json
```

## One-command orchestrator (steps 1-5)

If you prefer one command:

```bash
python analysis/run_infineon_500_pipeline.py
```

You can skip completed steps, for example:

```bash
python analysis/run_infineon_500_pipeline.py --skip-generate --skip-split
```

If you also ran runtime eval files:

```bash
python analysis/infineon_kpi_report.py \
  --split-report results/infineon_split_kpi_500.json \
  --eval-no-ml results/infineon_eval_100_no_ml.json \
  --eval-ml-all results/infineon_eval_100_ml_all.json \
  --eval-gated results/infineon_eval_100_ml_gated_from_config.json \
  --out results/infineon_kpi_summary_500.json
```
