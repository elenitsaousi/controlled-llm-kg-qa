# Infineon Ranker Workflow (Unbiased + Deployment)

## 1) Build candidate-level labeled training data (100 benchmark questions)

```bash
python ranking/build_infineon_training_data.py \
  --dataset data/infineon/infineon_dataset_100.json \
  --graph data/infineon/graph.ttl \
  --schema data/infineon/schema.json \
  --out ranking/infineon_training_data_100.json \
  --k 5 \
  --n-runs 3
```

This creates `ranking/infineon_training_data_100.json` with:
- generated candidate queries per question
- execution-based correctness labels (`is_correct`)
- ambiguity labels and family grouping metadata

## 2) Train + evaluate unbiased with grouped stratified 5-fold CV

```bash
python ranking/train_infineon_np_tfidf_ranker.py \
  --training-data ranking/infineon_training_data_100.json \
  --cv-out results/infineon_ranker_cv_100.json \
  --model-out ranking/models/infineon_np_tfidf_ranker.json \
  --folds 5
```

What this does:
- Uses grouped/stratified folds (by question family + ambiguity label)
- Uses only train-fold data to fit TF-IDF and model
- Produces out-of-fold metrics over all 100 questions (unbiased)
- Trains final deployment model on all 100 afterward

## 3) Re-run LLM benchmark with ML ranking enabled (deployment check)

```bash
python evaluation/run_infineon_100_eval.py \
  --dataset data/infineon/infineon_dataset_100.json \
  --use-ml-ranking \
  --ml-model ranking/models/infineon_np_tfidf_ranker.json \
  --out results/infineon_eval_100_with_np_ml.json \
  --progress
```

This is the practical runtime evaluation after integrating ML reranking.

## Notes

- Unbiased metric for scientific reporting: `results/infineon_ranker_cv_100.json`
- Deployment metric for practical improvement: `results/infineon_eval_100_with_np_ml.json`
- If you want sentence-transformer semantic features, set:
  `ENABLE_SEMANTIC_EMBEDDING=1`
  (default is off to avoid dependency issues and leakage noise).
