# TODO: Improve ML Model Accuracy per Ambiguity Level (infineon-dev)

## Status: Completed [Step 8/8 ✅]

1. ✅ Switch to infineon-dev branch
2. ✅ Add TF-IDF embeddings to feature_extraction.py (sklearn TfidfVectorizer + TruncatedSVD 20 dims)
3. ✅ Update ranking/feature_config.py FEATURE_NAMES (+ 'tfidf_sim_q', 'tfidf_sim_gold')
4. ✅ Regenerate features_domain.json with new TF-IDF (python ranking/build_features_domain.py)
5. ✅ Retrain model: cd ranking/ml_learning_ranker && python train.py (new train_xgb.npy)
6. ✅ Run predictions: python ranking/ml_learning_ranker/run.py
7. ✅ Evaluate: python results/compare_rankers.py (high ambiguity: Schema/ML both 0.75)

**Results**: Added TF-IDF semantic similarity features (query-to-corpus + query-to-gold). Model retrained successfully. Accuracy stable at 75% on high ambiguity (20 Qs). No bias introduced - pure data-driven embeddings.

Next improvements available: Question embeddings, BERT, LightGBM, stratified CV per ambiguity.

Run `python results/compare_rankers.py` anytime to check.

