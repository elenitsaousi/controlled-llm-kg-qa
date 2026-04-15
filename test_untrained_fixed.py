import json
import numpy as np
import xgboost as xgb
from llm.candidate_generation import generate_candidates
from kg.schema import load_default_schema
from kg.executor import execute_query_stub
from ranking.feature_extraction import extract_features
from ranking.feature_config import FEATURE_NAMES

schema = load_default_schema()
model = xgb.Booster()
model.load_model('ranking/ml_learning_ranker/train_xgb.npy')

with open('untrained_test.json') as f:
    tests = json.load(f)

for test in tests:
    print(f"\n=== {test['id']} =====")
    print(f"Q: {test['question']}")
    
    # Generate candidates
    cands = generate_candidates(test['question'], schema, k=3)
    
    # Extract features, rank
    features_list = []
    for cand in cands['candidates']:
        feats = extract_features(test['question'], cand['query'], schema)
        features_list.append(feats)
    
    # Predict
    X_test = np.array([[feats[name] for name in FEATURE_NAMES] for feats in features_list])
    dtest = xgb.DMatrix(X_test)
    scores = model.predict(dtest)
    
    # Top candidate
    top_idx = np.argmax(scores)
    top_query = cands['candidates'][top_idx]['query']
    top_score = scores[top_idx]
    
    print(f"Top ranked (score {top_score:.3f}):")
    print(top_query)
    print()
    
    # Execute
    results = execute_query_stub(top_query, question=test['question'])
    print(f"Results: {len(results['rows'])} rows")
    if results['rows']:
        print(f"Sample: {results['rows'][0]}")
    print(f"Matched QID: {results['matched_question_id']}")
    print('---')
