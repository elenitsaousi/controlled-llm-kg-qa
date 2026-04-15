# train_infineon_ranker.py
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from ranking.feature_config import FEATURE_NAMES

def load_training_data(path):
    with open(path) as f:
        data = json.load(f)
    
    X, y, groups = [], [], []
    
    for qid, candidates in data.items():
        for c in candidates:
            features = c.get("features", {})
            if not features:
                continue
            
            # Feature vector
            row = [float(features.get(name, 0)) for name in FEATURE_NAMES]
            X.append(row)
            y.append(c["is_correct"])
            groups.append(qid)
    
    return np.array(X), np.array(y), groups

def train_and_evaluate(X, y, groups):
    unique_groups = list(set(groups))
    groups = np.array(groups)
    
    print(f"Total samples: {len(X)}")
    print(f"Correct: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"Questions: {len(unique_groups)}")
    
    # Leave-one-question-out CV
    results = []
    
    for test_qid in unique_groups:
        train_mask = groups != test_qid
        test_mask = groups == test_qid
        
        if sum(test_mask) == 0:
            continue
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Skip if no positive samples in train
        if sum(y_train) == 0:
            continue
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Train
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0
        )
        model.fit(X_train_s, y_train)
        
        # Predict
        scores = model.predict_proba(X_test_s)[:, 1]
        best_idx = np.argmax(scores)
        
        # Check if top1 is correct
        top1_correct = int(y_test[best_idx] == 1)
        any_correct = int(sum(y_test) > 0)
        
        results.append({
            "qid": test_qid,
            "top1_correct": top1_correct,
            "any_correct": any_correct
        })
    
    # Summary
    evaluated = len(results)
    top1 = sum(r["top1_correct"] for r in results)
    any_c = sum(r["any_correct"] for r in results)
    
    print(f"\n=== CV RESULTS (Leave-One-Question-Out) ===")
    print(f"Evaluated questions: {evaluated}")
    print(f"Top1 correct: {top1}/{evaluated} ({top1/evaluated*100:.1f}%)")
    print(f"Any correct: {any_c}/{evaluated} ({any_c/evaluated*100:.1f}%)")
    
    return results

def train_final_model(X, y):
    """Train final model on all data."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0
    )
    model.fit(X_s, y)
    
    # Save
    joblib.dump({"model": model, "scaler": scaler}, 
                "ranking/models/infineon_ranker.joblib")
    print("\nModel saved to: ranking/models/infineon_ranker.joblib")
    
    return model, scaler

if __name__ == "__main__":
    # Load data
    X, y, groups = load_training_data("ranking/infineon_training_data.json")
    
    # Evaluate with CV
    results = train_and_evaluate(X, y, groups)
    
    # Train final model
    model, scaler = train_final_model(X, y)
    
    print("\n=== FEATURE IMPORTANCE ===")
    for name, coef in sorted(
        zip(FEATURE_NAMES, model.coef_[0]), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:10]:
        print(f"  {name}: {coef:.3f}")