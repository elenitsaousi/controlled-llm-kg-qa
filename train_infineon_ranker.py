# train_infineon_ranker.py
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
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
            row = [float(features.get(name, 0)) for name in FEATURE_NAMES]
            X.append(row)
            y.append(c["is_correct"])
            groups.append(qid)
    return np.array(X), np.array(y), groups


def train_and_evaluate(X, y, groups):
    unique_groups = list(set(groups))
    groups = np.array(groups)

    print(f"Total samples: {len(X)}")
    print(f"Correct: {int(sum(y))} ({sum(y)/len(y)*100:.1f}%)")
    print(f"Questions: {len(unique_groups)}")

    results_logistic = []
    results_xgb = []

    for test_qid in unique_groups:
        train_mask = groups != test_qid
        test_mask = groups == test_qid

        if sum(test_mask) == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if sum(y_train) == 0:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        any_correct = int(sum(y_test) > 0)

        # Logistic [7 - 0.4.4]
        log_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0
        )
        log_model.fit(X_train_s, y_train)
        log_scores = log_model.predict_proba(X_test_s)[:, 1]
        log_top1 = int(y_test[np.argmax(log_scores)] == 1)
        results_logistic.append({
            "qid": test_qid,
            "top1_correct": log_top1,
            "any_correct": any_correct
        })

        # XGBoost [7 - 0.4.5]
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_pos = neg / pos if pos > 0 else 1.0

        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            verbosity=0,
        )
        xgb_model.fit(X_train_s, y_train)
        xgb_scores = xgb_model.predict_proba(X_test_s)[:, 1]
        xgb_top1 = int(y_test[np.argmax(xgb_scores)] == 1)
        results_xgb.append({
            "qid": test_qid,
            "top1_correct": xgb_top1,
            "any_correct": any_correct
        })

    # Summary
    print(f"\n=== CV RESULTS (Leave-One-Question-Out) ===")
    for name, results in [("Logistic", results_logistic), ("XGBoost", results_xgb)]:
        evaluated = len(results)
        top1 = sum(r["top1_correct"] for r in results)
        any_c = sum(r["any_correct"] for r in results)
        print(f"\n{name}:")
        print(f"  Top1 correct: {top1}/{evaluated} ({top1/evaluated*100:.1f}%)")
        print(f"  Any correct:  {any_c}/{evaluated} ({any_c/evaluated*100:.1f}%)")

    return results_logistic, results_xgb


def train_final_model(X, y):
    """Train final XGBoost model on all data. [7 - 0.4.5]"""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    scale_pos = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        verbosity=0,
    )
    model.fit(X_s, y)

    joblib.dump({"model": model, "scaler": scaler},
                "ranking/models/infineon_ranker.joblib")
    print("\nModel saved to: ranking/models/infineon_ranker.joblib")

    # Feature importance
    print("\n=== FEATURE IMPORTANCE ===")
    importances = model.feature_importances_
    for name, imp in sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        print(f"  {name}: {imp:.3f}")

    return model, scaler


if __name__ == "__main__":
    X, y, groups = load_training_data("ranking/infineon_training_data.json")
    y = np.array(y)

    results_log, results_xgb = train_and_evaluate(X, y, groups)

    model, scaler = train_final_model(X, y)