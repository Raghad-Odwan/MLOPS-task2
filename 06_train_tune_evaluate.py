# Notebook 6: Train, tune, evaluate.
# Start with a simple baseline, tune a real model on validation, touch test only once.

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.dummy import DummyClassifier

train = pd.read_csv("features_train.csv")
val = pd.read_csv("features_val.csv")
test = pd.read_csv("features_test.csv")

X_train = train.drop(columns=["is_late"])
y_train = train["is_late"]
X_val = val.drop(columns=["is_late"])
y_val = val["is_late"]
X_test = test.drop(columns=["is_late"])
y_test = test["is_late"]

# Because the classes are imbalanced (most orders are on time), accuracy alone
# would be misleading. Use F1 score and ROC-AUC on the late (minority) class instead.

# 1. Simple baseline: always predict the majority class
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_val)
print("Baseline F1 (validation):", f1_score(y_val, baseline_pred))

# 2. Train a real model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_proba = model.predict_proba(X_val)[:, 1]

print("\nModel performance on validation:")
print("F1:", f1_score(y_val, val_pred))
print("Precision:", precision_score(y_val, val_pred))
print("Recall:", recall_score(y_val, val_pred))
print("ROC-AUC:", roc_auc_score(y_val, val_proba))
print(classification_report(y_val, val_pred))

# 3. Simple manual tuning: try a couple of max_depth values, pick the best by validation F1
best_score = -1
best_model = None
best_params = None
for depth in [6, 10, 15]:
    candidate = RandomForestClassifier(
        n_estimators=200, max_depth=depth, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    candidate.fit(X_train, y_train)
    pred = candidate.predict(X_val)
    score = f1_score(y_val, pred)
    print(f"max_depth={depth} -> validation F1={score:.4f}")
    if score > best_score:
        best_score = score
        best_model = candidate
        best_params = {"max_depth": depth}

print("\nBest params:", best_params, "with validation F1:", best_score)

# 4. Touch the test set once, at the very end, with the best model only
test_pred = best_model.predict(X_test)
test_proba = best_model.predict_proba(X_test)[:, 1]

print("\nFinal performance on test set:")
print("F1:", f1_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred))
print("Recall:", recall_score(y_test, test_pred))
print("ROC-AUC:", roc_auc_score(y_test, test_proba))
print(classification_report(y_test, test_pred))

# Save artifacts: the trained model and a results summary
joblib.dump(best_model, "final_model.joblib")

with open("results_summary.txt", "w") as f:
    f.write(f"Baseline F1 (validation, majority class): {f1_score(y_val, baseline_pred):.4f}\n")
    f.write(f"Best params: {best_params}\n")
    f.write(f"Validation F1: {best_score:.4f}\n")
    f.write(f"Test F1: {f1_score(y_test, test_pred):.4f}\n")
    f.write(f"Test Precision: {precision_score(y_test, test_pred):.4f}\n")
    f.write(f"Test Recall: {recall_score(y_test, test_pred):.4f}\n")
    f.write(f"Test ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}\n")

print("\nArtifacts saved: final_model.joblib, results_summary.txt")
