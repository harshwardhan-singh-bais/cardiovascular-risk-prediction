"""
fix_model.py — Re-saves model.pkl as RandomForestClassifier
============================================================
Run this once if model.pkl is a Lasso / wrong model.
It reuses the same training pipeline from the notebook.

Usage:
    python fix_model.py           (expects heart_disease.csv in same dir)
    python fix_model.py --check   (just inspect current pkl files)
"""

import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).parent

# ── 1. Inspect current PKL ─────────────────────────────────
model   = joblib.load(BASE / "model.pkl")
scaler  = joblib.load(BASE / "scaler.pkl")
features= joblib.load(BASE / "features.pkl")

print(f"[INFO] Current model  : {type(model).__name__}")
print(f"[INFO] Has predict_proba: {hasattr(model, 'predict_proba')}")
print(f"[INFO] Has feature_importances_: {hasattr(model, 'feature_importances_')}")
print(f"[INFO] Features ({len(features)}): {features}")

if "--check" in sys.argv:
    sys.exit(0)

if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
    print("[OK] model.pkl is already a tree classifier with predict_proba. No fix needed.")
    sys.exit(0)

# ── 2. Need to retrain ─────────────────────────────────────
csv_path = BASE / "heart_disease.csv"
if not csv_path.exists():
    print(f"[ERROR] {csv_path} not found. Cannot retrain.")
    print("  Please place heart_disease.csv in the project root and re-run.")
    sys.exit(1)

print(f"\n[FIX] Retraining RandomForestClassifier on {csv_path} ...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

CLF_TARGET  = "TenYearCHD"
FEATURE_COLS= [
    "male","age","education","currentSmoker","cigsPerDay",
    "BPMeds","prevalentStroke","prevalentHyp","diabetes",
    "totChol","sysBP","diaBP","BMI","heartRate","glucose"
]
BINARY_COLS = ["male","currentSmoker","BPMeds","prevalentStroke","prevalentHyp","diabetes"]

df = pd.read_csv(csv_path)
print(f"  Loaded: {df.shape}  CLF distribution: {df[CLF_TARGET].value_counts().to_dict()}")

# Impute
for col in df.columns:
    if df[col].isnull().sum() == 0: continue
    if col in BINARY_COLS:
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# IQR outlier removal
def remove_iqr(data, cols):
    d = data.copy()
    for col in cols:
        if col not in d.columns: continue
        Q1, Q3 = d[col].quantile(0.25), d[col].quantile(0.75)
        IQR = Q3 - Q1
        d = d[(d[col] >= Q1-1.5*IQR) & (d[col] <= Q3+1.5*IQR)]
    return d.reset_index(drop=True)

cont_cols = [c for c in FEATURE_COLS if c not in BINARY_COLS and c in df.columns]
df = remove_iqr(df, cont_cols)

FEAT_AVAIL = [c for c in FEATURE_COLS if c in df.columns]
X = df[FEAT_AVAIL].values.astype(np.float32)
y = df[CLF_TARGET].values.astype(int)

# Scale + SMOTE
new_scaler = StandardScaler()
X_scaled = new_scaler.fit_transform(X)

smote = SMOTE(random_state=42, k_neighbors=5)
X_syn, y_syn = smote.fit_resample(X_scaled, y)
print(f"  After SMOTE: {dict(pd.Series(y_syn).value_counts().sort_index())}")

# Train RF
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_syn, y_syn)

# Quick eval on original test split
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
from sklearn.metrics import accuracy_score, f1_score
preds = rf.predict(X_te)
print(f"  Test Accuracy: {accuracy_score(y_te, preds):.4f}  F1: {f1_score(y_te, preds, zero_division=0):.4f}")

# Save
joblib.dump(rf, BASE / "model.pkl")
joblib.dump(new_scaler, BASE / "scaler.pkl")
joblib.dump(FEAT_AVAIL, BASE / "features.pkl")
print(f"\n[DONE] Saved RandomForestClassifier → model.pkl")
print(f"       Features: {FEAT_AVAIL}")
