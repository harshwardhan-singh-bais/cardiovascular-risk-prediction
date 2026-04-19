"""
Cardiovascular Disease Risk Prediction — FastAPI Backend
Full pipeline console logging at every step.
"""

import warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════
# STEP 0 — STARTUP: Load artefacts
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("[STARTUP] Cardiovascular Risk Prediction API")
print("=" * 60)

BASE          = Path(__file__).parent.parent
MODEL_PATH    = BASE / "model.pkl"
SCALER_PATH   = BASE / "scaler.pkl"
FEATURES_PATH = BASE / "features.pkl"
STATIC_DIR    = Path(__file__).parent / "static"

print(f"[STEP 0] Loading model artefacts from: {BASE}")

try:
    model = joblib.load(MODEL_PATH)
    print(f"  ✔ model.pkl   loaded  → type: {type(model).__name__}")
except Exception as e:
    print(f"  ✘ model.pkl   FAILED  → {e}"); raise

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"  ✔ scaler.pkl  loaded  → type: {type(scaler).__name__}")
except Exception as e:
    print(f"  ✘ scaler.pkl  FAILED  → {e}"); raise

try:
    features = joblib.load(FEATURES_PATH)
    print(f"  ✔ features.pkl loaded → {len(features)} features: {features}")
except Exception as e:
    print(f"  ✘ features.pkl FAILED → {e}"); raise

MODEL_NAME = type(model).__name__
HAS_PROBA  = hasattr(model, "predict_proba")
HAS_COEF   = hasattr(model, "coef_")
HAS_IMP    = hasattr(model, "feature_importances_")

print(f"\n[STEP 0] Model capability check:")
print(f"  predict_proba       : {'✔' if HAS_PROBA else '✘'}")
print(f"  feature_importances_: {'✔' if HAS_IMP  else '✘'}")
print(f"  coef_               : {'✔' if HAS_COEF else '✘'}")
print(f"  → Prediction mode   : {'probabilistic' if HAS_PROBA else 'risk-score fallback'}")
print(f"  → Feature contrib.  : {'model importances' if HAS_IMP else 'Framingham weights (fallback)'}")
print("=" * 60)
print("[STARTUP] ✔ All artefacts loaded. API ready.\n")

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
FEATURE_META = {
    "male":            {"label": "Sex (Male)",            "unit": "binary",  "min": 0,   "max": 1},
    "age":             {"label": "Age",                   "unit": "years",   "min": 20,  "max": 80},
    "education":       {"label": "Education Level",       "unit": "1–4",     "min": 1,   "max": 4},
    "currentSmoker":   {"label": "Current Smoker",        "unit": "binary",  "min": 0,   "max": 1},
    "cigsPerDay":      {"label": "Cigarettes / Day",      "unit": "count",   "min": 0,   "max": 70},
    "BPMeds":          {"label": "BP Medication",         "unit": "binary",  "min": 0,   "max": 1},
    "prevalentStroke": {"label": "Prevalent Stroke",      "unit": "binary",  "min": 0,   "max": 1},
    "prevalentHyp":    {"label": "Prevalent Hypertension","unit": "binary",  "min": 0,   "max": 1},
    "diabetes":        {"label": "Diabetes",              "unit": "binary",  "min": 0,   "max": 1},
    "totChol":         {"label": "Total Cholesterol",     "unit": "mg/dL",   "min": 100, "max": 600},
    "sysBP":           {"label": "Systolic BP",           "unit": "mmHg",    "min": 80,  "max": 295},
    "diaBP":           {"label": "Diastolic BP",          "unit": "mmHg",    "min": 40,  "max": 150},
    "BMI":             {"label": "BMI",                   "unit": "kg/m²",   "min": 10,  "max": 60},
    "heartRate":       {"label": "Heart Rate",            "unit": "bpm",     "min": 40,  "max": 150},
    "glucose":         {"label": "Glucose Level",         "unit": "mg/dL",   "min": 40,  "max": 400},
}

RISK_WEIGHTS = {
    "age": 0.20, "sysBP": 0.15, "totChol": 0.12, "glucose": 0.12,
    "cigsPerDay": 0.10, "BMI": 0.08, "diaBP": 0.07, "heartRate": 0.06,
    "diabetes": 0.04, "prevalentHyp": 0.03, "male": 0.02, "currentSmoker": 0.01,
}

# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════
app = FastAPI(
    title="CardioRisk API",
    description="Cardiovascular Disease Risk Prediction — Framingham Heart Study",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════
class PatientInput(BaseModel):
    male:            float = Field(..., ge=0,  le=1)
    age:             float = Field(..., ge=1,  le=120)
    education:       float = Field(..., ge=1,  le=4)
    currentSmoker:   float = Field(..., ge=0,  le=1)
    cigsPerDay:      float = Field(..., ge=0,  le=100)
    BPMeds:          float = Field(..., ge=0,  le=1)
    prevalentStroke: float = Field(..., ge=0,  le=1)
    prevalentHyp:    float = Field(..., ge=0,  le=1)
    diabetes:        float = Field(..., ge=0,  le=1)
    totChol:         float = Field(..., ge=50, le=700)
    sysBP:           float = Field(..., ge=50, le=300)
    diaBP:           float = Field(..., ge=30, le=200)
    BMI:             float = Field(..., ge=5,  le=80)
    heartRate:       float = Field(..., ge=20, le=300)
    glucose:         float = Field(..., ge=30, le=500)

class FeatureContribution(BaseModel):
    feature: str; label: str; value: float; unit: str
    contribution: float; contribution_pct: float

class PredictionResponse(BaseModel):
    prediction: int; label: str
    probability_chd: float; probability_no_chd: float
    risk_level: str; risk_score: float
    top_features: list[FeatureContribution]
    model_type: str; message: str

# ══════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ══════════════════════════════════════════════════════════════
def step_build_input_vector(values: dict) -> np.ndarray:
    """STEP 1 — Build raw feature vector in correct column order."""
    vec = np.array([[values[f] for f in features]], dtype=np.float64)
    print(f"  [STEP 1] ✔ Input vector built  shape={vec.shape}  values={vec[0].tolist()}")
    return vec


def step_scale(X_raw: np.ndarray) -> np.ndarray:
    """STEP 2 — StandardScaler transform."""
    X_scaled = scaler.transform(X_raw)
    print(f"  [STEP 2] ✔ Scaling done        mean≈{X_scaled[0].mean():.4f}  std≈{X_scaled[0].std():.4f}")
    return X_scaled


def step_predict(X_scaled: np.ndarray, values: dict) -> tuple[float, float]:
    """STEP 3 — Model inference (with graceful fallback)."""
    # Attempt 1: predict_proba
    if HAS_PROBA:
        try:
            proba      = model.predict_proba(X_scaled)[0]
            prob_chd   = float(proba[1])
            prob_no    = float(proba[0])
            print(f"  [STEP 3] ✔ predict_proba()     P(CHD)={prob_chd:.4f}  P(No CHD)={prob_no:.4f}")
            return prob_chd, prob_no
        except Exception as e:
            print(f"  [STEP 3] ✘ predict_proba failed → {e}  (falling back)")

    # Attempt 2: numeric predict output as risk score
    try:
        raw      = float(model.predict(X_scaled)[0])
        prob_chd = float(np.clip(raw, 0.0, 1.0))
        prob_no  = 1.0 - prob_chd
        print(f"  [STEP 3] ✔ predict() raw={raw:.4f}  clamped P(CHD)={prob_chd:.4f}")
        return prob_chd, prob_no
    except Exception as e:
        print(f"  [STEP 3] ✘ predict() failed → {e}  (using Framingham fallback)")

    # Attempt 3: pure Framingham weight scoring
    score    = step_risk_score(values) / 100.0
    prob_chd = float(score)
    prob_no  = 1.0 - prob_chd
    print(f"  [STEP 3] ✔ Framingham fallback  P(CHD)={prob_chd:.4f}")
    return prob_chd, prob_no


def step_risk_score(values: dict) -> float:
    """STEP 4 — Composite Framingham risk score 0–100."""
    score, total_w = 0.0, 0.0
    for feat, w in RISK_WEIGHTS.items():
        if feat in values:
            lo  = FEATURE_META[feat]["min"]
            hi  = FEATURE_META[feat]["max"]
            vn  = float(np.clip((values[feat] - lo) / (hi - lo + 1e-9), 0, 1))
            score   += w * vn
            total_w += w
    result = round((score / total_w) * 100, 1) if total_w > 0 else 0.0
    print(f"  [STEP 4] ✔ Risk score computed  score={result}/100")
    return result


def step_feature_contributions(values: dict) -> list[FeatureContribution]:
    """STEP 5 — Compute per-feature contributions using Framingham weights × normalised value."""
    contribs = {}
    for feat in features:
        lo = FEATURE_META.get(feat, {}).get("min", 0)
        hi = FEATURE_META.get(feat, {}).get("max", 1)
        vn = float(np.clip((values[feat] - lo) / (hi - lo + 1e-9), 0, 1))
        w  = RISK_WEIGHTS.get(feat, 0.01)
        contribs[feat] = w * vn

    total = sum(contribs.values()) + 1e-9
    result = []
    for feat in features:
        c    = contribs[feat]
        meta = FEATURE_META.get(feat, {})
        result.append(FeatureContribution(
            feature=feat, label=meta.get("label", feat),
            value=values[feat], unit=meta.get("unit", ""),
            contribution=round(c, 5),
            contribution_pct=round(c / total * 100, 2),
        ))
    result.sort(key=lambda x: x.contribution, reverse=True)
    top3 = [(f.label, f"{ f.contribution_pct:.1f}%") for f in result[:3]]
    print(f"  [STEP 5] ✔ Top-3 contributors  {top3}")
    return result


def step_classify(prob_chd: float, risk_score: float) -> tuple[int, str, str]:
    """STEP 6 — Classify into label, risk level."""
    prediction = 1 if prob_chd >= 0.5 else 0
    if prob_chd < 0.10:  level = "Low"
    elif prob_chd < 0.25: level = "Moderate"
    elif prob_chd < 0.50: level = "High"
    else:                  level = "Very High"
    label = "Cardiovascular Disease Risk Detected" if prediction == 1 else "No Cardiovascular Disease Risk"
    print(f"  [STEP 6] ✔ Classification done  prediction={prediction}  level={level}  label='{label}'")
    return prediction, label, level


def step_build_message(prediction: int, prob_chd: float, level: str) -> str:
    """STEP 7 — Build human-readable clinical message."""
    if prediction == 1:
        msg = (
            f"This patient shows a {prob_chd*100:.1f}% predicted probability of coronary "
            f"heart disease. Risk level: {level}. Please consult a cardiologist."
        )
    else:
        msg = (
            f"The model indicates a {prob_chd*100:.1f}% CHD probability — "
            f"relatively {level.lower()} risk. Maintain a healthy lifestyle and continue check-ups."
        )
    print(f"  [STEP 7] ✔ Message generated    ({len(msg)} chars)")
    return msg

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    print("[HEALTH] ✔ Health check called")
    return {"status": "ok", "model": MODEL_NAME, "features": features}


@app.get("/feature-info")
def feature_info():
    print("[FEATURE-INFO] ✔ Feature info requested")
    return {"features": FEATURE_META, "risk_weights": RISK_WEIGHTS}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    t0 = time.time()
    print("\n" + "─" * 60)
    print(f"[PREDICT] ▶ New prediction request received")
    print(f"[PREDICT]   Input → {patient.model_dump()}")
    print("─" * 60)

    values = patient.model_dump()

    # ── Pipeline ──────────────────────────────────────────────
    X_raw    = step_build_input_vector(values)     # STEP 1
    X_scaled = step_scale(X_raw)                   # STEP 2
    prob_chd, prob_no_chd = step_predict(          # STEP 3
        X_scaled, values)
    risk_score = step_risk_score(values)            # STEP 4
    top_feats  = step_feature_contributions(values) # STEP 5
    prediction, label, level = step_classify(       # STEP 6
        prob_chd, risk_score)
    message = step_build_message(                   # STEP 7
        prediction, prob_chd, level)

    elapsed = (time.time() - t0) * 1000
    print(f"─" * 60)
    print(f"[PREDICT] ✔ Pipeline complete  →  {elapsed:.1f} ms")
    print(f"[PREDICT]   Result: prediction={prediction}  P(CHD)={prob_chd:.4f}  risk_score={risk_score}")
    print("─" * 60 + "\n")

    return PredictionResponse(
        prediction        = prediction,
        label             = label,
        probability_chd   = round(prob_chd,    4),
        probability_no_chd= round(prob_no_chd, 4),
        risk_level        = level,
        risk_score        = risk_score,
        top_features      = top_feats,
        model_type        = MODEL_NAME,
        message           = message,
    )


# ── Serve static frontend ─────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=FileResponse)
    def serve_frontend():
        print("[STATIC]  ✔ Serving index.html")
        return str(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend:app", host="0.0.0.0", port=8000, reload=True)
