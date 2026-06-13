from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import shap

app = FastAPI(
    title="Lead Conversion Prediction API",
    version="2.0.0",
    description="Predict lead conversion probability with SHAP-based explanations."
)

# ── Load model artifact ───────────────────────────────────────────────────────
artifact       = joblib.load("models/model.pkl")
model          = artifact["model"]
feature_names  = artifact["feature_names"]
label_encoders = artifact["label_encoders"]

# ── Build SHAP explainer once at startup (not per request) ───────────────────
# TreeExplainer works directly with Random Forest — fast and exact
explainer = shap.TreeExplainer(model)

FUNNEL_LABELS = {1: "Awareness", 2: "Consideration", 3: "Evaluation", 4: "Decision"}


# SCHEMAS

class PredictionRequest(BaseModel):
    source: str

    session_count: int
    total_events: int
    total_time_seconds: float

    pricing_page_views: int
    webinar_registrations: int

    max_funnel_order: int
    total_clicks: int

    engagement_span_days: float
    days_since_last_visit: float
    avg_session_gap_days: float

    class Config:
        json_schema_extra = {
            "example": {
                "source": "Google",
                "session_count": 7,
                "total_events": 35,
                "total_time_seconds": 2400,
                "pricing_page_views": 3,
                "webinar_registrations": 1,
                "max_funnel_order": 4,
                "total_clicks": 60,
                "engagement_span_days": 25,
                "days_since_last_visit": 5,
                "avg_session_gap_days": 4
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    conversion_probability: float
    confidence: str
    risk_level: str


class ExplainRequest(BaseModel):
    source: str

    session_count: int
    total_events: int
    total_time_seconds: float

    pricing_page_views: int
    webinar_registrations: int

    max_funnel_order: int
    total_clicks: int

    engagement_span_days: float
    days_since_last_visit: float
    avg_session_gap_days: float

    class Config:
        json_schema_extra = {
            "example": {
                "source": "Google",
                "session_count": 7,
                "total_events": 35,
                "total_time_seconds": 2400,
                "pricing_page_views": 3,
                "webinar_registrations": 1,
                "max_funnel_order": 4,
                "total_clicks": 60,
                "engagement_span_days": 25,
                "days_since_last_visit": 5,
                "avg_session_gap_days": 4
            }
        }


class ShapFactor(BaseModel):
    feature: str        # e.g. "total_time_seconds"
    value: str        # actual input value e.g. 2400.0
    shap_value: float   # how much this feature pushed the score up/down
    direction: str      # "increases" or "decreases"
    human_label: str    # plain English e.g. "Time on site"


class ExplainResponse(BaseModel):
    conversion_probability: float
    prediction: str
    summary: str
    top_factors: list[ShapFactor]   # top 5 features by absolute SHAP value


# HELPERS
# Human-readable labels for each feature name
FEATURE_LABELS = {
    "source":                 "Acquisition source",
    "session_count":          "Number of sessions",
    "total_events":           "Total interactions",
    "total_time_seconds":     "Time on site (seconds)",
    "pricing_page_views":     "Pricing page visits",
    "webinar_registrations":  "Webinar registrations",
    "max_funnel_order":       "Deepest funnel stage reached",
    "total_clicks":           "Total clicks",
    "engagement_span_days":   "Days active on site",
    "days_since_last_visit":  "Days since last visit",
    "avg_session_gap_days":   "Average days between sessions",
}


def prepare_features(data) -> np.ndarray:
    """Convert request object into numpy array matching training feature order."""
    values = data.model_dump()

    # Encode categorical
    for col in ["source"]:
        encoder = label_encoders[col]
        try:
            values[col] = int(encoder.transform([str(values[col])])[0])
        except ValueError:
            raise ValueError(
                f"Invalid value for '{col}'. "
                f"Allowed values: {list(encoder.classes_)}"
            )

    feature_vector = [values[col] for col in feature_names]
    return np.array([feature_vector], dtype=float)


def build_summary(probability: float, top_factors: list[ShapFactor],
                  request) -> str:
    
    funnel_name = FUNNEL_LABELS.get(request.max_funnel_order, "Unknown")

    # Lead the sentence with the score
    if probability >= 0.75:
        opening = f"This lead has a strong conversion probability of {probability:.0%}."
    elif probability >= 0.45:
        opening = f"This lead has a moderate conversion probability of {probability:.0%}."
    else:
        opening = f"This lead has a low conversion probability of {probability:.0%}."

    # Pick top 3 positive drivers (shap_value > 0)
    positive = [f for f in top_factors if f.shap_value > 0][:3]
    negative = [f for f in top_factors if f.shap_value < 0][:1]

    driver_parts = []
    for f in positive:
        driver_parts.append(f"{f.human_label} ({f.value})")

    if driver_parts:
        drivers = "Key positive signals: " + ", ".join(driver_parts) + "."
    else:
        drivers = "No strong positive signals detected."

    risk_parts = []
    for f in negative:
        risk_parts.append(f"{f.human_label} ({f.value})")

    risks = f"Main drag: {', '.join(risk_parts)}." if risk_parts else ""

    funnel_note = f"The lead reached the {funnel_name} stage of the funnel."

    return " ".join(filter(None, [opening, funnel_note, drivers, risks]))



# ENDPOINTS --------------------------------------

@app.get("/")
def root():
    return {
        "message": "Lead Conversion Prediction API",
        "version": "2.0.0",
        "model": "Random Forest",
        "features_used": len(feature_names),
        "explainability": "SHAP TreeExplainer",
        "endpoints": ["/health", "/predict", "/explain"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "supported_source_values": list(label_encoders["source"].classes_)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict conversion probability for a lead.

    Returns probability, prediction label (ACCEPT/REJECT),
    confidence level, and risk level.
    """
    try:
        X = prepare_features(request)
        probability = float(model.predict_proba(X)[0][1])

        prediction      = "ACCEPT" if probability >= 0.5 else "REJECT"
        confidence_score = max(probability, 1 - probability)
        confidence      = "high" if confidence_score >= 0.85 else "medium" if confidence_score >= 0.60 else "low"
        risk_level      = "low"  if probability >= 0.80 else "medium" if probability >= 0.50 else "high"

        return {
            "prediction":             prediction,
            "conversion_probability": round(probability, 4),
            "confidence":             confidence,
            "risk_level":             risk_level,
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    """
    Explain the conversion prediction using SHAP values.

    Returns:
    - **top_factors**: top 5 features ranked by absolute SHAP value,
      each with direction (increases/decreases), actual value, and human label.
    - **summary**: plain-English paragraph explaining the score.
    - **base_probability**: the model's average prediction (SHAP baseline).
    """
    try:
        X = prepare_features(request)

        # ── SHAP computation ─────────────────────────────────────────────────
        # shap_values shape: (n_classes, n_samples, n_features)
        # We want class 1 (converted), sample 0
        shap_values = explainer.shap_values(X)

        # For RandomForest with sklearn: shap_values is a list [class0, class1]
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            # Older SHAP versions
            sv = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            # Newer SHAP versions
            sv = shap_values[0, :, 1]
        else:
            # Fallback
            sv = shap_values[0]

        # Base value (expected value for class 1)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_val = float(explainer.expected_value[1])
        else:
            base_val = float(explainer.expected_value)

        # ── Model probability ────────────────────────────────────────────────
        probability = float(model.predict_proba(X)[0][1])
        prediction  = "ACCEPT" if probability >= 0.5 else "REJECT"

        # ── Build top factors ────────────────────────────────────────────────
        raw_values = X[0]  # actual input values

        # Sort features by absolute SHAP value descending
        sorted_idx = np.argsort(np.abs(sv))[::-1]

        top_factors = []
        for idx in sorted_idx[:5]:
            feat_name = feature_names[idx]
            shap_val = float(sv[idx])

            # Decode categorical feature
            if feat_name == "source":
                input_val = label_encoders["source"].inverse_transform(
                    [int(raw_values[idx])]
                )[0]
            else:
                input_val = float(raw_values[idx])

            top_factors.append(
                ShapFactor(
                    feature=feat_name,
                    value=str(input_val),   # <-- store as string
                    shap_value=round(shap_val, 4),
                    direction="increases" if shap_val > 0 else "decreases",
                    human_label=FEATURE_LABELS.get(feat_name, feat_name),
                )
            )
        # ── Build summary ────────────────────────────────────────────────────
        summary = build_summary(probability, top_factors, request)

        return ExplainResponse(
            conversion_probability = round(probability, 4),
            prediction             = prediction,
            summary                = summary,
            top_factors            = top_factors,
            
        )

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
