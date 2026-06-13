from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fastapi import HTTPException
import joblib

app = FastAPI(
    title="Lead Conversion Prediction API",
    version="1.0.0",
    description="Predict lead conversion probability using behavioral features."
)

artifact = joblib.load("models/model.pkl")

model = artifact["model"]
feature_names = artifact["feature_names"]
label_encoders = artifact["label_encoders"]


@app.get("/")
def root():
    return {
        "message": "Lead Conversion Prediction API",
        "version": "1.0.0",
        "model": "Random Forest",
        "features_used": 11,
        "endpoints": [
            "/health",
            "/predict",
            "/explain"
        ]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "supported_source_values": list(label_encoders["source"].classes_)
    }


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


def prepare_features(data: PredictionRequest):
    values = data.model_dump()

    # Encode categorical columns
    for col in ["source"]:
        encoder = label_encoders[col]

        try:
            values[col] = int(
                encoder.transform([str(values[col])])[0]
            )
        except ValueError:
            raise ValueError(
                f"Invalid value for '{col}'. "
                f"Allowed values: {list(encoder.classes_)}"
            )

    # Build feature vector in training order
    feature_vector = [
        values[col]
        for col in feature_names
    ]

    return np.array([feature_vector])

class PredictionResponse(BaseModel):
    prediction: str
    conversion_probability: float
    confidence: str
    risk_level: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Prepare features
        X = prepare_features(request)

        # Get probability of conversion
        probability = float(model.predict_proba(X)[0][1])

        prediction = "ACCEPT" if probability >= 0.5 else "REJECT"

        confidence_score = max(probability, 1 - probability)

        if confidence_score >= 0.85:
            confidence = "high"
        elif confidence_score >= 0.60:
            confidence = "medium"
        else:
            confidence = "low"

        if probability >= 0.80:
            risk_level = "low"
        elif probability >= 0.50:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "prediction": prediction,
            "conversion_probability": round(probability, 4),
            "confidence": confidence,
            "risk_level": risk_level,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )
    
class ExplainRequest(BaseModel):
    conversion_probability: float
    session_count: int
    pricing_page_views: int
    webinar_registrations: int
    max_funnel_order: int
    total_events: int

@app.post("/explain")
def explain(request: ExplainRequest):

    reasons = []

    if request.pricing_page_views > 0:
        reasons.append(
            "The lead viewed pricing-related content, indicating purchase intent."
        )

    if request.webinar_registrations > 0:
        reasons.append(
            "The lead registered for a webinar, suggesting active engagement."
        )

    if request.max_funnel_order >= 4:
        reasons.append(
            "The lead progressed to the decision stage of the sales funnel."
        )

    if request.session_count >= 4:
        reasons.append(
            "Multiple sessions indicate continued interest in the product."
        )

    if request.total_events >= 15:
        reasons.append(
            "A high number of interactions reflects strong engagement."
        )

    if not reasons:
        reasons.append(
            "The lead shows moderate activity with limited high-intent signals."
        )

    summary = " ".join(reasons)

    return {
        "summary": summary
    }
