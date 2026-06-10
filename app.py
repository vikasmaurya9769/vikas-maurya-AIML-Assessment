from fastapi import FastAPI
from pydantic import BaseModel
import math
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
        "message": "Lead Conversion Prediction API is running",
        "model_loaded": model is not None,
        "num_features": len(feature_names)
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }



from pydantic import BaseModel

class PredictionRequest(BaseModel):
    source: str
    company_size: str

    session_count: int
    total_events: int
    total_time_seconds: float

    pricing_page_views: int
    webinar_registrations: int

    max_funnel_order: int
    return_visitor_flag: bool

    total_clicks: int

import math
import numpy as np

import numpy as np

def prepare_features(data: PredictionRequest):
    values = data.model_dump()

    # Encode categorical columns
    for col in ["source", "company_size"]:
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

    # Convert bool → int
    values["return_visitor_flag"] = int(
        values["return_visitor_flag"]
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
        elif confidence_score >= 0.70:
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