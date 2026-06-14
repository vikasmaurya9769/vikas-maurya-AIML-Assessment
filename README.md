cat > /mnt/user-data/outputs/README.md << 'ENDOFFILE'
# 🎯 Lead Conversion Prediction System
### AI/ML Engineer Assessment — Vynqe
**Developed by:** [Vikas Maurya](https://github.com/vikasmaurya)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://shap.readthedocs.io/)
[![Render](https://img.shields.io/badge/Deployed-Render-46E3B7.svg)](https://vikas-maurya-aiml-assessment.onrender.com)

---

## 🚀 Live Demo

| | |
|---|---|
| **Base URL** | https://vikas-maurya-aiml-assessment.onrender.com |
| **Interactive Docs** | https://vikas-maurya-aiml-assessment.onrender.com/docs |
| **Health Check** | https://vikas-maurya-aiml-assessment.onrender.com/health |

> 💡 **Note:** Hosted on Render's free tier. The first request may take 20–30 seconds to wake up after inactivity. Subsequent requests are fast.

---

## 📋 Table of Contents

- [Overview](#overview)
- [What's Inside](#whats-inside)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Local Setup](#-local-setup)
- [Model Training](#-model-training)
- [Running the API](#-running-the-api)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Key EDA Findings](#-key-eda-findings)
- [Data Leakage Prevention](#-data-leakage-prevention)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Author](#-author)

---

## Overview

This project addresses a real B2B sales problem: **given a lead's firmographic profile and behavioral activity on the platform, how likely are they to convert into a paying customer?**

The system combines exploratory data analysis, temporal feature engineering, multi-model comparison, and a FastAPI inference layer with SHAP-based explainability — delivering real-time conversion scores alongside human-readable explanations of *why* a lead scored the way it did.

---

## What's Inside

- Full EDA with 10+ visualizations (`notebooks/EDA.ipynb`, `analysis.md`)
- Custom target variable derived from high-intent behavioral events (no pre-labeled data)
- **31 engineered features** from both datasets, including 3 temporal features extracted from timestamps
- Comparison of 4 ML models with automatic best-model selection by F1-score
- **SHAP TreeExplainer** integrated into the `/explain` endpoint — per-prediction feature attribution, not rule-based logic
- FastAPI REST service deployed on Render with full Pydantic validation
- Feature importance, confusion matrix, and ROC curve outputs

---

## 📂 Project Structure

```
vikas-maurya-AIML-Assessment/
├── README.md
├── analysis.md                   ← Full EDA write-up
├── requirements.txt
├── train.py                      ← End-to-end training pipeline
├── app.py                        ← FastAPI application with SHAP
├── data/
│   ├── leads (1).csv             ← (not committed — add locally)
│   └── interactions (1).csv      ← (not committed — add locally)
├── models/
│   └── model.pkl                 ← Serialized best model + encoders
├── notebooks/
│   └── EDA.ipynb                 ← Interactive data exploration
├── outputs/
│   ├── eda/
│   │   ├── conversion_distribution.png
│   │   ├── conversion_by_source.png
│   │   ├── conversion_by_segment.png
│   │   ├── funnel_conversion.png
│   │   ├── session_behavior.png
│   │   ├── monthly_patterns.png
│   │   ├── temporal_features.png
│   │   └── correlation_heatmap.png
│   └── model/
│       ├── model_metrics.json
│       ├── feature_importance.png
│       ├── confusion_matrix.png
│       └── roc_curve.png
└── .gitignore
```

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | Pandas, NumPy, Matplotlib, Seaborn |
| ML | Scikit-learn, XGBoost |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI, Uvicorn, Pydantic |
| Serialization | Joblib |
| Deployment | Render |

---

## 💻 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/vikasmaurya/vikas-maurya-AIML-Assessment.git
cd vikas-maurya-AIML-Assessment
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add data files

Place the provided CSV files in the `data/` folder:

```
data/
├── leads (1).csv
└── interactions (1).csv
```

These files are excluded from the repository via `.gitignore`.

---

## 🏋️ Model Training

```bash
python train.py
```

This will:

1. Load both CSV files
2. Derive the `converted` target label from high-intent interaction events
3. Engineer **31 features** including 3 temporal features from timestamps
4. Train Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
5. Select the best model by F1-score
6. Save the model artifact to `models/model.pkl`
7. Save all evaluation outputs to `outputs/model/`

**Expected terminal output:**

```
12:50:31 [INFO] Loading leads from data/leads (1).csv
12:50:31 [INFO] Leads shape: (2045, 21) | Interactions shape: (40000, 36)
12:50:31 [INFO] Conversion rate: 31.8% (644 / 2025 leads)
12:50:32 [INFO] Created 31 engineered features
12:50:32 [INFO] Final feature count: 11 | Samples: 2025
12:50:32 [INFO] Train shape: (1620, 11) | Test shape: (405, 11)
12:50:34 [INFO] Best model: Random Forest (F1=0.889)
12:50:36 [INFO] CV F1 scores: [0.912 0.892 0.899 0.893 0.925] | Mean: 0.904 ± 0.013
12:50:36 [INFO] Saved model artifact successfully
```

---

## 🌐 Running the API

```bash
python -m uvicorn app:app --reload
```

Local docs: `http://127.0.0.1:8000/docs`

---

## 📡 API Reference

All endpoints are live at `https://vikas-maurya-aiml-assessment.onrender.com`

---

### `GET /`

Returns basic API information.

🔗 https://vikas-maurya-aiml-assessment.onrender.com/

```json
{
  "message": "Lead Conversion Prediction API",
  "version": "2.0.0",
  "model": "Random Forest",
  "features_used": 11,
  "explainability": "SHAP TreeExplainer",
  "endpoints": ["/health", "/predict", "/explain"]
}
```

---

### `GET /health`

Returns model and explainer status.

🔗 https://vikas-maurya-aiml-assessment.onrender.com/health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "explainer_loaded": true,
  "model_type": "RandomForestClassifier",
  "feature_count": 11,
  "supported_source_values": [
    "Direct", "Email Campaign", "Facebook",
    "Google", "Instagram", "LinkedIn",
    "Organic Search", "Referral"
  ]
}
```

---

### `POST /predict`

Predicts conversion probability for a lead.

🔗 https://vikas-maurya-aiml-assessment.onrender.com/docs#/default/predict_predict_post

**Request:**

```json
{
  "source": "LinkedIn",
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
```

**Response:**

```json
{
  "prediction": "ACCEPT",
  "conversion_probability": 0.8214,
  "confidence": "high",
  "risk_level": "low"
}
```

**Input fields:**

| Field | Type | Description |
|-------|------|-------------|
| `source` | str | Acquisition channel — Google, LinkedIn, Facebook, Instagram, Direct, Referral, Email Campaign, Organic Search |
| `session_count` | int | Total number of sessions |
| `total_events` | int | Total interaction events across all sessions |
| `total_time_seconds` | float | Total time spent on site in seconds |
| `pricing_page_views` | int | Number of visits to the Pricing page |
| `webinar_registrations` | int | Webinar registration event count |
| `max_funnel_order` | int | Deepest funnel stage — 1=Awareness, 2=Consideration, 3=Evaluation, 4=Decision |
| `total_clicks` | int | Total click events |
| `engagement_span_days` | float | Days between first and last visit |
| `days_since_last_visit` | float | Days since the most recent visit |
| `avg_session_gap_days` | float | Average days between sessions |

---

### `POST /explain`

Explains the prediction using **SHAP values** — not hand-coded rules.

🔗 https://vikas-maurya-aiml-assessment.onrender.com/docs#/default/explain_explain_post

Takes the same input fields as `/predict`.

**Request:**

```json
{
  "source": "LinkedIn",
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
```

**Response:**

```json
{
  "conversion_probability": 0.8214,
  "prediction": "ACCEPT",
  "summary": "This lead has a strong conversion probability of 82%. The lead reached the Decision stage of the funnel. Key positive signals: Webinar registrations (1.0), Time on site (2400.0), Total clicks (60.0).",
  "top_factors": [
    {
      "feature": "webinar_registrations",
      "value": "1.0",
      "shap_value": 0.1423,
      "direction": "increases",
      "human_label": "Webinar registrations"
    },
    {
      "feature": "total_time_seconds",
      "value": "2400.0",
      "shap_value": 0.1187,
      "direction": "increases",
      "human_label": "Time on site (seconds)"
    },
    {
      "feature": "days_since_last_visit",
      "value": "5.0",
      "shap_value": -0.0312,
      "direction": "decreases",
      "human_label": "Days since last visit"
    }
  ]
}
```

The `top_factors` array ranks the 5 most influential features for this specific prediction by absolute SHAP value. Each factor shows the actual input value, the SHAP contribution, and whether it pushed the score up or down. This is model-level attribution — different leads will get different explanations even with similar scores.

---

## 📊 Model Performance

Four models evaluated on a stratified 80/20 split (1,620 train / 405 test).

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.904 | 0.785 | 0.961 | 0.864 | 0.972 |
| **Random Forest** ✅ | **0.926** | **0.851** | **0.930** | **0.889** | **0.979** |
| Gradient Boosting | 0.916 | 0.863 | 0.876 | 0.869 | 0.982 |
| XGBoost | 0.904 | 0.863 | 0.830 | 0.846 | 0.981 |

**Best model: Random Forest** — selected on highest F1-score (0.889).

5-fold cross-validation confirmed stable generalization: **CV F1 = 0.904 ± 0.013**.

All four models exceed the "Excellent" threshold (F1 > 0.75, AUC-ROC > 0.80) by a large margin.

### ROC Curve

![ROC Curve](outputs/model/roc_curve.png)

AUC of 0.979 — the model reliably separates converted from non-converted leads across all decision thresholds.

### Confusion Matrix

![Confusion Matrix](outputs/model/confusion_matrix.png)

On the held-out test set of 405 leads:

| | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | 255 ✅ | 21 ❌ |
| **Actual: Yes** | 9 ❌ | 120 ✅ |

Only **9 false negatives** — the model misses very few leads that would have converted. In a sales context, false negatives (missed hot leads) are more costly than false positives, so this balance is favorable.

### Feature Importance

![Feature Importance](outputs/model/feature_importance.png)

| Rank | Feature | Importance | What it captures |
|------|---------|-----------|-----------------|
| 1 | `webinar_registrations` | 0.170 | High-intent event participation |
| 2 | `total_time_seconds` | 0.167 | Overall engagement depth |
| 3 | `total_events` | 0.161 | Interaction volume |
| 4 | `total_clicks` | 0.154 | Active browsing behaviour |
| 5 | `max_funnel_order` | 0.107 | Funnel progression depth |
| 6 | `session_count` | 0.074 | Return frequency |
| 7 | `engagement_span_days` | 0.050 | ⏱ Days active on site (temporal) |
| 8 | `avg_session_gap_days` | 0.041 | ⏱ Consistency of return pattern (temporal) |
| 9 | `pricing_page_views` | 0.038 | Pricing interest signal |
| 10 | `days_since_last_visit` | 0.029 | ⏱ Recency of last visit (temporal) |
| 11 | `source` | 0.010 | Acquisition channel quality |

The three temporal features (marked ⏱) — extracted from interaction timestamps — collectively account for **~12% of model importance**. Together they contributed more than pricing page views or acquisition source, validating the decision to engineer them.

---

## 🔍 Key EDA Findings

**Funnel depth is the strongest conversion signal.** Decision-stage leads converted at ~63%, versus ~2% for those who never left Awareness. The jump from Consideration to Decision alone is enormous.

**Session count separates intent from curiosity.** Converted leads averaged ~8 sessions vs ~3.5 for non-converted. Repeat engagement is a genuine buying signal.

**Webinar registrations are the top model feature.** Leads who registered for a webinar demonstrated the highest individual feature importance — they were actively investing time, not just browsing.

**Temporal patterns reveal a 30-day sales cycle.** The median lead took ~30 days from first visit to conversion. `engagement_span_days` — how long they stayed active — had a +0.46 correlation with conversion, making it the strongest engineered feature. Leads who stayed engaged for ~57 days on average converted vs ~23 days for those who didn't.

**March was the peak month.** Highest traffic volume and highest conversion rate (61%) simultaneously. May showed a warning sign: high volume but lowest conversion (49%) — a sign of lower-quality traffic.

**Instagram underperforms badly.** ~15% conversion rate vs ~37% for LinkedIn and Google.

**Interns never converted.** Not a single intern-sourced lead converted in the entire dataset.

Full analysis with charts: [`analysis.md`](./analysis.md)

---

## 🛡 Data Leakage Prevention

Features used to define the conversion label — `demo_requests`, `free_trial_starts`, `contact_form_submits`, and `form_completed` — were deliberately excluded from model inputs. Using them as features would allow the model to predict conversion using the exact events that define it, producing misleading metrics that would completely fail on real unseen data.

---

## ⚠️ Limitations

- **Static dataset** — trained on a fixed snapshot. Real-world lead behavior shifts over time (concept drift), requiring periodic retraining.
- **Compact feature set** — 11 features were selected for API simplicity from 31 engineered. A richer set was built during training.
- **No probability calibration** — predicted probabilities reflect model confidence but aren't calibrated against empirical conversion rates. A model that says 0.84 may not convert exactly 84% of the time.
- **Single acquisition source encoding** — `source` is label-encoded. One-hot encoding may better capture non-ordinal channel relationships.

---

## 🚀 Future Work

- **Probability calibration** — wrap model with `CalibratedClassifierCV` (Platt scaling) so scores reflect true empirical rates
- **`/batch_predict` endpoint** — score multiple leads in a single API call for production pipeline use
- **Drift detection** — monitor prediction distribution over time and alert when conversion patterns shift
- **Streamlit dashboard** — visual lead scoring UI for non-technical sales team members
- **LLM-powered outreach** — use Claude/OpenAI API to generate personalized sales emails from SHAP explanations + lead profile
- **Docker containerization** — consistent deployment across any environment
- **Sequence modelling** — explore LSTM/transformer approaches to capture temporal order of session events

---

## 👤 Author

**Vikas Maurya**   
Interests: Machine Learning, NLP, Generative AI  
GitHub: [@vikasmaurya](https://github.com/vikasmaurya9769)

---

*Assessment submitted for the AI/ML Engineer role at Vynqe — June 2026*
