import os
import json
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
LEADS_PATH        = "data/leads (1).csv"
INTERACTIONS_PATH = "data/interactions (1).csv"
MODEL_PATH        = "model.pkl"
METRICS_PATH      = "outputs/model_metrics.json"
FI_PLOT_PATH      = "outputs/feature_importance.png"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(leads_path: str, interactions_path: str):
    """Load both CSVs and return as DataFrames."""
    log.info("Loading leads from %s", leads_path)
    leads = pd.read_csv(leads_path)

    log.info("Loading interactions from %s", interactions_path)
    interactions = pd.read_csv(interactions_path)

    log.info("Leads shape: %s | Interactions shape: %s", leads.shape, interactions.shape)
    return leads, interactions


# ─────────────────────────────────────────────────────────────────────────────
# 2. DERIVE TARGET VARIABLE
# ─────────────────────────────────────────────────────────────────────────────

def derive_target(leads: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 'converted' binary target column.

    A lead is converted if ANY of the following is true:
      - event_name in ['demo_request', 'free_trial_start', 'contact_form_submit']
      - form_completed == True
    """
    log.info("Deriving target variable...")

    high_intent = {"demo_request", "free_trial_start", "contact_form_submit"}

    event_conv  = set(interactions.loc[interactions["event_name"].isin(high_intent), "lead_id"])
    form_conv   = set(interactions.loc[interactions["form_completed"] == True,        "lead_id"])
    converted   = event_conv | form_conv

    # Deduplicate leads (keep first occurrence per lead_id)
    leads = leads.drop_duplicates(subset="lead_id", keep="first").copy()
    leads["converted"] = leads["lead_id"].isin(converted).astype(int)

    rate = leads["converted"].mean() * 100
    log.info("Conversion rate: %.1f%% (%d / %d leads)", rate, leads["converted"].sum(), len(leads))
    return leads


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(leads: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich feature set from both datasets.

    Features created:
        Session-level  : session count, total events, engagement metrics
        Behavioral     : pricing views, case study views, webinar activity
        Funnel         : funnel depth and progression
        Recency        : engagement span and return visitor behavior
        Lead Profile   : source, segment, company size, industry
    """
    log.info("Engineering features...")

    # ── Funnel ordering ──────────────────────────────────────────────────────
    funnel_order = {"Awareness": 1, "Consideration": 2, "Evaluation": 3, "Decision": 4}
    interactions = interactions.copy()
    interactions["funnel_order"] = interactions["funnel_stage"].map(funnel_order).fillna(1)

    # ── Parse timestamps ─────────────────────────────────────────────────────
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")

    # ── Aggregate per lead ───────────────────────────────────────────────────
    agg = interactions.groupby("lead_id").agg(
    session_count=("session_id", "nunique"),
    total_events=("interaction_id", "count"),

    total_time_seconds=("time_on_page_seconds", "sum"),
    avg_time_per_page=("time_on_page_seconds", "mean"),

    avg_scroll_depth=("scroll_depth_percent", "mean"),
    max_scroll_depth=("scroll_depth_percent", "max"),

    pricing_page_views=("page_name", lambda x: (x == "Pricing").sum()),
    contact_page_views=("page_name", lambda x: (x == "Contact").sum()),
    case_study_views=("page_name", lambda x: (x == "Case Studies").sum()),
    roi_calc_views=("page_name", lambda x: (x == "ROI Calculator").sum()),
    features_page_views=("page_name", lambda x: (x == "Features").sum()),
    webinar_views=("page_name", lambda x: (x == "Webinar").sum()),

    pricing_events=("event_name", lambda x: (x == "pricing_page_view").sum()),
    doc_downloads=("event_name", lambda x: (x == "document_download").sum()),
    webinar_registrations=("event_name", lambda x: (x == "webinar_registration").sum()),
    blog_reads=("event_name", lambda x: (x == "blog_read").sum()),

    max_funnel_order=("funnel_order", "max"),
    funnel_stages_reached=("funnel_order", "nunique"),

    first_visit=("timestamp", "min"),
    last_visit=("timestamp", "max"),

    max_session_number=("session_number", "max"),
    return_visitor_flag=("is_return_visitor", "max"),

    total_clicks=("click_count", "sum"),
    avg_mouse_activity=("mouse_activity_score", "mean")
    ).reset_index()

    # Days between first and last visit (engagement span)
    agg["engagement_span_days"] = (
        (agg["last_visit"] - agg["first_visit"])
        .dt.total_seconds()
        .div(86400)
        .fillna(0)
        .clip(lower=0)
    )

    # High-intent engagement ratio
    agg["high_intent_page_ratio"] = (
        (
            agg["pricing_page_views"]
            + agg["contact_page_views"]
            + agg["case_study_views"]
        )
        / agg["total_events"].replace(0, 1)
    )

    agg["avg_events_per_day"] = (
    agg["total_events"]
    / (agg["engagement_span_days"] + 1)
    )

    # Events per session (engagement density)
    agg["events_per_session"] = agg["total_events"] / agg["session_count"].replace(0, 1)

    # Log-transform skewed numeric columns
    for col in ["total_time_seconds", "total_events", "total_clicks"]:
        agg[f"log_{col}"] = np.log1p(agg[col])

    agg.drop(columns=["first_visit", "last_visit"], inplace=True)

    # ── Merge with leads ─────────────────────────────────────────────────────
    leads = leads.copy()
    leads["created_at"] = pd.to_datetime(leads["created_at"], errors="coerce")

    # Lead-level categorical features
    categorical_cols = [
        "source", "lead_segment", "company_size", "industry",
        "funding_stage", "job_role", "account_type",
        "device_type", "employee_growth_band", "region", "first_touch_channel"
    ]
    for col in categorical_cols:
        if col in leads.columns:
            leads[col] = leads[col].fillna("Unknown")

    # Numeric lead features
    leads["employee_count"]    = leads["employee_count"].fillna(leads["employee_count"].median())
    leads["company_age_years"] = leads["company_age_years"].fillna(leads["company_age_years"].median())
    leads["log_employee_count"] = np.log1p(leads["employee_count"])

    # Merge
    df = leads.merge(agg, on="lead_id", how="left")

    # Fill missing interaction features with 0 (leads with no interactions)
    numeric_agg_cols = [c for c in agg.columns if c != "lead_id"]
    df[numeric_agg_cols] = df[numeric_agg_cols].fillna(0)

    log.info(
    "Created %d engineered features",
    df.shape[1] - len(leads.columns)
    )

    log.info(
        "Feature matrix shape after engineering: %s",
        df.shape
    )

    return df

def main():

    leads, interactions = load_data(
        LEADS_PATH,
        INTERACTIONS_PATH
    )

    leads = derive_target(
        leads,
        interactions
    )

    df = engineer_features(
        leads,
        interactions
    )

    log.info("Feature engineering completed successfully")


if __name__ == "__main__":
    main()