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
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Paths 
LEADS_PATH        = "data/leads (1).csv"
INTERACTIONS_PATH = "data/interactions (1).csv"

# 1. LOAD DATA

def load_data(leads_path: str, interactions_path: str):
    """Load both CSVs and return as DataFrames."""
    log.info("Loading leads from %s", leads_path)
    leads = pd.read_csv(leads_path)

    log.info("Loading interactions from %s", interactions_path)
    interactions = pd.read_csv(interactions_path)

    log.info("Leads shape: %s | Interactions shape: %s", leads.shape, interactions.shape)
    return leads, interactions


# 2. DERIVE TARGET VARIABLE --------------------

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


# 3. FEATURE ENGINEERING -----------------------------

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

    # Funnel ordering 
    funnel_order = {"Awareness": 1, "Consideration": 2, "Evaluation": 3, "Decision": 4}
    interactions = interactions.copy()
    interactions["funnel_order"] = interactions["funnel_stage"].map(funnel_order).fillna(1)

    # Parse timestamps
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")

    # Aggregate per lead
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


    # ── Temporal features from timestamps ────────────────────────────────────────

    # Reference date = latest timestamp in the dataset
    ref_date = interactions["timestamp"].max()

    first_visit = interactions.groupby("lead_id")["timestamp"].min()
    last_visit  = interactions.groupby("lead_id")["timestamp"].max()

    # Feature 1 — How long were they active? (corr: +0.46, strongest temporal signal)
    # Converted leads stayed active for ~57 days avg vs ~23 days for non-converted
    engagement_span = (last_visit - first_visit).dt.total_seconds().div(86400).clip(lower=0)

    # Feature 2 — How recently did they visit? (corr: -0.18)
    # Recent leads convert more: Q1 (most recent) = 39%, Q4 (oldest) = 19%
    days_since_last = (ref_date - last_visit).dt.total_seconds().div(86400).clip(lower=0)

    # Feature 3 — How consistent is their return pattern? (corr: -0.24)
    # Tight gap = habitual returner. Short gap leads (Q1) convert at 45%, long gap (Q4) at 18%
    interactions_sorted = interactions.sort_values(["lead_id", "timestamp"])
    interactions_sorted["date"] = interactions_sorted["timestamp"].dt.date
    session_dates = (
        interactions_sorted
        .groupby(["lead_id", "date"])["session_id"]
        .nunique()
        .reset_index()
    )
    session_dates["date"] = pd.to_datetime(session_dates["date"])
    session_dates = session_dates.sort_values(["lead_id", "date"])
    session_dates["gap_days"] = session_dates.groupby("lead_id")["date"].diff().dt.days
    avg_gap = session_dates.groupby("lead_id")["gap_days"].mean()

    # Build temporal dataframe and merge into agg
    temporal = pd.DataFrame({
        "lead_id":               engagement_span.index,
        "engagement_span_days":  engagement_span.values,
        "days_since_last_visit": days_since_last.values,
        "avg_session_gap_days":  avg_gap.reindex(engagement_span.index).values,
    }).reset_index(drop=True)

    agg = agg.merge(temporal, on="lead_id", how="left")
    agg["avg_session_gap_days"] = agg["avg_session_gap_days"].fillna(0)

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

    if "return_visitor_flag" in df.columns:
        df["return_visitor_flag"] = (
            df["return_visitor_flag"].astype(str).str.lower().map({
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0
            })
            .fillna(0)
        )
    log.info("Created %d engineered features", df.shape[1] - len(leads.columns) )

    log.info(
        "Feature matrix shape after engineering: %s",
        df.shape
    )
    return df


# 4. PREPROCESSING ---------------------------------

def preprocess(df: pd.DataFrame):
    """
    Encode categoricals and select model features.
    Returns X (features), y (target), and the feature column names.
    """
    log.info("Preprocessing features...")

    # Columns to drop (IDs, raw text, target, and LEAKY features)
    # IMPORTANT: demo_requests, free_trial_starts, contact_form_submits, forms_completed
    # are used to DEFINE the target variable, so they must be excluded to prevent data leakage.
    drop_cols = [
    "lead_id", "business_email", "city", "state", "campaign",
    "browser", "annual_revenue_band", "screen_size",
    "created_at", "converted" ]

    categorical_to_encode = [
        "source", "lead_segment", "company_size", "industry",
        "funding_stage", "job_role", "account_type",
        "device_type", "employee_growth_band", "region", "first_touch_channel"
    ]
    

    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    # Label-encode categoricals
    encoders = {}
    for col in categorical_to_encode:
        if col in df_model.columns:
            encoder = LabelEncoder()
            df_model[col] = encoder.fit_transform(
                df_model[col].astype(str)
            )
            encoders[col] = encoder

    # Drop any remaining object columns
    obj_cols = df_model.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        log.warning("Dropping remaining object columns: %s", obj_cols)
        df_model.drop(columns=obj_cols, inplace=True)

    # Drop boolean columns that may cause issues
    bool_cols = df_model.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df_model[col] = df_model[col].astype(int)

    if df_model.isnull().sum().sum() > 0:
        log.warning(
            "Remaining missing values detected. Filling with 0."
        )
        df_model = df_model.fillna(0)

    selected_features = [
        "source",
        "session_count",
        "total_events",
        "total_time_seconds",
        "pricing_page_views",
        "webinar_registrations",
        "max_funnel_order",
        "total_clicks",
        "engagement_span_days",     # how long were they active
        "days_since_last_visit",    # are they still warm or gone cold
        "avg_session_gap_days",     # are they a consistent returner
    ]

    df_model = df_model[selected_features]

    y = df["converted"].values
    X = df_model.values
    feature_names = df_model.columns.tolist()

    log.info(
    "Final feature count: %d | Samples: %d",
    X.shape[1],
    X.shape[0]
    )

    log.info(
        "Target distribution: %d converted / %d total", y.sum(), len(y) )
    return X, y, feature_names, encoders


# 5. TRAIN & EVALUATE MODELS ---------------------

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, stratify=y, random_state=42 )

    log.info(
        "Train shape: %s | Test shape: %s",
        X_train.shape,
        X_test.shape
    )

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Compute all required metrics for a fitted model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred),              4),
        "precision": round(precision_score(y_test, y_pred,  zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,     zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,         zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_proba),              4),
    }
    return metrics


def train_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train Logistic Regression, Random Forest, XGBoost and Gradient Boosting.
    Return the best model (by F1 score) and all results.
    """
    # Class weights to handle imbalance
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {0: cw[0], 1: cw[1]}

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                class_weight=class_weight,
                max_iter=1000,
                random_state=42
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        ),
    }

    results   = []
    best_f1   = -1
    best_model = None
    best_name  = ""

    for name, model in models.items():
        log.info("Training %s ...", name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print("\n", metrics)

        log.info(
            "  %s → Acc=%.3f  Pre=%.3f  Rec=%.3f  F1=%.3f  AUC=%.3f",
            name, metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"], metrics["auc_roc"]
        )

        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_model = model
            best_name  = name

    log.info("Best model: %s (F1=%.3f)", best_name, best_f1)
    best_metrics = next(
        r for r in results
        if r["model"] == best_name
    )

    log.info("Best model metrics: %s", best_metrics)

    # Cross-validation for best model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv_scores = cross_val_score(best_model, X_all, y_all, cv=cv, scoring="f1")
    log.info("CV F1 scores: %s | Mean: %.3f ± %.3f",
             cv_scores.round(3), cv_scores.mean(), cv_scores.std())

    return best_model, best_name, results, cv_scores


def plot_feature_importance(feature_importance):
    top_features = feature_importance.head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_features["feature"],
        top_features["importance"] )
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig( "outputs/model/feature_importance.png" )
    plt.close()
    log.info(
        "Saved feature importance plot"
    )



def save_metrics(results, cv_scores):
    os.makedirs("outputs/model", exist_ok=True)
    metrics_data = {
        "models": results,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std())
    }
    with open( "outputs/model/model_metrics.json", "w") as f:
        json.dump(
            metrics_data, f, indent=4
        )
    log.info(
        "Saved model_metrics.json"
    )



def plot_confusion_matrix( model, X_test, y_test ):
    os.makedirs(
        "outputs/model",
        exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(
        y_test,
        y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm)
    plt.title( "Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center" )

    plt.tight_layout()
    plt.savefig(
        "outputs/model/confusion_matrix.png"
    )
    plt.close()
    log.info(
        "Saved confusion_matrix.png" )


def plot_roc_curve( model, X_test, y_test ):
    os.makedirs(
        "outputs/model",
        exist_ok=True )

    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(6,5))
    plt.plot( fpr, tpr )
    plt.plot(
        [0,1],
        [0,1],
        "--" )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(
        "outputs/model/roc_curve.png" )
    plt.close()
    log.info(
        "Saved roc_curve.png")


def save_model_artifacts( best_model, feature_names, encoders ):
    os.makedirs(
        "models",
        exist_ok=True
    )

    artifact = {
        "model": best_model,
        "feature_names": feature_names,
        "label_encoders": encoders
    }

    joblib.dump(
        artifact,
        "models/model.pkl"
    )

    log.info(
        "Saved model artifact successfully"
    )


def main():

    # Load data
    leads, interactions = load_data(
        LEADS_PATH,
        INTERACTIONS_PATH
    )

    # Create target
    leads = derive_target(
        leads,
        interactions
    )

    # Feature engineering
    df = engineer_features(
        leads,
        interactions
    )

    # Preprocessing
    X, y, feature_names, encoders = preprocess(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Train models
    best_model, best_name, results, cv_scores = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names
    )

    save_model_artifacts(
    best_model,
    feature_names,
    encoders
)

    print("\nResults:")
    for r in results:
        print(r)

    if best_name == "Logistic Regression":
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(
                best_model.named_steps["clf"].coef_[0]
            )
        })

    elif best_name == "Random Forest":
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.feature_importances_
        })

    elif best_name == "XGBoost":
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.feature_importances_
        })

    elif best_name == "Gradient Boosting":
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.feature_importances_
        })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\nTop 15 Important Features:")
    print(feature_importance.head(15))
    plot_feature_importance(
        feature_importance
    )
    save_metrics(
        results,
        cv_scores
    )
    plot_confusion_matrix(
        best_model,
        X_test,
        y_test
    )
    plot_roc_curve(
        best_model,
        X_test,
        y_test
    )
    
    artifact = joblib.load("models/model.pkl")
    print(artifact.keys())
    print(artifact["feature_names"])
    print(len(feature_names))

if __name__ == "__main__":
    main()
