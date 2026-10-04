"""
Predict What Happens Next - PS ID: ALG-DATA-02
AI4I 2020 Predictive Maintenance - Full Pipeline
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Headless: save plots, never open windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve
)

import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_DIR = "data"
OUT_DIR = "predictions"

os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "Machine failure"

# These columns are derived from individual failure modes.
# They are NOT used as model features because they can cause
# target leakage.
LEAK_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Explicit and consistent encoding for machine type
TYPE_MAP = {
    "L": 0,
    "M": 1,
    "H": 2
}


# ============================================================
# 1. LOAD DATA
# ============================================================

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found. Put the AI4I 2020 CSV inside the data/ folder."
    )

# Avoid accidentally loading test_unseen.csv as the training file
training_files = [
    f for f in csv_files
    if os.path.basename(f).lower() != "test_unseen.csv"
]

if not training_files:
    raise FileNotFoundError(
        "No training CSV found. Put the AI4I 2020 training CSV inside data/."
    )

train_path = training_files[0]

print(f"Loading training data: {train_path}")

df = pd.read_csv(train_path)

print(f"\nOriginal shape: {df.shape}")

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in the dataset."
    )

print("\nTarget balance:")
print(df[TARGET].value_counts())

print("\nTarget balance (%):")
print(df[TARGET].value_counts(normalize=True) * 100)


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

# Remove leakage columns if present
df = df.drop(
    columns=LEAK_COLS,
    errors="ignore"
)

print("\nShape after cleaning:", df.shape)


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

num_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# ------------------------------------------------------------
# Distribution plots
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 8)
)

for ax, column in zip(axes.flat, num_cols):

    for cls, color in [
        (0, "#4C8BF5"),
        (1, "#EA4335")
    ]:

        sns.histplot(
            df.loc[df[TARGET] == cls, column],
            bins=40,
            color=color,
            ax=ax,
            label=f"failure={cls}",
            alpha=0.6
        )

    ax.set_title(column)
    ax.legend()

# Hide unused subplot
axes.flat[-1].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "eda_distributions.png"),
    dpi=120
)

plt.close()

print("Saved:", os.path.join(OUT_DIR, "eda_distributions.png"))


# ------------------------------------------------------------
# Correlation matrix
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    df[num_cols + [TARGET]].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "eda_corr.png"),
    dpi=120
)

plt.close()

print("Saved:", os.path.join(OUT_DIR, "eda_corr.png"))


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def engineer(data):
    """
    Create engineered features from the original AI4I columns.
    """

    data = data.copy()

    # --------------------------------------------------------
    # Machine type
    # --------------------------------------------------------

    data["type_code"] = data["Type"].map(TYPE_MAP)

    # Check for unexpected machine types
    if data["type_code"].isna().any():
        unknown_types = data.loc[
            data["type_code"].isna(),
            "Type"
        ].unique()

        raise ValueError(
            f"Unknown machine Type values found: {unknown_types}"
        )

    # --------------------------------------------------------
    # Temperature features
    # --------------------------------------------------------

    data["temp_delta"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    data["temp_ratio"] = (
        data["Process temperature [K]"]
        / data["Air temperature [K]"]
    )

    # --------------------------------------------------------
    # Mechanical power
    #
    # Power = Torque × Angular velocity
    # Angular velocity = RPM × 2π / 60
    # --------------------------------------------------------

    data["power"] = (
        data["Torque [Nm]"]
        * data["Rotational speed [rpm]"]
        * 2
        * np.pi
        / 60
    )

    # --------------------------------------------------------
    # Torque × wear interaction
    # --------------------------------------------------------

    data["torque_wear"] = (
        data["Torque [Nm]"]
        * data["Tool wear [min]"]
    )

    # --------------------------------------------------------
    # Power × wear interaction
    # --------------------------------------------------------

    data["power_wear"] = (
        data["power"]
        * np.sqrt(
            data["Tool wear [min]"] + 1
        )
    )

    # --------------------------------------------------------
    # Deviation from approximate torque reference
    # --------------------------------------------------------

    data["torque_dev"] = np.abs(
        data["Torque [Nm]"] - 340.0
    )

    return data


df = engineer(df)


# ============================================================
# 5. SELECT FEATURES
# ============================================================

FEATURES = [
    "type_code",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_delta",
    "temp_ratio",
    "power",
    "torque_wear",
    "power_wear",
    "torque_dev"
]

X = df[FEATURES].copy()

y = df[TARGET].copy()

print("\nNumber of features:", len(FEATURES))

print("\nFeatures:")
for feature in FEATURES:
    print(" -", feature)


# ============================================================
# 6. IMPORTANT FOR XGBOOST
# ============================================================
#
# XGBoost does not allow feature names containing:
# [, ], <
#
# Therefore we create safe column names.
#
# The actual data remains unchanged.
# Only the column names are made XGBoost-compatible.
# ============================================================

SAFE_FEATURES = [
    "type_code",
    "air_temperature_K",
    "process_temperature_K",
    "rotational_speed_rpm",
    "torque_Nm",
    "tool_wear_min",
    "temp_delta",
    "temp_ratio",
    "power",
    "torque_wear",
    "power_wear",
    "torque_dev"
]

X.columns = SAFE_FEATURES

print("\nXGBoost-safe feature names:")
print(list(X.columns))


# ============================================================
# 7. MODEL DEFINITIONS
# ============================================================

positive_count = (y == 1).sum()
negative_count = (y == 0).sum()

scale_pos_weight = negative_count / positive_count

print(
    f"\nScale positive weight for XGBoost: "
    f"{scale_pos_weight:.2f}"
)


models = {

    # --------------------------------------------------------
    # Logistic Regression
    # Scaling is included directly in the pipeline so that
    # CV and final training use exactly the same processing.
    # --------------------------------------------------------

    "LogisticRegression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ]),

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    "XGBoost": xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        n_jobs=-1
    ),

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )
}


# ============================================================
# 8. STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []

oof_store = {}


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    # Out-of-fold predictions
    oof = np.zeros(len(y))

    # --------------------------------------------------------
    # Five-fold CV
    # --------------------------------------------------------

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        print(f"Fold {fold}/5...")

        # Clone the model so every fold starts fresh
        from sklearn.base import clone

        m = clone(model)

        m.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        oof[val_idx] = m.predict_proba(
            X.iloc[val_idx]
        )[:, 1]

    print(f"\n--- {name} done, computing metrics ---")

    # --------------------------------------------------------
    # Find threshold that maximizes F1
    # --------------------------------------------------------

    precision, recall, thresholds = precision_recall_curve(
        y,
        oof
    )

    f1_scores = (
        2
        * precision[:-1]
        * recall[:-1]
        / (
            precision[:-1]
            + recall[:-1]
            + 1e-12
        )
    )

    best_index = np.argmax(f1_scores)

    best_threshold = thresholds[best_index]

    predictions = (
        oof >= best_threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    pr_auc = average_precision_score(
        y,
        oof
    )

    roc_auc = roc_auc_score(
        y,
        oof
    )

    f1 = f1_score(
        y,
        predictions
    )

    precision_score_value = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall_score_value = recall_score(
        y,
        predictions,
        zero_division=0
    )

    row = {
        "model": name,
        "threshold": round(
            float(best_threshold),
            3
        ),
        "PR_AUC": round(
            pr_auc,
            4
        ),
        "ROC_AUC": round(
            roc_auc,
            4
        ),
        "F1": round(
            f1,
            4
        ),
        "Precision": round(
            precision_score_value,
            4
        ),
        "Recall": round(
            recall_score_value,
            4
        )
    }

    results.append(row)

    oof_store[name] = (
        oof,
        best_threshold
    )

    print("\nMetrics:")
    print(row)

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y,
            predictions
        )
    )


# ============================================================
# 9. MODEL LEADERBOARD
# ============================================================

results_df = (
    pd.DataFrame(results)
    .sort_values(
        "F1",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n")
print("=" * 70)
print("MODEL LEADERBOARD")
print("=" * 70)

print(results_df.to_string(index=False))


leaderboard_path = os.path.join(
    OUT_DIR,
    "model_leaderboard.csv"
)

results_df.to_csv(
    leaderboard_path,
    index=False
)

print(
    f"\nSaved leaderboard: {leaderboard_path}"
)


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

best_name = results_df.iloc[0]["model"]

best_thr = float(
    results_df.iloc[0]["threshold"]
)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model: {best_name}")
print(f"Threshold: {best_thr}")


# ============================================================
# 11. FINAL MODEL
# ============================================================

print(
    f"\nTraining final {best_name} model on ALL training data..."
)

final_model = models[best_name]

final_model.fit(
    X,
    y
)

# Save model, feature names and threshold
model_package = {
    "model": final_model,
    "features": SAFE_FEATURES,
    "original_features": FEATURES,
    "threshold": best_thr,
    "type_map": TYPE_MAP
}

model_path = "final_model.joblib"

joblib.dump(
    model_package,
    model_path
)

print(
    f"Saved final model: {model_path}"
)


# ============================================================
# 12. UNSEEN TEST PREDICTIONS
# ============================================================

test_path = os.path.join(
    DATA_DIR,
    "test_unseen.csv"
)

if os.path.exists(test_path):

    print("\nFound unseen test dataset.")

    test = pd.read_csv(
        test_path
    )

    print(
        f"Unseen test shape: {test.shape}"
    )

    # Remove leakage columns if present
    test = test.drop(
        columns=LEAK_COLS,
        errors="ignore"
    )

    # Engineer same features
    test = engineer(test)

    # Select original features
    test_X = test[FEATURES].copy()

    # Rename to safe XGBoost-compatible names
    test_X.columns = SAFE_FEATURES

    # Predict probability
    probabilities = final_model.predict_proba(
        test_X
    )[:, 1]

    # Convert probability to class
    predictions = (
        probabilities >= best_thr
    ).astype(int)

    # Create output
    output = pd.DataFrame({
        "machine_failure_probability": probabilities,
        "machine_failure_prediction": predictions
    })

    predictions_path = os.path.join(
        OUT_DIR,
        "unseen_predictions.csv"
    )

    output.to_csv(
        predictions_path,
        index=False
    )

    print(
        f"Saved predictions: {predictions_path}"
    )

    print("\nPrediction summary:")

    print(
        output["machine_failure_prediction"]
        .value_counts()
    )

else:

    print(
        "\nNo test_unseen.csv found."
    )

    print(
        "Put the hidden/unseen test set inside data/ "
        "and rerun the script."
    )


# ============================================================
# 13. SHAP FEATURE IMPORTANCE
# ============================================================

if best_name in [
    "RandomForest",
    "XGBoost",
    "LightGBM"
]:

    try:

        import shap

        print("\nGenerating SHAP feature importance...")

        sample_size = min(
            500,
            len(X)
        )

        sample = X.sample(
            sample_size,
            random_state=RANDOM_STATE
        )

        explainer = shap.TreeExplainer(
            final_model
        )

        shap_values = explainer.shap_values(
            sample
        )

        # Handle different SHAP output formats
        if isinstance(
            shap_values,
            list
        ):

            vals = shap_values[1]

        else:

            vals = shap_values

        if vals.ndim == 3:

            vals = vals[:, :, 1]

        plt.figure(
            figsize=(8, 6)
        )

        shap.summary_plot(
            vals,
            sample,
            show=False
        )

        plt.tight_layout()

        shap_path = os.path.join(
            OUT_DIR,
            "shap_summary.png"
        )

        plt.savefig(
            shap_path,
            dpi=120,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved SHAP plot: {shap_path}"
        )

    except Exception as e:

        print(
            "SHAP skipped:",
            e
        )

else:

    print(
        "\nSHAP skipped because the best model "
        "is not a tree-based model."
    )


# ============================================================
# 14. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)

print(
    "\nGenerated files:"
)

print(
    f" - {leaderboard_path}"
)

print(
    f" - {model_path}"
)

if os.path.exists(
    os.path.join(
        OUT_DIR,
        "unseen_predictions.csv"
    )
):

    print(
        f" - {os.path.join(OUT_DIR, 'unseen_predictions.csv')}"
    )

if os.path.exists(
    os.path.join(
        OUT_DIR,
        "shap_summary.png"
    )
):

    print(
        f" - {os.path.join(OUT_DIR, 'shap_summary.png')}"
    )

print("\n")
