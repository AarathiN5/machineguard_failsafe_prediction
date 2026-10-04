"""
Predict What Happens Next - PS ID: ALG-DATA-02
Streamlit web interface for the predictive maintenance model.
Deploy free on Streamlit Community Cloud.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance - ALG-DATA-02",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load("final_model.joblib")

    return (
        bundle["model"],
        bundle["features"],
        bundle["threshold"]
    )


try:

    model, FEATURES, THRESHOLD = load_model()

    MODEL_OK = True

except Exception as e:

    MODEL_OK = False
    MODEL_ERROR = str(e)


# ============================================================
# MACHINE TYPE MAPPING
# ============================================================

TYPE_MAP = {
    "L": 0,
    "M": 1,
    "H": 2
}


# ============================================================
# SIDEBAR - MACHINE SENSOR INPUT
# ============================================================

st.sidebar.header("Machine Sensor Input")

mtype = st.sidebar.selectbox(
    "Product type",
    ["L", "M", "H"],
    help="L = Low, M = Medium, H = High quality variant"
)

air_temp = st.sidebar.slider(
    "Air temperature [K]",
    295.0,
    305.0,
    298.0,
    0.1
)

proc_temp = st.sidebar.slider(
    "Process temperature [K]",
    305.0,
    315.0,
    308.5,
    0.1
)

speed = st.sidebar.slider(
    "Rotational speed [rpm]",
    1000,
    3000,
    1500,
    10
)

torque = st.sidebar.slider(
    "Torque [Nm]",
    10.0,
    80.0,
    40.0,
    0.5
)

wear = st.sidebar.slider(
    "Tool wear [min]",
    0,
    260,
    30,
    1
)


# ============================================================
# FEATURE ENGINEERING - SINGLE MACHINE
# ============================================================

def engineer_row(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque_value,
    tool_wear
):
    """
    Create one-row DataFrame with exactly the same
    feature names and calculations used during training.
    """

    power = (
        torque_value
        * rotational_speed
        * 2
        * np.pi
        / 60
    )

    row = {

        # ----------------------------------------------------
        # Machine type
        # ----------------------------------------------------

        "type_code": TYPE_MAP[machine_type],

        # ----------------------------------------------------
        # Original sensor features
        #
        # IMPORTANT:
        # These names are XGBoost-safe.
        # ----------------------------------------------------

        "air_temperature_K": air_temperature,

        "process_temperature_K": process_temperature,

        "rotational_speed_rpm": rotational_speed,

        "torque_Nm": torque_value,

        "tool_wear_min": tool_wear,

        # ----------------------------------------------------
        # Engineered features
        # ----------------------------------------------------

        "temp_delta": (
            process_temperature
            - air_temperature
        ),

        "temp_ratio": (
            process_temperature
            / air_temperature
        ),

        "power": power,

        "torque_wear": (
            torque_value
            * tool_wear
        ),

        "power_wear": (
            power
            * np.sqrt(tool_wear + 1)
        ),

        "torque_dev": abs(
            torque_value - 340.0
        )
    }

    return pd.DataFrame([row])[FEATURES]


# ============================================================
# TITLE
# ============================================================

st.title("Predict What Happens Next")

st.caption(
    "PS ID: ALG-DATA-02 | AI4I 2020 Predictive Maintenance | "
    "Machine Learning + Physics-Based Features | 5-Fold Stratified CV"
)


# ============================================================
# MODEL ERROR
# ============================================================

if not MODEL_OK:

    st.error(
        "final_model.joblib could not be loaded."
    )

    st.code(
        MODEL_ERROR
    )

    st.info(
        "Run `python pipeline.py` first to train and save the model."
    )

    st.stop()


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "Predict (single machine)",
        "Batch upload / Model card"
    ]
)


# ============================================================
# TAB 1 - SINGLE MACHINE PREDICTION
# ============================================================

with tab1:

    row = engineer_row(
        mtype,
        air_temp,
        proc_temp,
        speed,
        torque,
        wear
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    proba = float(
        model.predict_proba(row)[0, 1]
    )

    fail = proba >= THRESHOLD

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    c1.metric(
        "Failure probability",
        f"{proba:.1%}"
    )

    c2.metric(
        "Decision threshold",
        f"{THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # Risk gauge
    # --------------------------------------------------------

    st.progress(
        min(
            max(proba, 0.0),
            1.0
        ),
        text="Failure risk gauge"
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if fail:

        st.error(
            f"MACHINE LIKELY TO FAIL "
            f"(p = {proba:.1%} >= {THRESHOLD:.2f}) "
            f"- schedule maintenance"
        )

    else:

        st.success(
            f"Machine healthy "
            f"(p = {proba:.1%} < {THRESHOLD:.2f})"
        )

    # --------------------------------------------------------
    # Show engineered features
    # --------------------------------------------------------

    with st.expander(
        "Engineered features used by the model"
    ):

        st.dataframe(
            row.T.rename(
                columns={0: "value"}
            )
        )


# ============================================================
# TAB 2 - BATCH UPLOAD
# ============================================================

with tab2:

    st.subheader(
        "Batch prediction - upload an unlabeled CSV"
    )

    st.write(
        "Your CSV should contain these columns:"
    )

    st.code(
        "Type\n"
        "Air temperature [K]\n"
        "Process temperature [K]\n"
        "Rotational speed [rpm]\n"
        "Torque [Nm]\n"
        "Tool wear [min]"
    )

    up = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if up is not None:

        try:

            data = pd.read_csv(up)

            # ------------------------------------------------
            # Required columns
            # ------------------------------------------------

            required_columns = [
                "Type",
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

                st.stop()

            # ------------------------------------------------
            # Feature engineering
            # ------------------------------------------------

            def engineer_df(d):

                d = d.copy()

                # Consistent machine type encoding
                d["type_code"] = (
                    d["Type"].map(TYPE_MAP)
                )

                # Check for invalid types
                if d["type_code"].isna().any():

                    invalid_types = (
                        d.loc[
                            d["type_code"].isna(),
                            "Type"
                        ]
                        .unique()
                        .tolist()
                    )

                    raise ValueError(
                        f"Unknown Type values: "
                        f"{invalid_types}. "
                        f"Expected L, M, or H."
                    )

                # Temperature features
                d["temp_delta"] = (
                    d["Process temperature [K]"]
                    - d["Air temperature [K]"]
                )

                d["temp_ratio"] = (
                    d["Process temperature [K]"]
                    / d["Air temperature [K]"]
                )

                # Mechanical power
                d["power"] = (
                    d["Torque [Nm]"]
                    * d["Rotational speed [rpm]"]
                    * 2
                    * np.pi
                    / 60
                )

                # Torque × wear
                d["torque_wear"] = (
                    d["Torque [Nm]"]
                    * d["Tool wear [min]"]
                )

                # Power × wear
                d["power_wear"] = (
                    d["power"]
                    * np.sqrt(
                        d["Tool wear [min]"] + 1
                    )
                )

                # Torque deviation
                d["torque_dev"] = np.abs(
                    d["Torque [Nm]"] - 340.0
                )

                # Create XGBoost-safe feature names
                d["air_temperature_K"] = (
                    d["Air temperature [K]"]
                )

                d["process_temperature_K"] = (
                    d["Process temperature [K]"]
                )

                d["rotational_speed_rpm"] = (
                    d["Rotational speed [rpm]"]
                )

                d["torque_Nm"] = (
                    d["Torque [Nm]"]
                )

                d["tool_wear_min"] = (
                    d["Tool wear [min]"]
                )

                return d

            # ------------------------------------------------
            # Engineer uploaded data
            # ------------------------------------------------

            data = engineer_df(data)

            # ------------------------------------------------
            # Select exactly the features expected by model
            # ------------------------------------------------

            model_input = data[FEATURES].copy()

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            proba = model.predict_proba(
                model_input
            )[:, 1]

            data["failure_probability"] = proba

            data["prediction"] = (
                proba >= THRESHOLD
            ).astype(int)

            # ------------------------------------------------
            # Display results
            # ------------------------------------------------

            st.success(
                f"Successfully predicted "
                f"{len(data)} machines."
            )

            st.dataframe(
                data.head(20)
            )

            # ------------------------------------------------
            # Prediction summary
            # ------------------------------------------------

            st.subheader(
                "Prediction Summary"
            )

            failure_count = int(
                data["prediction"].sum()
            )

            healthy_count = (
                len(data)
                - failure_count
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Machines predicted healthy",
                healthy_count
            )

            col2.metric(
                "Machines predicted to fail",
                failure_count
            )

            # ------------------------------------------------
            # Download predictions
            # ------------------------------------------------

            st.download_button(
                "Download predictions CSV",

                data.to_csv(
                    index=False
                ).encode(),

                "predictions.csv",

                "text/csv"
            )

            # ------------------------------------------------
            # Probability distribution
            # ------------------------------------------------

            st.subheader(
                "Failure Probability Distribution"
            )

            fig, ax = plt.subplots()

            ax.hist(
                proba,
                bins=30,
                color="#4C8BF5"
            )

            ax.axvline(
                THRESHOLD,
                color="red",
                linestyle="--",
                label=(
                    f"threshold {THRESHOLD:.2f}"
                )
            )

            ax.set_xlabel(
                "Failure probability"
            )

            ax.set_ylabel(
                "Number of machines"
            )

            ax.legend()

            st.pyplot(fig)

        except Exception as e:

            st.error(
                "Could not process the uploaded CSV."
            )

            st.exception(e)


# ============================================================
# MODEL CARD / LEADERBOARD
# ============================================================

with tab2:

    st.subheader(
        "Model card / leaderboard"
    )

    lb_path = os.path.join(
        "predictions",
        "model_leaderboard.csv"
    )

    if os.path.exists(lb_path):

        leaderboard = pd.read_csv(
            lb_path
        )

        st.table(
            leaderboard.set_index(
                "model"
            )
        )

    else:

        st.warning(
            "model_leaderboard.csv not found. "
            "Run pipeline.py first."
        )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    st.markdown(
        "**Validation:** 5-fold stratified cross-validation; "
        "classification threshold tuned on out-of-fold predictions."
    )

    st.markdown(
        "**Primary metric:** PR-AUC because the dataset is highly "
        "imbalanced with approximately 3.4% positive machine failures."
    )

    st.markdown(
        "**Leakage prevention:** TWF, HDF, PWF, OSF and RNF "
        "failure-type columns are excluded from model features."
    )

    st.markdown(
        "**Feature engineering:** temperature difference, "
        "temperature ratio, mechanical power, torque-wear interaction, "
        "power-wear interaction and torque deviation."
    )
