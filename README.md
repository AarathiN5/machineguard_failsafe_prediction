# FailSafe AI

**Predict What Happens Next** | Hackathon Problem ID: ALG-DATA-02

FailSafe AI is a predictive maintenance system that forecasts industrial machine
failures from sensor data. Built on the AI4I 2020 Predictive Maintenance Dataset,
it predicts whether a machine is likely to fail based on air temperature, process
temperature, rotational speed, torque, and tool wear.

## Live Demo

- **Deployed app:** https://failsafeformachines.streamlit.app
- **Dataset:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## The Problem

Historical data reveals patterns that help anticipate future behavior. The goal
is to build an end-to-end predictive pipeline: explore and clean data, engineer
features, select and train models, validate with suitable metrics, and generate
predictions for unseen data.

Key difficulty: only **3.4% of machines fail**, so accuracy is meaningless
(a model that always says "no failure" scores 96.6%). All model selection was
done using **PR-AUC, F1, Precision, and Recall** instead.

## Key Design Decisions

1. **Leakage prevention.** The columns TWF, HDF, PWF, OSF, and RNF are derived
   from the target variable. Using them would inflate scores dishonestly, so
   they are dropped. This is documented in the pipeline.

2. **Physics-informed features.** Engineered features mirror real failure
   mechanisms:
   - `temp_delta` and `temp_ratio`: overheating drives heat dissipation failures
   - `power = torque x rotational speed`: overload drives power failures
   - `torque_wear` and `power_wear`: a worn tool under high load is the
     most dangerous operating condition
   - `torque_dev`: deviation from nominal 340 Nm operating point

3. **Imbalance-aware validation.** 5-fold stratified cross-validation for every
   reported number. The decision threshold is tuned on out-of-fold predictions,
   never on evaluation data.

## Results (5-fold stratified CV)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| RandomForest | 0.8798 | 0.9738 | 0.8685 | 0.9384 | ... |
| LightGBM | ... | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... | ... |
| LogisticRegression | ... | ... | ... | ... | ... |

(Fill in from `predictions/model_leaderboard.csv` after running the pipeline.)

## Anti-Hardcoding Guarantee

This repository contains **no precomputed answers**. All predictions are
generated at runtime by the trained model. We demonstrate this by evaluating on
a held-out slice of data that was excluded from all training, with scores
revealed live in the demo video. See `evaluate_hidden.py`.
## Project Structure
data/                  training CSV (ai4i2020.csv) and hidden test CSV
predictions/           leaderboard and unseen prediction outputs
pipeline.py            full ML pipeline: EDA, features, CV, training, export
app.py                 Streamlit web interface
notebook.ipynb         narrated walkthrough of the methodology
evaluate_hidden.py     hidden final evaluation (simulates hackathon test set)
final_model.joblib     trained model bundle (features + threshold)
requirements.txt       dependencies

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, Streamlit,
Matplotlib, Seaborn, SHAP, GitHub, Streamlit Community Cloud.


## How to Run

```bash
pip install -r requirements.txt
python pipeline.py          # trains models, runs CV, saves final_model.joblib
streamlit run app.py        # local version of the deployed app

