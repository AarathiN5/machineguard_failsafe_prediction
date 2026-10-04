# Deployment guide

## 1. GitHub Repository
    git init
    git add .
    git commit -m "Predictive maintenance - ALG-DATA-02"
    git branch -M main
    git remote add origin https://github.com/<your-username>/<repo-name>.git
    git push -u origin main

## 2. Live/Deployed Project (free)
1. Train the model locally first (creates final_model.joblib):  python pipeline.py
2. Push the repo INCLUDING final_model.joblib and app.py.
3. Go to https://share.streamlit.io -> New app -> pick your repo ->
   main file: app.py -> Deploy. You get a public URL.
   Test locally first with:  streamlit run app.py

## 3. Prototype Demo Video (2-4 min, upload to Google Drive)
1. Intro (15s): problem statement + dataset
2. Pipeline walkthrough (60s): EDA plots, 3.4% imbalance, leakage columns dropped,
   engineered features, CV leaderboard with PR-AUC
3. Live app demo (90s): open the deployed Streamlit link, move sliders
   (raise torque + tool wear), watch the gauge flip to LIKELY TO FAIL,
   then batch-upload a CSV and download predictions
4. Close (15s): metrics summary + GitHub link on screen
