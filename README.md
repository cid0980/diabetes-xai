# Explainable AI System for Early Diabetes Prediction 🩺

Based on **Sisodia & Sisodia, ICCIDS 2018** — "Prediction of Diabetes using Classification Algorithms" (PIMA Indians Diabetes Dataset, 768 patients).

Paper baselines: Naive Bayes **76.30%** | SVM 65.10% | Decision Tree 73.82%

## What this project adds
1. **Reproduces** the paper's 3 models (10-fold CV)
2. **Improves** with median imputation + scaling + Random Forest / XGBoost + tuning
3. **Explains** every prediction with SHAP + LIME + doctor-friendly notes + PDF-style report

## Quick start
```bash
cd diabetes-xai
pip install -r requirements.txt
python train.py
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Files
- `train.py` — trains baselines + improved models, saves to `models/`
- `app.py` — Streamlit web app (Predict / Comparison / Global / About)
- `data/diabetes.csv` — PIDD dataset
- `models/` — best_model.pkl, all_models.pkl, preprocessor.pkl, results.json, comparison.csv

## Demo flow (viva)
1. **Predict** → try High-risk preset → Predict → show SHAP bar + LIME + report download
2. **Model Comparison** → paper vs ours table + confusion matrix
3. **Global Explanations** → feature importance + SHAP summary

⚠️ Educational demo only — not medical advice.
