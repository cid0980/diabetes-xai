"""Explainable AI System for Early Diabetes Prediction.
Built on: Sisodia et al. ICCIDS 2018 (PIMA Indians Diabetes Dataset).
Run: streamlit run app.py --server.address 0.0.0.0 --server.port 8501
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# Optional XAI libs (graceful fallback if missing)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
try:
    from lime import lime_tabular
    HAS_LIME = True
except Exception:
    HAS_LIME = False

BASE = Path(__file__).parent

# Cloud-deploy safety: auto-download data + auto-train on first run if missing
from bootstrap import ensure_artifacts
ensure_artifacts(BASE)

st.set_page_config(page_title="Explainable Diabetes Predictor", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# Hide Streamlit's in-app toolbar (Share/star/edit/GitHub/menu) for EVERYONE
st.markdown(
    """<style>
/* Hide toolbar buttons (Share/star/GitHub/menu) but KEEP the >> button: it lives INSIDE the toolbar! */
[data-testid="stToolbarActions"] {display: none !important;}
[data-testid="stMainMenu"] {display: none !important;}
</style>""",
    unsafe_allow_html=True,
)

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Friendly ranges / defaults (medians from PIDD)
DEFAULTS = {"Pregnancies": 3, "Glucose": 117, "BloodPressure": 72,
            "SkinThickness": 29, "Insulin": 125, "BMI": 32.0,
            "DiabetesPedigreeFunction": 0.47, "Age": 33}
BOUNDS = {"Pregnancies": (0, 17), "Glucose": (0, 250), "BloodPressure": (0, 130),
          "SkinThickness": (0, 100), "Insulin": (0, 900), "BMI": (0.0, 70.0),
          "DiabetesPedigreeFunction": (0.05, 2.5), "Age": (10, 90)}

HEALTHY_PRESET = {"Pregnancies": 1, "Glucose": 95, "BloodPressure": 66,
                  "SkinThickness": 25, "Insulin": 85, "BMI": 24.5,
                  "DiabetesPedigreeFunction": 0.25, "Age": 26}
RISK_PRESET = {"Pregnancies": 5, "Glucose": 185, "BloodPressure": 80,
               "SkinThickness": 35, "Insulin": 200, "BMI": 34.5,
               "DiabetesPedigreeFunction": 0.85, "Age": 48}


@st.cache_resource
def load_artifacts():
    prep = joblib.load(BASE / "models" / "preprocessor.pkl")
    all_models = joblib.load(BASE / "models" / "all_models.pkl")
    best = joblib.load(BASE / "models" / "best_model.pkl")
    with open(BASE / "models" / "results.json") as f:
        results = json.load(f)
    bg_scaled = joblib.load(BASE / "models" / "background.pkl")
    # Original-scale background for LIME (inverse transform)
    scaler = prep["scaler"]
    bg_original = scaler.inverse_transform(bg_scaled)
    df = pd.read_csv(BASE / "data" / "diabetes.csv")
    return prep, all_models, best, results, bg_scaled, bg_original, df


prep, all_models, best_model, results, BG_SCALED, BG_ORIGINAL, DF = load_artifacts()
MEDIANS = prep["medians"]
SCALER = prep["scaler"]
BEST_NAME = results["best_model"]

MODEL_NAMES = list(all_models.keys())


def preprocess_input(values: dict):
    """Apply same pipeline as training: median impute zeros -> scale."""
    row = {}
    for f in FEATURES:
        v = float(values[f])
        if f in MEDIANS and v == 0:
            v = MEDIANS[f]
        row[f] = v
    orig = np.array([[row[f] for f in FEATURES]])
    scaled = pd.DataFrame(SCALER.transform(orig), columns=FEATURES)
    return orig, scaled


def risk_tier(p: float):
    if p < 0.35:
        return "LOW", "green"
    if p < 0.65:
        return "MEDIUM", "orange"
    return "HIGH", "red"


def textual_explanation(values: dict, shap_ranking=None):
    """Rule-based medical-style notes + SHAP ranking."""
    notes = []
    g = values["Glucose"]
    if g >= 200:
        notes.append(f"🩸 Glucose {g:.0f} is VERY HIGH (diabetes threshold ≥200) — strongest concern.")
    elif g >= 140:
        notes.append(f"🩸 Glucose {g:.0f} is HIGH (normal <140) — key risk driver.")
    elif g >= 100:
        notes.append(f"🩸 Glucose {g:.0f} is slightly elevated (normal fasting <100).")
    else:
        notes.append(f"🩸 Glucose {g:.0f} is in normal range. Good.")
    bmi = values["BMI"]
    if bmi >= 30:
        notes.append(f"⚖️ BMI {bmi:.1f} indicates obesity (≥30) — increases insulin resistance.")
    elif bmi >= 25:
        notes.append(f"⚖️ BMI {bmi:.1f} is overweight (25–29.9).")
    else:
        notes.append(f"⚖️ BMI {bmi:.1f} is healthy (18.5–24.9).")
    age = values["Age"]
    if age >= 45:
        notes.append(f"🎂 Age {age:.0f} — risk rises after 45.")
    else:
        notes.append(f"🎂 Age {age:.0f} — younger, lower age-related risk.")
    bp = values["BloodPressure"]
    if bp >= 90:
        notes.append(f"💓 Diastolic BP {bp:.0f} is HIGH (normal <80).")
    elif bp >= 80:
        notes.append(f"💓 Diastolic BP {bp:.0f} is borderline (normal <80).")
    dpf = values["DiabetesPedigreeFunction"]
    if dpf >= 0.8:
        notes.append(f"🧬 Pedigree function {dpf:.2f} is high — strong family history signal.")
    if shap_ranking:
        top3 = ", ".join(shap_ranking[:3])
        notes.append(f"🤖 Model says top drivers for THIS patient: {top3}.")
    return notes


# ---------------- Sidebar ----------------
PAGES = ["🔮 Predict", "📊 Model Comparison", "🧠 Global Explanations", "📄 About Paper"]
st.sidebar.title("🩺 Diabetes XAI")
st.sidebar.markdown("---")
st.sidebar.info(f"**Best model:** {BEST_NAME}\n\nDataset: PIDD (768 patients)\n\n⚠️ Educational demo — not medical advice.")
# Main-area navigation: always visible on phone + desktop, never needs the sidebar
page = st.radio("Navigate", PAGES, horizontal=True)

# ---------------- Page: Predict ----------------
if page == "🔮 Predict":
    st.title("Explainable Diabetes Risk Predictor")
    st.caption("Enter the 8 clinical values (same as the paper) → get prediction + WHY the model thinks so.")

    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        st.subheader("1️⃣ Patient values")
        preset = st.radio("Preset", ["Custom / Median", "Healthy example", "High-risk example"], horizontal=True)
        if preset == "Healthy example":
            for k, v in HEALTHY_PRESET.items():
                st.session_state[f"num_{k}"] = v
        elif preset == "High-risk example":
            for k, v in RISK_PRESET.items():
                st.session_state[f"num_{k}"] = v

        values = {}
        for f in FEATURES:
            lo, hi = BOUNDS[f]
            step = 0.1 if f in ("BMI", "DiabetesPedigreeFunction") else 1.0
            wkey = f"num_{f}"
            if wkey in st.session_state:
                values[f] = st.number_input(f, min_value=float(lo), max_value=float(hi),
                                            step=step, key=wkey)
            else:
                values[f] = st.number_input(f, min_value=float(lo), max_value=float(hi),
                                            value=float(DEFAULTS[f]), step=step, key=wkey)

        st.subheader("2️⃣ Model")
        choice = st.selectbox("Choose model", MODEL_NAMES,
                              index=MODEL_NAMES.index(BEST_NAME) if BEST_NAME in MODEL_NAMES else 0)
        model = all_models[choice]
        predict_btn = st.button("🔍 Predict Risk", type="primary", width="stretch")

    with col_out:
        if predict_btn:
            orig, scaled = preprocess_input(values)
            proba = float(model.predict_proba(scaled)[0, 1])
            pred = int(proba >= 0.5)
            tier, color = risk_tier(proba)

            st.subheader("Result")
            if pred == 1:
                st.error(f"### ⚠️ Diabetic — {tier} RISK ({proba:.1%})")
            else:
                st.success(f"### ✅ Non-Diabetic — {tier} RISK ({proba:.1%})")
            st.progress(proba, text=f"Diabetes probability: {proba:.1%}")
            st.caption(f"Model used: {choice}")

            # ---- SHAP local explanation ----
            shap_ranking = None
            if HAS_SHAP:
                try:
                    with st.spinner("Computing SHAP explanation..."):
                        explainer = shap.TreeExplainer(model) if hasattr(model, "estimators_") or "XGB" in choice or "Forest" in choice or "Tree" in choice else shap.Explainer(model, BG_SCALED[:30])
                        sv = explainer(scaled)
                        vals = np.array(sv.values)
                        # normalize shape to (n_features,)
                        if vals.ndim == 3:
                            vals = vals[0, :, 1] if vals.shape[2] > 1 else vals[0, :, 0]
                        elif vals.ndim == 2:
                            vals = vals[0]
                        order = np.argsort(np.abs(vals))[::-1]
                        shap_ranking = [FEATURES[i] for i in order]
                        st.markdown("#### 🤖 Why? — SHAP (per-patient impact)")
                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        y_pos = np.arange(len(FEATURES))
                        sorted_idx = order[::-1]
                        colors = ["#d62728" if vals[i] > 0 else "#2ca02c" for i in sorted_idx]
                        ax.barh(y_pos, vals[sorted_idx], color=colors)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels([FEATURES[i] for i in sorted_idx])
                        ax.set_xlabel("SHAP value  (red → pushes toward diabetic, green → toward healthy)")
                        ax.set_title(f"SHAP impact — {choice}")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.warning(f"SHAP unavailable for this model ({e}). Showing feature importance instead.")
                    if hasattr(model, "feature_importances_"):
                        imp = model.feature_importances_
                        order = np.argsort(imp)[::-1]
                        shap_ranking = [FEATURES[i] for i in order]
                        fig, ax = plt.subplots(figsize=(7, 3))
                        ax.barh([FEATURES[i] for i in order[::-1]], imp[order[::-1]], color="#1f77b4")
                        ax.set_xlabel("Importance")
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                st.info("SHAP not installed — install `shap` for visual explanations.")

            # ---- Textual explanation ----
            st.markdown("#### 🩺 Doctor-friendly summary")
            for n in textual_explanation(values, shap_ranking):
                st.write("- " + n)

            # ---- LIME ----
            if HAS_LIME:
                try:
                    with st.spinner("Computing LIME explanation..."):
                        lime_exp = lime_tabular.LimeTabularExplainer(
                            BG_ORIGINAL, feature_names=FEATURES,
                            class_names=["Healthy", "Diabetic"],
                            mode="classification", discretize_continuous=True,
                            random_state=42)
                        def lime_predict_fn(x):
                            return model.predict_proba(
                                pd.DataFrame(SCALER.transform(x), columns=FEATURES))
                        exp = lime_exp.explain_instance(orig[0], lime_predict_fn, num_features=6)
                        st.markdown("#### 🔬 LIME — local rules for this patient")
                        fig = exp.as_pyplot_figure()
                        fig.set_size_inches(7, 3.5)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        with st.expander("See LIME feature weights"):
                            st.write(exp.as_list())
                except Exception as e:
                    st.warning(f"LIME failed: {e}")

            # ---- Downloadable report ----
            lines = [f"DIABETES RISK REPORT (demo, not medical advice)",
                     f"Model: {choice}", f"Prediction: {'Diabetic' if pred==1 else 'Non-Diabetic'}",
                     f"Probability: {proba:.1%} ({tier} risk)", "",
                     "Input values:"]
            for f in FEATURES:
                lines.append(f"  - {f}: {values[f]}")
            lines += ["", "Notes:"]
            for n in textual_explanation(values, shap_ranking):
                lines.append("  - " + n)
            st.download_button("📥 Download report (.txt)", "\n".join(lines),
                               file_name="diabetes_risk_report.txt", width="stretch")
        else:
            st.info("👈 Enter values and click **Predict Risk**. Try the presets!")

# ---------------- Page: Comparison ----------------
elif page == "📊 Model Comparison":
    st.title("📊 Paper vs Our System")
    st.caption("Paper: Sisodia et al., ICCIDS 2018 — WEKA, 10-fold CV on raw PIDD.")

    paper = results["paper"]
    improved = results["improved_10fold"]
    holdout = results["holdout_20pct"]

    comp_rows = []
    for m in ["Naive Bayes", "SVM", "Decision Tree"]:
        comp_rows.append({"Model": m,
                          "Paper Acc %": paper[m]["accuracy"],
                          "Our 10-fold Acc %": round(improved.get(m, {}).get("accuracy", 0), 2),
                          "Our ROC": round(improved.get(m, {}).get("roc_auc", 0), 3)})
    for m in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if m in improved:
            comp_rows.append({"Model": m + " (new)", "Paper Acc %": None,
                              "Our 10-fold Acc %": round(improved[m]["accuracy"], 2),
                              "Our ROC": round(improved[m]["roc_auc"], 3)})
    st.dataframe(pd.DataFrame(comp_rows), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Accuracy: Paper vs Ours")
        dfp = pd.DataFrame([{"Model": r["Model"], "Paper": r["Paper Acc %"] or 0,
                             "Ours": r["Our 10-fold Acc %"]} for r in comp_rows[:6]])
        melted = dfp.melt("Model", var_name="Source", value_name="Accuracy")
        grouped = (
            alt.Chart(melted)
            .mark_bar()
            .encode(x=alt.X("Model:N", axis=alt.Axis(labelAngle=-30)),
                    xOffset="Source:N", y="Accuracy:Q", color="Source:N")
            .properties(height=320)
        )
        st.altair_chart(grouped, width="stretch")
    with c2:
        st.markdown("#### Holdout (20%) — tuned models")
        ht = pd.DataFrame([{"Model": k, "Acc": round(v["accuracy"], 1), "ROC": round(v["roc_auc"], 3)}
                           for k, v in holdout.items()]).sort_values("ROC", ascending=False)
        st.dataframe(ht, width="stretch")

    st.markdown(f"#### 🏆 Best model: {BEST_NAME}")
    cm = holdout[BEST_NAME]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Healthy", "Pred Diabetic"])
    ax.set_yticklabels(["True Healthy", "True Diabetic"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {BEST_NAME} (holdout)")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📌 Note on SVM difference"):
        st.write("The paper's SVM scored 65.10% with ROC 0.50 — its confusion matrix shows it predicted "
                 "everything as negative (500 TN, 0 TP). Our sklearn SVM (75.9% raw / 74.7% processed, "
                 "ROC ~0.82) uses probability estimates + scaling, so it actually separates classes. "
                 "This is a genuine improvement and a good viva point.")

# ---------------- Page: Global ----------------
elif page == "🧠 Global Explanations":
    st.title("🧠 What drives diabetes risk overall?")
    st.caption(f"Global explanations for best model: {BEST_NAME}")

    model = all_models[BEST_NAME]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1]
        st.markdown("#### Feature importance (model built-in)")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([FEATURES[i] for i in order], imp[order], color="#6a1b9a")
        ax.set_ylabel("Importance"); ax.set_title(f"Feature importance — {BEST_NAME}")
        plt.xticks(rotation=20); plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.write("Ranking:", " → ".join(FEATURES[i] for i in order))
    else:
        st.info("Best model has no feature_importances_. Switch best model to a tree/ensemble for this view.")

    if HAS_SHAP and st.button("Compute SHAP summary (takes ~30s)"):
        try:
            with st.spinner("Computing SHAP on 100 patients..."):
                explainer = shap.TreeExplainer(model)
                sv = explainer(BG_SCALED)
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(sv, BG_SCALED, feature_names=FEATURES, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        except Exception as e:
            st.error(f"SHAP summary failed: {e}")

    st.markdown("#### Dataset insight (PIDD)")
    st.write(f"768 patients: {results['class_balance']['negative']} healthy, "
             f"{results['class_balance']['positive']} diabetic. "
             "Glucose has the strongest single correlation with outcome.")
    corr = DF[FEATURES + ["Outcome"]].corr(numeric_only=True)["Outcome"].drop("Outcome").sort_values(ascending=False)
    st.bar_chart(corr)

# ---------------- Page: About ----------------
else:
    st.title("📄 About this project")
    st.markdown("""
**Explainable AI System for Early Diabetes Prediction with Risk Justification**

Based on: *Sisodia & Sisodia, "Prediction of Diabetes using Classification Algorithms",
Procedia Computer Science 132 (2018), ICCIDS 2018.*

| Item | Paper | Our system |
|---|---|---|
| Dataset | PIDD, 768 × 8 | Same PIDD |
| Models | NB, SVM, DT (WEKA) | NB, SVM, DT + LogReg, RF, XGBoost + tuned |
| Validation | 10-fold CV | 10-fold CV + 20% holdout |
| Best | NB 76.30% | RF-tuned ~76.6% holdout, ROC 0.83 |
| Explainability | None (ROC only) | SHAP + LIME + textual report |

**How to demo (viva):**
1. Predict page → load *High-risk example* → Predict → show SHAP bar + doctor notes.
2. Comparison page → show you reproduced the paper and beat/fixed SVM.
3. Global page → show Glucose/BMI/Age dominate.

⚠️ *Educational project only — not for real medical use.*
""")
    st.markdown("---")
    st.code("python train.py   # retrain all models\nstreamlit run app.py  # launch this app", language="bash")
