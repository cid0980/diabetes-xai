"""
Train script for Explainable Diabetes Prediction Project.
Reproduces Sisodia et al. ICCIDS 2018 baselines + improved models.

Paper baselines (WEKA, 10-fold CV, raw PIDD):
  Naive Bayes: 76.30% | SVM: 65.10% | Decision Tree: 73.82%

This script:
  1. Reproduces baselines (raw data, 10-fold CV)
  2. Trains improved pipeline (median imputation for biologically-impossible zeros + scaling)
     with NB, SVM, DT, Logistic Regression, Random Forest, XGBoost + tuning
  3. Saves best model, preprocessor, metrics, comparison
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "diabetes.csv"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
# Columns where 0 is biologically impossible -> treat as missing
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

PAPER_RESULTS = {
    "Naive Bayes": {"precision": 0.759, "recall": 0.763, "f1": 0.760, "accuracy": 76.30, "roc_auc": 0.819},
    "SVM": {"precision": 0.424, "recall": 0.651, "f1": 0.513, "accuracy": 65.10, "roc_auc": 0.500},
    "Decision Tree": {"precision": 0.735, "recall": 0.738, "f1": 0.736, "accuracy": 73.82, "roc_auc": 0.751},
}

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    print(f"XGBoost not available ({e}), will use GradientBoosting instead.")
    HAS_XGB = False


def load_data():
    if not DATA_PATH.exists():
        import urllib.request
        print("📥 Downloading PIDD dataset...")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols")
    print("Class balance:", df[TARGET].value_counts().to_dict())
    return df


def get_cv():
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def evaluate_cv(model, X, y):
    """10-fold CV like the paper. Returns mean metrics."""
    cv = get_cv()
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "accuracy": float(np.mean(res["test_accuracy"]) * 100),
        "precision": float(np.mean(res["test_precision"])),
        "recall": float(np.mean(res["test_recall"])),
        "f1": float(np.mean(res["test_f1"])),
        "roc_auc": float(np.mean(res["test_roc_auc"])),
    }


def preprocess_fit(df):
    """Compute medians for zero-as-missing cols (fit on full data for simplicity,
    app uses same medians). Returns medians dict + scaler + processed X."""
    medians = {}
    df_fixed = df.copy()
    for col in ZERO_AS_MISSING:
        n_zeros = (df_fixed[col] == 0).sum()
        median_val = df_fixed.loc[df_fixed[col] != 0, col].median()
        medians[col] = float(median_val)
        df_fixed.loc[df_fixed[col] == 0, col] = median_val
        print(f"  {col}: {n_zeros} zeros -> median {median_val:.2f}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_fixed[FEATURES].values)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)
    return medians, scaler, X_scaled


def main():
    df = load_data()
    X_raw = df[FEATURES]
    y = df[TARGET].values

    print("\n" + "=" * 60)
    print("PHASE 1: Reproduce paper baselines (raw data, 10-fold CV)")
    print("=" * 60)
    baselines = {
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }
    reproduced = {}
    for name, model in baselines.items():
        m = evaluate_cv(model, X_raw, y)
        reproduced[name] = m
        paper_acc = PAPER_RESULTS[name]["accuracy"]
        print(f"{name:15s} -> Acc {m['accuracy']:.2f}% (paper {paper_acc:.2f}%) | "
              f"P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} ROC {m['roc_auc']:.3f}")

    print("\n" + "=" * 60)
    print("PHASE 2: Improved pipeline (median imputation + scaling, 10-fold CV)")
    print("=" * 60)
    medians, scaler, X_proc = preprocess_fit(df)

    # Save preprocessor for the app
    joblib.dump({"medians": medians, "scaler": scaler, "features": FEATURES},
                MODELS_DIR / "preprocessor.pkl")
    print("Saved preprocessor.pkl")

    # Save a background sample for SHAP/LIME
    joblib.dump(X_proc.sample(100, random_state=42).values, MODELS_DIR / "background.pkl")

    models = {
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True, C=1.0, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        scale_pos = (y == 0).sum() / (y == 1).sum()
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                          subsample=0.9, colsample_bytree=0.9,
                                          scale_pos_weight=scale_pos,
                                          eval_metric="logloss", random_state=42, n_jobs=-1)
    else:
        models["GradBoost"] = GradientBoostingClassifier(random_state=42)

    improved = {}
    for name, model in models.items():
        m = evaluate_cv(model, X_proc, y)
        improved[name] = m
        print(f"{name:20s} -> Acc {m['accuracy']:.2f}% | P {m['precision']:.3f} "
              f"R {m['recall']:.3f} F1 {m['f1']:.3f} ROC {m['roc_auc']:.3f}")

    # Tune the two best ensemble models quickly
    print("\nTuning Random Forest / XGBoost (small grid)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, stratify=y, random_state=42)

    tuned = {}
    # RF tuning
    rf_grid = {"n_estimators": [200, 300], "max_depth": [None, 10, 15],
               "min_samples_split": [2, 5]}
    gs_rf = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
                         rf_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    tuned["Random Forest (tuned)"] = gs_rf.best_estimator_
    print(f"  Best RF: {gs_rf.best_params_} CV-ROC {gs_rf.best_score_:.3f}")

    if HAS_XGB:
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_grid = {"max_depth": [3, 4, 5], "learning_rate": [0.03, 0.07],
                    "n_estimators": [200, 300]}
        gs_xgb = GridSearchCV(XGBClassifier(subsample=0.9, colsample_bytree=0.9,
                                            scale_pos_weight=scale_pos, eval_metric="logloss",
                                            random_state=42, n_jobs=-1),
                              xgb_grid, cv=5, scoring="roc_auc", n_jobs=-1)
        gs_xgb.fit(X_train, y_train)
        tuned["XGBoost (tuned)"] = gs_xgb.best_estimator_
        print(f"  Best XGB: {gs_xgb.best_params_} CV-ROC {gs_xgb.best_score_:.3f}")

    # Holdout evaluation for tuned models + pick best overall by ROC
    print("\nHoldout (20%) evaluation of tuned models:")
    best_name, best_model, best_roc = None, None, -1
    holdout = {}
    # also evaluate untuned on same holdout for fair pick
    candidates = {k: v for k, v in models.items()}
    candidates.update(tuned)
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        m = {
            "accuracy": float(accuracy_score(y_test, pred) * 100),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        holdout[name] = m
        print(f"  {name:22s} -> Acc {m['accuracy']:.2f}% ROC {m['roc_auc']:.3f} CM {m['confusion_matrix']}")
        if m["roc_auc"] > best_roc:
            best_roc = m["roc_auc"]
            best_name = name
            best_model = model

    print(f"\nBEST MODEL: {best_name} (holdout ROC {best_roc:.3f})")

    # Refit best on full processed data for deployment
    best_model.fit(X_proc, y)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    # Save all candidates fitted on full data for model-switching in app
    all_fitted = {}
    for name, model in candidates.items():
        try:
            model.fit(X_proc, y)
            all_fitted[name] = model
        except Exception as e:
            print(f"  Skip saving {name}: {e}")
    joblib.dump(all_fitted, MODELS_DIR / "all_models.pkl")
    print(f"Saved best_model.pkl ({best_name}) + all_models.pkl ({len(all_fitted)} models)")

    # Save results JSON
    results = {
        "paper": PAPER_RESULTS,
        "reproduced_baseline_10fold": reproduced,
        "improved_10fold": improved,
        "holdout_20pct": holdout,
        "best_model": best_name,
        "features": FEATURES,
        "dataset_rows": int(df.shape[0]),
        "class_balance": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
    }
    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results.json")

    # Comparison CSV for the app
    rows = []
    for name in ["Naive Bayes", "SVM", "Decision Tree"]:
        p = PAPER_RESULTS[name]
        r = improved.get(name, {})
        rows.append({"Model": name, "Paper_Acc": p["accuracy"],
                     "Ours_10fold_Acc": round(r.get("accuracy", 0), 2),
                     "Ours_ROC": round(r.get("roc_auc", 0), 3)})
    for name in ["Logistic Regression", "Random Forest", "XGBoost",
                 "Random Forest (tuned)", "XGBoost (tuned)"]:
        if name in improved or name in holdout:
            src = improved.get(name, holdout.get(name, {}))
            rows.append({"Model": name, "Paper_Acc": None,
                         "Ours_10fold_Acc": round(src.get("accuracy", 0), 2),
                         "Ours_ROC": round(src.get("roc_auc", 0), 3)})
    pd.DataFrame(rows).to_csv(MODELS_DIR / "comparison.csv", index=False)
    print("Saved comparison.csv")
    print("\nDone! Files in models/:", [p.name for p in MODELS_DIR.iterdir()])


if __name__ == "__main__":
    main()
