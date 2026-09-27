"""Full UI test: every page + every model + every button.
Run: python3 test_app.py   (needs: pip install -r requirements.txt)
"""
from streamlit.testing.v1 import AppTest

ALL_MODELS = ["Naive Bayes", "SVM", "Decision Tree", "Logistic Regression",
              "Random Forest", "XGBoost",
              "Random Forest (tuned)", "XGBoost (tuned)"]


def fresh():
    at = AppTest.from_file("app.py", default_timeout=600)
    at.run()
    assert not at.exception, f"initial run failed: {at.exception}"
    return at


def test_predict_all_models():
    for m in ALL_MODELS:
        at = fresh()
        # 1. preset must actually fill the inputs
        at.radio[1].set_value("High-risk example").run()
        assert not at.exception, f"preset failed: {at.exception}"
        g = [w for w in at.number_input if w.label == "Glucose"][0]
        assert float(g.value) == 185, f"preset broken for {m}: Glucose={g.value}"
        # 2. select model
        at.selectbox[0].set_value(m).run()
        assert not at.exception, f"select {m} failed: {at.exception}"
        # 3. predict (runs SHAP + LIME + report)
        at.button[0].click().run()
        assert not at.exception, f"PREDICT crashed for {m}: {at.exception}"
        print(f"  PASS predict+SHAP+LIME: {m}", flush=True)


def test_comparison():
    at = fresh()
    at.radio[0].set_value("\U0001F4CA Model Comparison").run()
    assert not at.exception, f"comparison failed: {at.exception}"
    print("  PASS Comparison page", flush=True)


def test_global():
    at = fresh()
    at.radio[0].set_value("\U0001F9E0 Global Explanations").run()
    assert not at.exception, f"global failed: {at.exception}"
    at.button[0].click().run()  # SHAP summary button
    assert not at.exception, f"SHAP summary failed: {at.exception}"
    print("  PASS Global page + SHAP summary", flush=True)


def test_about():
    at = fresh()
    at.radio[0].set_value("\U0001F4C4 About Paper").run()
    assert not at.exception, f"about failed: {at.exception}"
    print("  PASS About page", flush=True)


if __name__ == "__main__":
    print("Testing Predict page x 8 models...", flush=True)
    test_predict_all_models()
    print("Testing other pages...", flush=True)
    test_comparison()
    test_global()
    test_about()
    print("ALL TESTS PASSED")
