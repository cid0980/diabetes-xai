"""One-time self setup for fresh cloud deploys (Hugging Face / Streamlit Cloud).

If data/ or models/ are missing (fresh upload), this downloads the PIDD
dataset and trains all models automatically. Safe to call every startup.
"""
import urllib.request
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"


def ensure_artifacts(base=None):
    base = Path(base or Path(__file__).parent)
    data_path = base / "data" / "diabetes.csv"
    best_path = base / "models" / "best_model.pkl"

    if not data_path.exists():
        print("📥 Dataset not found — downloading PIDD...", flush=True)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, data_path)
        print(f"✅ Downloaded {data_path}", flush=True)

    if not best_path.exists():
        print("🔧 Models not found — training now (one-time, ~1-2 min)...", flush=True)
        import train
        train.main()
        print("✅ Training done.", flush=True)
    else:
        print("✅ Data + models ready.", flush=True)


if __name__ == "__main__":
    ensure_artifacts()
