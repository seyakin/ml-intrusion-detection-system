import os
import sys
import urllib.request
import pandas as pd


DATASET_URL = (
    "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "UNSW_NB15_training-set.csv")


def download():
    if os.path.exists(OUTPUT_PATH):
        print(f"[OK] Dataset already present at {OUTPUT_PATH}")
        return

    print("[INFO] Attempting to download UNSW-NB15 training set …")
    try:
        urllib.request.urlretrieve(DATASET_URL, OUTPUT_PATH)
        df = pd.read_csv(OUTPUT_PATH)
        print(f"[OK] Downloaded {len(df):,} records → {OUTPUT_PATH}")
    except Exception as exc:
        print(f"[WARN] Download failed ({exc}). Generating synthetic dataset instead.")
        _generate_fallback()


def _generate_fallback():
    """
    Import the generator from the main pipeline and save the dataset so
    subsequent runs can reuse it without regenerating.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ids_pipeline import generate_unsw_nb15_sample  # noqa: E402
    df = generate_unsw_nb15_sample(82332)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Synthetic UNSW-NB15-style dataset saved → {OUTPUT_PATH}")
    print(f"     Rows: {len(df):,} | Columns: {len(df.columns)}")


if __name__ == "__main__":
    download()
