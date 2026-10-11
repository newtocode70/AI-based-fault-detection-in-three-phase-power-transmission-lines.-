from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "detect_dataset_clean.csv"
MEASUREMENT_COLS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

def load_raw(data_path=DATA_FILE):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Copy detect_dataset_clean.csv into the project's data/ folder."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all")
    if "Output (S)" in df.columns and "fault" not in df.columns:
        df = df.rename(columns={"Output (S)": "fault"})
    return df

if __name__ == "__main__":
    df = load_raw()
    print(f"{df.shape[0]} rows, {df.shape[1]} columns")
    print(df.columns.tolist(), end="\n\n")
    print(df.head(5).to_string(index=False))
