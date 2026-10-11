from pathlib import Path
import numpy as np
from src.load_data import load_raw, MEASUREMENT_COLS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
ARRAYS.mkdir(parents=True, exist_ok=True)

def prepare_data():
    df = load_raw()
    required = MEASUREMENT_COLS + ["fault"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")

    for col in MEASUREMENT_COLS + ["fault"]:
        df[col] = __import__("pandas").to_numeric(df[col], errors="coerce")

    print("Raw rows / columns:", df.shape)
    print("Columns:", df.columns.tolist())

    # Remove only fully empty or constant non-feature/non-label columns.
    extra_cols = [c for c in df.columns if c not in required]
    drop_cols = [c for c in extra_cols if df[c].isna().all() or df[c].nunique(dropna=False) <= 1]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    print("Columns removed:", drop_cols)

    print("\nMissing values per column:")
    print(df[required].isna().sum())
    if df[required].isna().any().any():
        raise ValueError("Missing values found in required columns. Inspect and resolve before splitting.")
    if not set(df["fault"].unique()).issubset({0, 1}):
        raise ValueError("The fault label must contain only 0 and 1.")

    print("\nFault rows:", int((df["fault"] == 1).sum()))
    print("No-fault rows:", int((df["fault"] == 0).sum()))

    # Exact split specified in the assignment.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"{name} rows: {len(part)} | fault fraction: {part['fault'].mean():.6f}")

    X_train_raw = train_df[MEASUREMENT_COLS].to_numpy(dtype=float)
    X_val_raw = val_df[MEASUREMENT_COLS].to_numpy(dtype=float)
    X_test_raw = test_df[MEASUREMENT_COLS].to_numpy(dtype=float)
    y_train = train_df["fault"].to_numpy(dtype=int)
    y_val = val_df["fault"].to_numpy(dtype=int)
    y_test = test_df["fault"].to_numpy(dtype=int)

    mean_R = X_train_raw.mean(axis=0)
    std_R = X_train_raw.std(axis=0)
    std_R[std_R == 0] = 1.0
    X_train_R = (X_train_raw - mean_R) / std_R
    X_val_R = (X_val_raw - mean_R) / std_R
    X_test_R = (X_test_raw - mean_R) / std_R

    # Set Q: raw features followed by squares; standardize all 12 columns
    # using training-set statistics, as required by the assignment.
    X_train_Q_raw = np.column_stack([X_train_raw, X_train_raw ** 2])
    X_val_Q_raw = np.column_stack([X_val_raw, X_val_raw ** 2])
    X_test_Q_raw = np.column_stack([X_test_raw, X_test_raw ** 2])

    mean_Q = X_train_Q_raw.mean(axis=0)
    std_Q = X_train_Q_raw.std(axis=0)
    std_Q[std_Q == 0] = 1.0
    X_train_Q_scaled = (X_train_Q_raw - mean_Q) / std_Q
    X_val_Q_scaled = (X_val_Q_raw - mean_Q) / std_Q
    X_test_Q_scaled = (X_test_Q_raw - mean_Q) / std_Q

    # Logistic regression includes an intercept column. SVM uses Set R without it.
    X_train_R_intercept = np.column_stack([np.ones(len(X_train_R)), X_train_R])
    X_val_R_intercept = np.column_stack([np.ones(len(X_val_R)), X_val_R])
    X_test_R_intercept = np.column_stack([np.ones(len(X_test_R)), X_test_R])
    X_train_Q = np.column_stack([np.ones(len(X_train_Q_scaled)), X_train_Q_scaled])
    X_val_Q = np.column_stack([np.ones(len(X_val_Q_scaled)), X_val_Q_scaled])
    X_test_Q = np.column_stack([np.ones(len(X_test_Q_scaled)), X_test_Q_scaled])

    np.savez_compressed(
        ARRAYS / "prepared_data.npz",
        X_train_R=X_train_R, X_val_R=X_val_R, X_test_R=X_test_R,
        X_train_R_intercept=X_train_R_intercept,
        X_val_R_intercept=X_val_R_intercept,
        X_test_R_intercept=X_test_R_intercept,
        X_train_Q=X_train_Q, X_val_Q=X_val_Q, X_test_Q=X_test_Q,
        y_train=y_train, y_val=y_val, y_test=y_test,
        train_mean_R=mean_R, train_std_R=std_R,
        train_mean_Q=mean_Q, train_std_Q=std_Q,
        feature_columns=np.array(MEASUREMENT_COLS)
    )
    df.to_csv(RESULTS / "cleaned_data.csv", index=False)
    print("\nSaved prepared arrays to:", ARRAYS / "prepared_data.npz")
    print("Set R train / validation / test:", X_train_R.shape, X_val_R.shape, X_test_R.shape)
    print("Set Q train / validation / test:", X_train_Q.shape, X_val_Q.shape, X_test_Q.shape)

if __name__ == "__main__":
    prepare_data()
