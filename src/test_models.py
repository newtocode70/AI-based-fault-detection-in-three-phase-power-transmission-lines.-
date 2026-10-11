from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.logistic import predict
from src.metrics import accuracy, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = ROOT / "results" / "arrays"
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

def main():
    data = np.load(ARRAYS / "prepared_data.npz")
    y_test = data["y_test"]
    X_test_R_intercept = data["X_test_R_intercept"]
    X_test_Q = data["X_test_Q"]
    X_test_R = data["X_test_R"]

    params = np.load(MODELS / "logistic_parameters.npz")
    theta_R, theta_Q = params["theta_R"], params["theta_Q"]
    best_linear = joblib.load(MODELS / "best_linear_svm.joblib")
    best_rbf = joblib.load(MODELS / "best_rbf_svm.joblib")

    predictions = {
        "Logistic Regression (Set R)": predict(X_test_R_intercept, theta_R),
        "Logistic Regression (Set Q)": predict(X_test_Q, theta_Q),
        "Linear SVM": best_linear.predict(X_test_R),
        "RBF SVM": best_rbf.predict(X_test_R)
    }

    rows = []
    for name, pred in predictions.items():
        cm = confusion_matrix(y_test, pred)
        tn, fp = cm[0]
        fn, tp = cm[1]
        score = accuracy(y_test, pred)
        print(f"\nModel: {name}")
        print(f"Accuracy: {score * 100:.4f}%")
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(cm)
        print("True negatives:", int(tn))
        print("False alarms:", int(fp))
        print("Missed faults:", int(fn))
        print("True positives:", int(tp))
        rows.append({"Model": name, "Accuracy (%)": score * 100,
                     "TN": int(tn), "False alarms (FP)": int(fp),
                     "Missed faults (FN)": int(fn), "TP": int(tp)})
    result_df = pd.DataFrame(rows)
    result_df.to_csv(TABLES / "test_results.csv", index=False)
    print("\nFinal test results:")
    print(result_df.to_string(index=False, formatters={"Accuracy (%)": "{:.4f}".format}))

if __name__ == "__main__":
    main()
