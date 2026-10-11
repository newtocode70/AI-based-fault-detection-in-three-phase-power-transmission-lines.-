from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from src.metrics import accuracy

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = ROOT / "results" / "arrays"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "results" / "models"
for folder in [FIGURES, TABLES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

def main():
    data = np.load(ARRAYS / "prepared_data.npz")
    X_train, X_val = data["X_train_R"], data["X_val_R"]
    y_train, y_val = data["y_train"], data["y_val"]
    C_values = [0.1, 1, 10, 100]
    gamma_values = [0.1, 1]
    linear_models, rbf_models = {}, {}
    rows = []

    for C in C_values:
        model = SVC(kernel="linear", C=C)
        model.fit(X_train, y_train)
        score = accuracy(y_val, model.predict(X_val))
        linear_models[C] = model
        rows.append({"Kernel": "linear", "C": C, "Gamma": "-",
                     "Validation accuracy": score,
                     "Support vectors": int(model.n_support_.sum())})

    for C in C_values:
        for gamma in gamma_values:
            model = SVC(kernel="rbf", C=C, gamma=gamma)
            model.fit(X_train, y_train)
            score = accuracy(y_val, model.predict(X_val))
            rbf_models[(C, gamma)] = model
            rows.append({"Kernel": "rbf", "C": C, "Gamma": gamma,
                         "Validation accuracy": score,
                         "Support vectors": int(model.n_support_.sum())})

    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "svm_validation_results.csv", index=False)
    print(results.to_string(index=False, formatters={"Validation accuracy": "{:.4f}".format}))

    linear_rows = results[results["Kernel"] == "linear"]
    best_linear_C = float(linear_rows.loc[linear_rows["Validation accuracy"].idxmax(), "C"])
    rbf_rows = results[results["Kernel"] == "rbf"]
    best_rbf_row = rbf_rows.loc[rbf_rows["Validation accuracy"].idxmax()]
    best_rbf_C, best_rbf_gamma = float(best_rbf_row["C"]), float(best_rbf_row["Gamma"])

    best_linear = linear_models[best_linear_C]
    best_rbf = rbf_models[(best_rbf_C, best_rbf_gamma)]
    joblib.dump(best_linear, MODELS / "best_linear_svm.joblib")
    joblib.dump(best_rbf, MODELS / "best_rbf_svm.joblib")

    print("\nBest linear SVM: C =", best_linear_C)
    print("Validation accuracy:", accuracy(y_val, best_linear.predict(X_val)) * 100, "%")
    print("Support vectors:", int(best_linear.n_support_.sum()))
    print("\nBest RBF SVM: C =", best_rbf_C, "gamma =", best_rbf_gamma)
    print("Validation accuracy:", accuracy(y_val, best_rbf.predict(X_val)) * 100, "%")
    print("Support vectors:", int(best_rbf.n_support_.sum()))

    plt.figure(figsize=(9, 6))
    plt.plot(linear_rows["C"], linear_rows["Validation accuracy"] * 100,
             marker="o", label="Linear SVM")
    for gamma in gamma_values:
        subset = rbf_rows[rbf_rows["Gamma"] == gamma].sort_values("C")
        plt.plot(subset["C"], subset["Validation accuracy"] * 100,
                 marker="o", label=f"RBF SVM, gamma={gamma}")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("Validation accuracy (%)")
    plt.title("SVM validation accuracy vs C")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "svm_validation_accuracy.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
