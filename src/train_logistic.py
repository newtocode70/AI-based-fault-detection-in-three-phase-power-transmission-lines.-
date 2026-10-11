from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.logistic import fit_logistic, predict
from src.metrics import accuracy

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = ROOT / "results" / "arrays"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "results" / "models"
for folder in [FIGURES, TABLES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

def choose_alpha(X, y, feature_set):
    alphas = [0.01, 0.1, 1, 10]
    results = {}
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        theta, history = fit_logistic(X, y, alpha, 2000)
        results[alpha] = (theta, history)
        plt.plot(np.arange(1, len(history) + 1), history, label=f"alpha={alpha}")
        print(f"{feature_set}: alpha={alpha}, final cost={history[-1]:.6f}, "
              f"decreases every iteration={bool(np.all(np.diff(history) < 0))}")
    plt.xlabel("Iteration")
    plt.ylabel("J(theta)")
    plt.title(f"Logistic regression cost — Set {feature_set}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / f"logistic_cost_set_{feature_set}.png", dpi=300)
    plt.show()

    valid = [a for a in alphas if np.all(np.diff(results[a][1]) < 0)]
    if not valid:
        raise RuntimeError(f"No tested alpha has strictly decreasing cost for Set {feature_set}.")
    return max(valid), results

def main():
    data = np.load(ARRAYS / "prepared_data.npz")
    y_train, y_val = data["y_train"], data["y_val"]
    specs = {
        "R": (data["X_train_R_intercept"], data["X_val_R_intercept"]),
        "Q": (data["X_train_Q"], data["X_val_Q"])
    }
    rows = []
    learned = {}
    for name, (X_train, X_val) in specs.items():
        alpha, experiments = choose_alpha(X_train, y_train, name)
        theta, history = fit_logistic(X_train, y_train, alpha, 2000)
        train_acc = accuracy(y_train, predict(X_train, theta))
        val_acc = accuracy(y_val, predict(X_val, theta))
        learned[f"theta_{name}"] = theta
        learned[f"history_{name}"] = history
        rows.append({
            "Feature set": name, "Chosen alpha": alpha,
            "Final J(theta)": history[-1],
            "Training accuracy": train_acc,
            "Validation accuracy": val_acc
        })
        print(f"\nSet {name}: chosen alpha={alpha}")
        print("Training accuracy:", f"{train_acc * 100:.4f}%")
        print("Validation accuracy:", f"{val_acc * 100:.4f}%")
        print("Theta:", theta)

    np.savez_compressed(MODELS / "logistic_parameters.npz", **learned)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "logistic_results.csv", index=False)
    print("\nLogistic results:")
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
