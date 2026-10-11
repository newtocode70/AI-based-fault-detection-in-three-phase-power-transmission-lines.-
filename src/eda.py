from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.load_data import load_raw, MEASUREMENT_COLS

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

def main():
    df = load_raw()
    for col in MEASUREMENT_COLS:
        df[col] = __import__("pandas").to_numeric(df[col], errors="coerce")
    df["fault"] = __import__("pandas").to_numeric(df["fault"], errors="coerce")
    if df[MEASUREMENT_COLS + ["fault"]].isna().any().any():
        raise ValueError("Missing or nonnumeric required data found.")

    first = df.iloc[:2000]
    plt.figure(figsize=(14, 6))
    for col in ["Ia", "Ib", "Ic"]:
        plt.plot(first.index, first[col], label=col)
    fault_values = first["fault"].to_numpy()
    start = None
    for i, value in enumerate(fault_values):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            plt.axvspan(start, i, alpha=0.2)
            start = None
    if start is not None:
        plt.axvspan(start, len(fault_values) - 1, alpha=0.2)
    plt.xlabel("Row number")
    plt.ylabel("Current (A)")
    plt.title("Ia, Ib, Ic vs row number (first 2000 rows)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "ia_ib_ic_first_2000.png", dpi=300)
    plt.show()

    no_fault = df["fault"] == 0
    fault = df["fault"] == 1
    plt.figure(figsize=(8, 6))
    plt.scatter(df.loc[no_fault, "Ia"], df.loc[no_fault, "Ib"], label="No fault", alpha=0.5)
    plt.scatter(df.loc[fault, "Ia"], df.loc[fault, "Ib"], label="Fault", alpha=0.5)
    plt.xlabel("Ia")
    plt.ylabel("Ib")
    plt.title("Ia vs Ib coloured by fault")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "ia_vs_ib_scatter.png", dpi=300)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.hist(df.loc[no_fault, "Ia"], bins=50, alpha=0.5, label="No fault")
    plt.hist(df.loc[fault, "Ia"], bins=50, alpha=0.5, label="Fault")
    plt.xlabel("Ia")
    plt.ylabel("Frequency")
    plt.title("Ia distribution for fault and no-fault samples")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "ia_histogram_fault_vs_no_fault.png", dpi=300)
    plt.show()

    print("Observations to write after inspecting the plots:")
    print("1. Describe current changes during shaded fault periods.")
    print("2. Describe whether fault and no-fault samples overlap in the Ia-Ib scatter plot.")
    print("3. Describe overlap between the two Ia histograms.")
    print("4. Answer whether one straight line can separate the classes in the Ia-Ib plane.")

if __name__ == "__main__":
    main()
