# Machine Learning for Electrical Fault Detection: Why Non-Linearity Matters

*Comparing Logistic Regression, Feature Augmentation, and Support Vector Machines across 12,000+ Three-Phase Power System Observations.*

---

## 1. Introduction

In modern electrical grids, rapid and reliable fault detection is essential for protecting expensive infrastructure, preventing service interruptions, and mitigating catastrophic fire and safety hazards. In three-phase power systems, abnormal conditions—such as line-to-ground or line-to-line faults—cause sudden transients in phase currents and voltages.

An operational protection system faces an asymmetrical risk profile:
* **False Alarms (False Positives):** Result in nuisance tripping and unnecessary operational downtime.
* **Missed Faults (False Negatives):** Can result in catastrophic transformer damage, transmission line destruction, or uncontrolled grid instability.

This project investigates the limits of linear separation versus non-linear decision boundaries on three-phase fault classification. Specifically, we compare:
1. **Baseline Logistic Regression** using raw measurements (Set $R$).
2. **Feature-Engineered Logistic Regression** incorporating quadratic power terms (Set $Q$).
3. **Linear Support Vector Machines (SVM)**.
4. **Non-Linear Radial Basis Function (RBF) Kernel SVM**.

---

## 2. Dataset and Preprocessing

The benchmark is conducted on a dataset containing **12,001 observations** from a simulated three-phase electrical transmission network.

### Feature Subsets

* **Original Features (Set $R$ - 6 dimensions):**
  * Phase Currents: $I_a, I_b, I_c$
  * Phase Voltages: $V_a, V_b, V_c$
* **Augmented Features (Set $Q$ - 12 dimensions):**
  * The baseline 6 physical measurements plus their squared terms:
    $$[I_a, I_b, I_c, V_a, V_b, V_c, I_a^2, I_b^2, I_c^2, V_a^2, V_b^2, V_c^2]$$
  * Adding quadratic terms allows a linear classifier to capture parabolic and circular decision boundaries in the original feature space.

### Preprocessing and Hygiene
1. **Cleaning:** Two unpopulated columns were removed during data preparation.
2. **Partitioning:** The dataset was partitioned into training, validation, and test subsets using a fixed random seed for reproducibility.
3. **Data Leakage Prevention:** Z-score standardisation statistics ($\mu, \sigma$) were computed strictly from the training partition and subsequently applied to scale the validation and test sets.

---

## 3. Exploratory Data Analysis (EDA)

Exploratory analysis reveals why traditional planar models struggle with raw electrical features:

```
         Phase Currents Over Time (First 2,000 Samples)
Current
  ▲             /\    /\         *Fault Interval*
  │            /  \  /  \       ┌────────────────┐
  │     /\    /    \/    \      │  /\/\    /\/\  │  ◄── Severe phase divergence
  │    /  \  /            \     │ /    \  /    \ │      and amplitude surge
  0───┴────┴───────────────┴────┴─┴──────┴──────┴┴────► Time (Samples)
```

1. **Current Magnitude Divergence:** During balanced operation, the three phase currents ($I_a, I_b, I_c$) maintain balanced, symmetric oscillatory amplitudes. Under a fault state, instantaneous currents surge significantly on the affected phases while voltages experience localized sag.
2. **Non-Linear Class Overlap:** Two-dimensional scatter plots of $I_a$ versus $I_b$ reveal that normal operating points cluster tightly in the center, while fault conditions radiate outward along multiple non-linear trajectories. A straight linear hyperplane cannot separate these classes without substantial overlap.

---

## 4. Model Training and Hyperparameter Tuning

### Logistic Regression
Logistic Regression was implemented from scratch using the standard sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Parameters were optimized via batch gradient ascent on the log-likelihood function. Multiple learning rates ($\alpha$) were tested, selecting the largest $\alpha$ ensuring monotonic convergence:
* **Set $R$ (Raw):** Best $\alpha = 1.0$. Final cost: **$0.6772$**; Training Accuracy: **$72.58\%$**.
* **Set $Q$ (Quadratic):** Best $\alpha = 10.0$. Final cost: **$0.0534$**; Training Accuracy: **$97.88\%$**.

The dramatic drop in loss ($0.6772 \rightarrow 0.0534$) highlights the immediate benefit of adding second-order feature terms.

### Support Vector Machines (SVM)
SVM classifiers were trained using standardized features. A validation grid search evaluated:
* **Linear SVM:** Soft-margin penalty $C \in \{0.1, 1, 10, 100\}$.
* **RBF Kernel SVM:** Joint tuning across $C \in \{0.1, 1, 10, 100\}$ and kernel bandwidth $\gamma \in \{0.001, 0.01, 0.1, 1.0\}$.

---

## 5. Test Performance and Evaluation

All finalized models were evaluated on the held-out test partition. In power systems, raw accuracy is secondary to the distribution of operational errors:

| Model | Feature Space | Test Accuracy | False Alarms (FP) | Missed Faults (FN) | System Viability |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | Set $R$ (6 raw features) | 72.64% | **0** | 657 | Unacceptable (High FN rate) |
| **Linear SVM** | Set $R$ (6 raw features) | 73.64% | **0** | 633 | Unacceptable (High FN rate) |
| **Logistic Regression** | Set $Q$ (12 quadratic features) | 97.88% | 6 | 45 | Substantial improvement |
| **RBF SVM** | Set $R$ ($C=100, \gamma=1.0$) | **99.58%** | 4 | **6** | **Optimal Protection Candidate** |

---

## 6. Analytical Insights and Discussion

### Why Did Linear Models Hit a Wall?
Both baseline Logistic Regression (72.64%) and Linear SVM (73.64%) plateau near identical performance thresholds, missing over 630 true faults. Because both models are constrained to a flat decision boundary in $\mathbb{R}^6$, they are structurally incapable of encircling healthy operating clusters.

### Physical Interpretation of Quadratic Weights
Inspecting the learned weights of Logistic Regression on Set $Q$ reveals interpretable electrical relationships:
* **Squared Current Weights ($I_a^2, I_b^2, I_c^2$):** $+18.89$, $+4.75$, $+13.96$
* **Squared Voltage Weights ($V_a^2, V_b^2, V_c^2$):** $-6.03$, $-6.46$, $-6.14$

The positive coefficients for squared currents signify that large deviations in current exponentially increase the log-odds of a fault. Conversely, negative weights for squared voltages reflect nominal grid behaviour: healthy circuits maintain steady operating voltages.

### Support Vector Efficiency
* **Linear SVM:** Retained **5,835 support vectors** ($C = 100$), indicating that the optimization struggled to find a wide, clean margin.
* **RBF SVM:** Collapsed the support vector count to just **266 vectors** ($C = 100, \gamma = 1.0$), while improving accuracy to 99.58%.

The Gaussian RBF kernel projects the raw 6-dimensional vectors into an infinite-dimensional feature space where the fault boundary can be cleanly isolated with minimal boundary anchors.

---

## 7. Limitations and Engineering Considerations

1. **Temporal Leakage in Random Splits:** 
   The evaluation utilized a random train/validation/test split on sequential measurement data. Because consecutive electrical readings are autocorrelated, random partitioning can lead to overly optimistic performance estimates. Future work should evaluate models using **chronological splits** or completely separate feeder systems.
2. **Real-Time Latency:** 
   While the RBF SVM achieved a $99.58\%$ score with only 6 missed faults, evaluating test points against 266 support vectors incurs higher computational overhead than computing the dot product of a 12-element vector in Logistic Regression. Hardware relays with microsecond trip deadlines may require balancing this trade-off.

---

## 8. Conclusion

* Linear models operating on raw measurements miss over $35\%$ of genuine faults and cannot be safely deployed for electrical fault detection.
* Quadratic feature expansion allows simple linear classifiers like Logistic Regression to leap from **72.64% to 97.88%** accuracy.
* The **RBF Kernel SVM achieved state-of-the-art results ($99.58\%$ accuracy, only 6 missed faults)** by naturally capturing non-linear boundary dynamics.

---

## Appendix: Python Code to Generate the Notebook

To view or publish this article as a ready-to-run Jupyter Notebook on GitHub, run the following Python snippet:

```python
import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Electrical Fault Detection: Logistic Regression vs. Support Vector Machines\n",
                "*A comparative study on feature engineering and kernel methods in 3-phase power systems.*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Summary Benchmark Results\n",
                "results = pd.DataFrame({\n",
                "    'Model': ['Logistic Regression (Set R)', 'Linear SVM', 'Logistic Regression (Set Q)', 'RBF SVM'],\n",
                "    'Features': ['Raw (6)', 'Raw (6)', 'Quadratic (12)', 'Raw (6)'],\n",
                "    'Test Accuracy (%)': [72.64, 73.64, 97.88, 99.58],\n",
                "    'False Alarms (FP)': [0, 0, 6, 4],\n",
                "    'Missed Faults (FN)': [657, 633, 45, 6]\n",
                "})\n",
                "results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualizing Model Performance\n",
                "fig, ax = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "ax[0].barh(results['Model'], results['Test Accuracy (%)'], color=['#94a3b8', '#94a3b8', '#38bdf8', '#22c55e'])\n",
                "ax[0].set_xlim(60, 102)\n",
                "ax[0].set_title('Test Accuracy Comparison (%)')\n",
                "ax[0].set_xlabel('Accuracy (%)')\n",
                "\n",
                "ax[1].bar(results['Model'], results['Missed Faults (FN)'], color=['#ef4444', '#ef4444', '#f59e0b', '#10b981'])\n",
                "ax[1].set_title('Critical Metric: Missed Faults (Lower is Better)')\n",
                "ax[1].set_ylabel('Number of False Negatives')\n",
                "plt.xticks(rotation=20, ha='right')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("Electrical_Fault_Detection_Blog.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Saved 'Electrical_Fault_Detection_Blog.ipynb' successfully!")
```
