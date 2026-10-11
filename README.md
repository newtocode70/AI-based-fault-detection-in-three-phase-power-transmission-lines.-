# Three-Phase Transmission Line Fault Detection Using Machine Learning

**Course:** AI4003 — Applied Machine Learning  
**Assignment:** Assignment No. 02 — Fall 2026  
**Team Members:** Kashif Gulzar and Muhammad Umar Daraz

## 1. Project Overview

This project investigates the detection of faults in a three-phase transmission line using instantaneous current and voltage measurements. It compares logistic regression implemented from scratch with NumPy against Support Vector Machines (SVMs) using linear and Radial Basis Function (RBF) kernels.

The workflow covers dataset preparation, exploratory data analysis, feature engineering, model training, validation-based hyperparameter selection, final test evaluation, result analysis, and a Streamlit web interface for comparing two trained models.

> **Important:** This is an academic machine-learning project. Its predictions are not certified protection-relay decisions and should not be used for real-world power-system protection without further engineering validation and safety testing.

## 2. Dataset

The project uses the **Electrical Fault Detection and Classification** dataset, which contains simulated three-phase power-system measurements.

- **Source:** [Kaggle — Electrical Fault Detection and Classification](https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification)
- **Dataset used:** `detect_dataset.csv` (or the actual filename configured in `src/load_data.py`)
- **Rows reported after loading/cleaning:** 12,001
- **Columns after cleaning:** 7
- **Input features:** `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`
- **Target:** `fault` (0 = no fault, 1 = fault)

The current measurements represent the three phase currents, and the voltage measurements represent the three phase voltages. The original label column is `Output (S)` and is renamed to `fault` during preparation. The six measurement columns are numeric.

### Dataset split

The assignment specifies a random permutation with seed 42 and a 60% / 20% / remaining split:

| Split | Rows | Fault fraction |
|---|---:|---:|
| Training | 7,200 | 0.460694 |
| Validation | 2,400 | 0.451250 |
| Test | 2,401 | 0.456476 |

Standardization statistics are calculated from the training set and then applied to validation and test data. The test set is reserved for the final evaluation and is not used to select hyperparameters.

## 3. Project Structure

```text
AML_Assignment_2/
├── data/
│   └── detect_dataset.csv             # Dataset file; see loader configuration
├── src/
│   ├── load_data.py                   # Loads the dataset
│   ├── prepare.py                     # Task 1: cleaning, split, standardization
│   ├── eda.py                         # Task 2: exploratory plots
│   ├── logistic.py                    # Task 3: logistic-regression functions
│   ├── train_logistic.py              # Task 3: learning-rate experiments/final models
│   ├── train_svm.py                   # Task 4: linear and RBF SVM experiments
│   ├── metrics.py                     # Task 5: accuracy and confusion matrix
│   └── test_models.py                 # Task 5: final test-set evaluation
├── results/
│   ├── analysis.md                    # Task 6: questions 6.1–6.6
│   ├── figures/                       # Plots from EDA and model experiments
│   ├── data_preparation.csv           # Optional exported Task 1 summary
│   ├── logistic_results.csv           # Optional Task 3 summary
│   ├── svm_validation_results.csv     # Optional Task 4 table
│   ├── test_results.csv               # Optional Task 5 table
│   └── theta_Q.npy                    # Optional saved Set Q parameters
├── app/
│   └── app.py                         # Streamlit interface
├── best_linear_svm.joblib             # Saved best linear SVM
├── best_rbf_svm.joblib                # Saved best RBF SVM
├── preprocessing_stats.npz            # Standardization statistics and theta_Q
├── requirements.txt                   # Python dependencies
├── README.md
└── AML_Assignment_2.ipynb             # Optional notebook with experiments
```

**Note:** Keep filenames and model-file locations consistent with the paths in `src/load_data.py` and `app/app.py`. The results CSV files and `theta_Q.npy` are optional exports; create them if you want to include those artifacts in the repository. The `results/analysis.md` file is part of the required analysis deliverable.

## 4. Methodology

### Task 1 — Data Loading and Preparation

1. Load the dataset and print its shape, column names, and first five rows.
2. Inspect columns, remove any uninformative columns if present, rename `Output (S)` to `fault`, and check data types and missing values.
3. Count the fault and no-fault observations.
4. Split rows using `np.random.default_rng(42)` into training, validation, and test sets.
5. Standardize features using only the training-set means and standard deviations, then apply those same statistics to validation and test data.

### Task 2 — Exploratory Data Analysis

Three plots are produced from raw, unstandardized measurements:

- `Ia`, `Ib`, and `Ic` against row number for the first 2,000 rows, with fault regions shaded.
- A scatter plot of `Ia` against `Ib`, coloured by fault label.
- Overlaid histograms of `Ia` for fault and no-fault observations.

The plots help inspect current behaviour, class overlap, and whether a straight-line boundary based on raw measurements is likely to separate the classes well.

### Task 3 — Logistic Regression Implemented by Hand

Logistic regression is implemented with NumPy rather than a ready-made fitting estimator. The implementation includes:

- `sigmoid(z)`
- `hypothesis(X, theta)`
- `cost(X, y, theta)` with hypothesis clipping before logarithms
- Batch gradient descent starting from zero-valued parameters
- `predict(X, theta)` using a 0.5 probability threshold

Two feature sets are evaluated:

- **Set R:** six standardized raw features — `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`.
- **Set Q:** the six raw features plus their six squared values. The squares are calculated from raw measurements first, and all 12 features are standardized using training-set statistics.

The learning rates `0.01`, `0.1`, `1`, and `10` are tested for 2,000 iterations. The selected rates are `1` for Set R and `10` for Set Q, based on the requirement that cost decreases at every iteration among the tested rates.

### Task 4 — Support Vector Machines

SVM models are trained on the six standardized Set R features using `sklearn.svm.SVC`.

- **Linear kernel:** `C` values of `0.1`, `1`, `10`, and `100`.
- **RBF kernel:** the same four `C` values, each tested with `gamma = 0.1` and `gamma = 1`.

Validation accuracy and total support-vector counts are recorded. The best configuration for each kernel is selected using validation accuracy, and the chosen models are saved with `joblib`.

### Task 5 — Final Test Evaluation

Four final models are evaluated on the reserved test set:

1. Logistic regression — Set R
2. Logistic regression — Set Q
3. Best linear SVM
4. Best RBF SVM

Accuracy and a manually implemented confusion matrix are reported for each model. The analysis distinguishes true negatives (TN), false alarms (FP), missed faults (FN), and true positives (TP). Missed faults are important because they represent fault cases predicted as no fault.

### Task 6 — Analysis

`results/analysis.md` discusses:

1. Why logistic regression on Set R and a linear SVM perform similarly.
2. The signs of Set Q squared-current and squared-voltage weights.
3. The support-vector counts of the selected linear and RBF SVMs.
4. How `C` relates to margin violations and the observed validation results.
5. Why a random split of consecutive time samples may yield optimistic test estimates.
6. How missed faults and false alarms inform model selection for a protection application.

## 5. Results

### Logistic Regression Results

| Feature set | Selected learning rate | Final training cost | Training accuracy | Validation accuracy |
|---|---:|---:|---:|---:|
| Set R | 1 | 0.677161 | 72.58% | 73.08% |
| Set Q | 10 | 0.053434 | 97.88% | 97.83% |

### Best SVM Validation Results

| Model | Selected parameters | Validation accuracy | Support vectors |
|---|---|---:|---:|
| Linear SVM | `C = 0.1` | 72.58% | 5,849 |
| RBF SVM | `C = 100`, `gamma = 1` | 99.67% | 266 |

### Final Test Results

| Model | Test accuracy | TN | False alarms (FP) | Missed faults (FN) | TP |
|---|---:|---:|---:|---:|---:|
| Logistic Regression — Set R | 72.6364% | 1,305 | 0 | 657 | 439 |
| Logistic Regression — Set Q | 97.8759% | 1,299 | 6 | 45 | 1,051 |
| Linear SVM (`C = 0.1`) | 73.5943% | 1,305 | 0 | 634 | 462 |
| RBF SVM (`C = 100`, `gamma = 1`) | 99.5835% | 1,301 | 4 | 6 | 1,090 |

The RBF SVM achieved the highest test accuracy among the four evaluated models. In this test split, it produced 6 missed faults and 4 false alarms. These are dataset-specific results, and the model would require further validation under realistic operating conditions before any safety-critical use.

## 6. General Findings

- The Set R logistic-regression model and linear SVM achieved similar validation performance, consistent with both using linear decision boundaries on the six original features.
- Adding squared features in Set Q substantially improved logistic-regression performance on this dataset.
- The RBF SVM performed best among the evaluated models on the reserved test split.
- Accuracy alone is not sufficient for fault-detection applications: false alarms and missed faults should also be considered.
- Because adjacent rows are consecutive samples and the assignment uses a random split, samples from similar conditions may appear in both training and test sets. Test results may therefore be optimistic for a deployment scenario involving future or different operating conditions.

## 7. Front End

The Streamlit web interface is located in `app/app.py`. It accepts the six measurements (`Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`) and displays the predictions from:

- Logistic regression using Set Q features.
- The trained RBF SVM.

The app uses the saved model and preprocessing files. Ensure `best_rbf_svm.joblib` and `preprocessing_stats.npz` are in the project-root locations expected by `app/app.py`, or update the paths if you store them elsewhere. The interface is a demonstration and should not be treated as a certified relay controller.

## 8. How to Run

### Requirements

- Python installed
- Jupyter Notebook / IPython for the `%run` commands below
- Dependencies listed in `requirements.txt`
- Dataset in `data/` using the filename configured in `src/load_data.py`
- Saved model and preprocessing files for the front end

Install the dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

### Run Tasks 1–5 from Jupyter Notebook

Open Jupyter Notebook from the `AML_Assignment_2` project root and run each command in a separate notebook cell, in this order:

```python
%run -m src.load_data
```

```python
%run -m src.prepare
```

```python
%run -m src.eda
```

```python
%run -m src.train_logistic
```

```python
%run -m src.train_svm
```

```python
%run -m src.test_models
```

These commands use IPython's `%run` magic and should be run in a notebook/IPython environment, not pasted directly into a standard Windows Command Prompt. Confirm that the scripts use the same file paths and output locations described in this README.

### Run the Streamlit Front End

From a notebook cell, run:

```python
!python -m streamlit run app/app.py
```

Alternatively, from a regular Command Prompt opened in the project root, run:

```bash
python -m streamlit run app/app.py
```

Open the local URL printed by Streamlit in your browser. If the app reports a missing file, confirm the saved model/preprocessing paths in `app/app.py` and make sure the files exist there.

## 9. Figures and Artifacts

Store the following deliverables under `results/` (or the paths configured in the scripts):

- Task 2 exploratory plots.
- Task 3 cost curves for Set R and Set Q.
- Task 4 validation-accuracy plot.
- SVM validation table and final test-results table.
- `analysis.md` with Task 6 responses.
- Saved model files and learned parameters, if your repository layout places them under `results/`.
- A screenshot of the Streamlit interface for the README or project documentation.

## 10. Project Links

- **Medium Blog:** Add the published Medium article URL here.
- **GitHub Repository:** Add the public repository URL here.

## 11. Team

- **Kashif Gulzar**
- **Muhammad Umar Daraz**
