# Electrical Fault Detection Using Logistic Regression and Support Vector Machines

## Introduction

Electrical faults in a three-phase power system can damage equipment, interrupt electricity supply, and create safety risks. A fault-detection system should identify abnormal electrical conditions accurately and, particularly, avoid missing genuine faults. In this project, I developed and compared machine-learning models to classify electrical measurements as either normal operation or a fault.

The main objective was to investigate whether feature engineering improves Logistic Regression and whether Support Vector Machines (SVMs) can identify nonlinear patterns more effectively. Four models were evaluated: Logistic Regression using Set R, Logistic Regression using Set Q, Linear SVM, and RBF SVM.

## Dataset and Features

The dataset contains 12,001 rows and measurements from a three-phase electrical system. The six input features are phase currents (Ia, Ib, Ic) and phase voltages (Va, Vb, Vc). The target variable, `fault`, identifies whether a fault is present: 0 represents no fault and 1 represents a fault.

Two empty columns were removed during data cleaning. The six measurements were used as the original features in Set R. Set Q contained these six measurements plus their squares, giving the model 12 features. Squared features can help Logistic Regression represent nonlinear relationships that a model using only the original features cannot easily capture.

The data was divided into training, validation, and test sets using the required random seed. The training set was used to learn model parameters, the validation set to select hyperparameters, and the test set to evaluate the final models. Standardisation statistics were calculated from the training data and applied to the other sets to prevent information leakage during preprocessing.

## Exploratory Data Analysis

The exploratory plots helped examine how electrical measurements behave during normal operation and faults. The current time-series plot shows how the three phase currents change across the first 2,000 rows, with fault intervals highlighted. These changes help reveal whether faults are associated with unusual current patterns.

**[http://localhost:8888/files/AML_Assignment_2_Project/results/figures/ia_ib_ic_first_2000.png?_xsrf=2%7C48392b47%7C11c03bd744d5f2b5fc16bd25c2621248%7C1791548175]**

*Figure 1. Phase-current measurements and the locations of fault-labelled samples.*

The scatter plot of Ia against Ib helps compare the two classes in a two-dimensional feature space. The histogram of Ia compares current distributions for fault and no-fault samples. Together, these plots help explain why the original measurements may not be separated effectively by a single straight line. Faults can produce complex changes in current, so adding squared features or using a nonlinear decision boundary may improve classification.

**[http://localhost:8888/files/AML_Assignment_2_Project/results/figures/ia_vs_ib_scatter.png?_xsrf=2%7C48392b47%7C11c03bd744d5f2b5fc16bd25c2621248%7C1791548175]**

*Figure 2. Exploratory comparison of electrical measurements for fault and no-fault samples.*

## Model Development

Logistic Regression was implemented manually using the sigmoid function, a cost function, and batch gradient ascent. Different learning rates were tested, and the selected learning rate was the largest one for which the cost decreased at every iteration.

For Set R, the chosen learning rate was 1. The final cost was 0.6772, and training accuracy was approximately 72.58%. For Set Q, the chosen learning rate was 10. The final cost decreased to 0.0534, and training accuracy reached 97.88%. This comparison demonstrates that the additional squared features substantially improved the performance of Logistic Regression on this dataset.

The SVM models were trained using the six standardised original measurements. Several values of the regularisation parameter C were evaluated for the linear kernel. The RBF kernel was tested using different C and gamma values. The validation set was used to select the best configuration for each kernel.

## Test Results

The four final models were evaluated on the held-out test set. The results are summarised below.

| Model | Accuracy | False alarms (FP) | Missed faults (FN) |
|---|---:|---:|---:|
| Logistic Regression (Set R) | 72.64% | 0 | 657 |
| Logistic Regression (Set Q) | 97.88% | 6 | 45 |
| Linear SVM | 73.64% | 0 | 633 |
| RBF SVM | **99.58%** | 4 | **6** |

The RBF SVM achieved the highest test accuracy, at 99.58%. It correctly identified 1,090 fault samples and missed only six actual faults. Set Q Logistic Regression also performed well, achieving 97.88% accuracy with 45 missed faults. In contrast, Set R Logistic Regression and Linear SVM missed hundreds of faults.

These results show why accuracy alone is insufficient for evaluating an electrical fault-detection model. A false alarm occurs when a normal condition is classified as a fault. A missed fault occurs when a real fault is classified as normal. In a protection system, missed faults can be particularly dangerous because they may prevent a fault from being recognised in time.

## Analysis and Discussion

**Why do Set R Logistic Regression and Linear SVM perform similarly?** Both models use the six original standardised measurements and learn a linear decision boundary. Their test accuracies, 72.64% and 73.64%, are close. This suggests that a linear boundary using the original measurements alone is insufficient to represent all the patterns separating fault and no-fault samples.

**Interpreting the squared-feature weights**
The trained Set Q logistic regression model assigns positive weights to all three squared-current features: \(I_a^2\), \(I_b^2\), and \(I_c^2\), with weights approximately 18.89, 4.75, and 13.96. In contrast, the squared-voltage features \(V_a^2\), \(V_b^2\), and \(V_c^2\) have negative weights of approximately −6.03, −6.46, and −6.14. This pattern suggests that the model uses higher current-related features as evidence for a fault and voltage-related features as evidence against it, when other features are held constant. This is consistent with the idea that faults can produce increased currents and voltage changes. However, the coefficients describe learned statistical relationships, not independent proof of physical causation.

**How do the support-vector counts compare?** The best linear SVM used C = 100 and had 5,835 support vectors. The best RBF SVM used C = 100 and gamma = 1, with 266 support vectors. The linear model relies on many training samples to define its decision boundary, whereas the RBF model represents the observed patterns with a much smaller support-vector set. A high support-vector count can indicate that many observations are important to defining the boundary or that the chosen kernel does not represent the data efficiently.

**What happens when C increases?** In the linear SVM validation results, accuracy changes very little as C increases. For the RBF SVM, validation accuracy generally improves as C increases, while the support-vector count generally decreases. The parameter C controls the penalty for classification errors: a larger value places more emphasis on correctly classifying training samples and allows less regularisation. The best observed validation accuracy was 99.67%, using C = 100 and gamma = 1.

**Could the test accuracy be optimistic?** Yes. The dataset contains consecutive time samples, but the rows were randomly divided into training, validation, and test sets. Neighbouring samples can be very similar, so closely related observations may appear in both training and test data. This can make the test results more optimistic than performance on genuinely new operating conditions. A time-based split and evaluation on independent data would provide stronger evidence of real-world generalisation.

## Limitations and Future Work

Although the RBF SVM performed extremely well, the results do not guarantee equivalent performance on every electrical system. The data may not represent all fault types, noise levels, equipment conditions, or operating environments. Future work should evaluate the models on independent measurements and time-based splits. The system should also be tested carefully for missed faults and false alarms before any practical use.

## Conclusion

This project compared two Logistic Regression feature sets and two SVM kernels for three-phase electrical fault detection. Adding squared features improved Logistic Regression test accuracy from 72.64% to 97.88%. The RBF SVM achieved the best test accuracy of 99.58% and missed only six faults. However, the random split may overestimate performance on new data. These results make the RBF SVM a promising candidate for further investigation, not a substitute for independent validation and electrical protection testing.

## Project Repository

The code, preprocessing steps, trained models, figures, and instructions are available in the accompanying GitHub repository: **[Insert your public GitHub repository URL before publishing.]**
