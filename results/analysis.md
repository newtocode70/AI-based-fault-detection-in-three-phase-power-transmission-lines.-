1. Which model performs best, and why?

Answer:
The RBF SVM performs best, achieving 99.58% test accuracy. It correctly identifies 1,090 faults and misses only 6 actual faults. 
Its nonlinear RBF kernel can learn complex relationships between the electrical measurements and fault classes.
Logistic Regression with Set Q also performs well, achieving 97.88% accuracy because the squared features help it represent nonlinear relationships.



2. Why does Set Q perform better than Set R?

Answer:
Set R contains the original six electrical measurements: Ia, Ib, Ic, Va, Vb, and Vc. 
Set Q adds the squares of these features, giving the model more information about nonlinear patterns. 
Logistic Regression accuracy increases from 72.64% with Set R to 97.88% with Set Q. 
This shows that the additional squared features help distinguish normal conditions from fault conditions.


3. Compare Linear SVM and RBF SVM.

Answer:
Linear SVM achieves 73.64% test accuracy, while RBF SVM achieves 99.58%. 
The linear kernel creates a linear decision boundary, which may not adequately separate the electrical fault patterns. 
The RBF kernel can represent more complex nonlinear boundaries, resulting in better classification and fewer missed faults.


4. Which model is better at detecting faults, and why are false negatives important?

Answer:
RBF SVM is the strongest model for fault detection in these results. 
It produces only 6 false negatives, compared with 45 for Set Q Logistic Regression, 633 for Linear SVM, and 657 for Set R Logistic Regression.
A false negative means an actual electrical fault is classified as normal, potentially allowing a dangerous condition to go undetected. 
Therefore, reducing false negatives is especially important in electrical fault detection.


5. What do the validation and test results tell us?

Answer:
Set Q Logistic Regression achieves 97.83% validation accuracy and 97.88% test accuracy, indicating consistent performance on these datasets.
RBF SVM achieves 99.67% validation accuracy with (C=100) and (gamma=1), while its final test accuracy is 99.58%. 
The similar validation and test results suggest good generalization to the held-out test data, although further testing on independent electrical measurements would provide stronger evidence.


6. What are the limitations of the models and experiments?

Answer:
The results are based on the available dataset and the selected features and hyperparameters. 
High test accuracy does not guarantee the same performance on real electrical systems with different operating conditions or noise.
The models should therefore be tested on additional data, especially data containing different fault types and operating conditions. 
False negatives should also be monitored carefully because undetected electrical faults can have serious consequences.

Summary of your actual test results:

|   Model	                         | Test accuracy|	False positives	 |  False negatives|
|   Logistic Regression (Set R)      |	72.64%      |     	0	         |     657         |
|   Logistic Regression (Set Q)      |	97.88%      |    	6	         |     45          |
|   Linear SVM	                     |  73.64%	    |       0        	 |     633         |
|   RBF SVM	                         |  99.58%      |     	4            |     6           |