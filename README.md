# Predict Subscription of Term Deposit (Classification)

---

### Objective

Predict if a bank's customer would subscribe to a term deposit. (Constraint of an imbalanced dataset) Determine significant features that help explain the target variable. Customer segmentation to identify key customer groups to target the product to.

![image](https://user-images.githubusercontent.com/60640107/107608293-2f581180-6c01-11eb-922d-4860619a2e95.png)

### Table of Contents

- Data Description
- Choosing a Metric
- ML Algorithms with Hyperparameter Tuning
- Cost Sensitive Algorithms
- Data Sampling
- Feature Importance
- Future Work

### Data Description

  Features:

- Age
- Job
- Marital Status
- Education
- Loan Default
- Account Balance
- Housing
- Active Loan
- Contact
- Day
- Month
- Duration

> Distribution of Categorical Features
![image](https://user-images.githubusercontent.com/60640107/107608986-3718b580-6c03-11eb-870a-fa68e0c64736.png)

>Distribution of Continuous Features
![image](https://user-images.githubusercontent.com/60640107/107609036-557eb100-6c03-11eb-901f-fbca24937cbd.png)

### Choosing a Metric

The specified goal requires us to choose the model based on the metric accuracy.

Using accuracy as the metric is not optimal to making the best prediction especially due to the fact that our data set is imbalanced.
For Ex: Suppose we have 80% of the data points of just class A ; we can achieve 80% accuracy by just making every prediction A

To achieve an optimal result we would like to maximize two components:

1. The "True Positive Rate" aka Sensitivity aka Recall. Given by: TP/(TP+FN)

2. The Precision - How many of the positive predictions, are in fact correct. Given by: TP/(TP+FP)

To obtain a balance between both we use F Measure which is given by: (2 x Precision x Recall)/(Precision+Recall)
![image](https://user-images.githubusercontent.com/60640107/107609128-a0002d80-6c03-11eb-9145-aa9566427d5c.png)

[Back To The Top](#Objective)

### ML Algorithms with Hyperparameter Tuning

Model Results:

Model | F2 (Test Score)
------------ | -------------
Logistic Regression (L2 Penalty) | 0.47
Logistic Regression (L1 Penalty) | 0.48
Logistic Regression (Elastic Penalty) | 0.62
KNN | 0.15
Decision Tree | 0.37
Linear SVC | 0.21
Kernel SVC | 0.37
SGD Classifier | 0.58
LDA | 0.31
MLP Classifier | 0.43
Bagging (Decision Tree) | 0.34
Random Forest | 0.3
Gradient Boositing | 0.42
XGBoost | 0.43
CATBoost | 0.42
LightGBM | 0.43

### Cost Sensitive Algorithms

The idea behind using Cost Sensitive Algorithms is to given varying weights to each of the classes while building the model in an effort to built better predictors

Model Results:

Model | F2 (Test Score)
------------ | -------------
Logistic Regression (L2 Penalty) | 0.61
Logistic Regression (Elastic Penalty) | 0.46
Kernel SVC | 0.55
SGD Classifier | 0.59
XGBoost | 0.64
CATBoost | 0.65
LightGBM | 0.65

### Data Sampling

The idea is to change the dataset used to build the models by:

1) Adding copies of instances from the under-represented class aka over-sampling (via various SMOTE methods)

2) Deleting instances from the over-represented class, aka under-sampling

Model Results:

Model | F2 (Test Score)
------------ | -------------
Logistic Regression (Elastic Penalty) | 0.63
SGD Classifier | 0.62
XGBoost | 0.47
CATBoost | 0.61

### Feature Importances

Since Cost Sensitive CatBoost gave us the best test and validation set scores we will assess CATBoost Feature Importances
