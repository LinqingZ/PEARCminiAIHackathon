# Sentiment Analysis on Movie Reviews

## Overview

This project aims to predict the sentiment (positive or negative) of movie reviews using various machine learning models. Below is a summary of the different models and hyperparameters tested, along with their respective accuracies.

## Models and Experiments

### 1. Random Forest Classifier
**Model:** `RandomForestClassifier(n_estimators=100)`  
**Method:** `get_preds(test_feat, train_feat, y_test, y_train, RandomForestClassifier(n_estimators=100))`  
**Accuracy:** 0.8358

**Description:** A Random Forest Classifier with 100 trees was used as the initial model. It provided a decent baseline accuracy.

### 2. Logistic Regression with L2 Penalty
**Model:** `LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)`  
**Method:** `get_preds(test_feat, train_feat, y_test, y_train, LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42))`  
**Accuracy:** 0.8548

**Description:** A Logistic Regression model with L2 regularization showed better performance than the Random Forest Classifier.

### 3. Logistic Regression with Reduced Regularization Strength
**Model:** `LogisticRegression(penalty='l2', max_iter=500, C=0.1, random_state=42)`  
**Method:** `get_preds(test_feat, train_feat, y_test, y_train, LogisticRegression(penalty='l2', max_iter=500, C=0.1, random_state=42))`  
**Accuracy:** 0.875

**Description:** Reducing the regularization strength (C=0.1) improved the accuracy further.

### 4. Logistic Regression with Class Weights Balanced
**Model:** `LogisticRegression(penalty='l2', max_iter=500, C=1, class_weight='balanced', random_state=42)`  
**Method:** `get_preds(test_feat, train_feat, y_test, y_train, LogisticRegression(penalty='l2', max_iter=500, C=1, class_weight='balanced', random_state=42))`  
**Accuracy:** 0.855

**Description:** Applying class weights to balance the dataset did not significantly improve accuracy.

### 5. Logistic Regression with Liblinear Solver
**Model:** `LogisticRegression(penalty='l2', max_iter=500, C=1, solver='liblinear', random_state=42)`  
**Method:** `get_preds(test_feat, train_feat, y_test, y_train, LogisticRegression(penalty='l2', max_iter=500, C=1, solver='liblinear', random_state=42))`  
**Accuracy:** 0.8544

**Description:** Changing the solver to 'liblinear' had a negligible effect on accuracy.

### 6. Logistic Regression with Saga Solver (Unsuccessful Run)
**Model:** `LogisticRegression(penalty='l2', max_iter=1000, C=0.5, solver='saga', class_weight='balanced', random_state=42)`  
**Method:** `get_preds(test_feat, train_feat, y_test, y_train, LogisticRegression(penalty='l2', max_iter=1000, C=0.5, solver='saga', class_weight='balanced', random_state=42))`  
**Status:** **Not able to run**

**Description:** Attempted to use the 'saga' solver with a longer maximum iteration limit and balanced class weights. Unfortunately, this model configuration failed to execute properly.

## Conclusion

The best model, based on accuracy, was the Logistic Regression with `C=0.1`, achieving an accuracy of 0.875. Future work could include further tuning, exploring different models, or addressing the issue with the 'saga' solver.
