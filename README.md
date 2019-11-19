# LoanDefaultClassification
Exploring how different models work in loan default classification.

Treshold value for classification is calculated by maximizing sensitivity and specificity on the training set. Includes functions for plotting ROC and calculating AUC and for plotting variable importances.

Logistic regression feature importances:
![logistic](https://github.com/KaroRonty/LoanDefaultClassification/blob/master/logistic_feature_importances.png)

XGBoost feature importances:
![xgboost](https://github.com/KaroRonty/LoanDefaultClassification/blob/master/xgboost_feature_importances.png)