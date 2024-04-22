# AI/ML in Financial Services: Credit Risk Prediction using XGBoost

Author: Edouard Ratiarson

## Introduction

In this project, the focus was on creating a machine learning model to predict credit risk using the XGBoost algorithm. The initiative aimed to grasp the practical application and intricate data relationships vital for financial analytics.

## Executive Summary

The assignment's core was to build a predictive model with XGBoost that encompassed data preprocessing, feature engineering, and hyperparameter tuning. This model serves as a hands-on experience in the prediction of financial risk.

## Achievements and Insights

The initial iteration of the model yielded a ROC-AUC score of 0.736 and an accuracy of 78%. Post-tuning, these metrics improved significantly, showcasing the efficacy of the selected parameters and data handling strategies in enhancing prediction accuracy.

## Data Management and Model Training Process

- Loaded the dataset from a public GitHub repository and conducted preliminary data exploration.
- Employed `train_test_split` for dataset division, ensuring a robust testing framework.
- Trained the XGBoost classifier with initial parameters to establish a baseline model.
- Utilized `accuracy_score`, `confusion_matrix`, and `classification_report` for model evaluation.
- Explored feature importance to guide future feature engineering and model improvement efforts.

## Model Evaluation and Optimization

- Applied techniques like log transformation to normalize skewed features, improving the model's interpretability of the data.
- Handled class imbalance with SMOTE to refine the model's sensitivity to the minority class.
- Conducted a grid search to find the optimal set of hyperparameters, significantly boosting the model's performance.

## Conclusion and Future Steps

The enhanced XGBoost model demonstrated its robustness in predicting credit risk with an improved ROC-AUC score and a more balanced classification approach. The project underscored the importance of combining thorough data preprocessing with meticulous hyperparameter tuning to refine the predictive power of machine learning models in financial applications.

### Additional Optimizations for Future Exploration

To further improve the model, extended grid search, random search, Bayesian optimization, and more intricate cross-validation strategies will be considered. These approaches, along with a re-evaluation of feature importance and model complexity, will ensure a delicate balance between overfitting and predictive accuracy.

## Repository Contents

This repository contains the Jupyter notebook `EdouardRatiarson-AI4FS-XGBoost.ipynb` detailing all the code and processes involved in the model's development.

For further details on the model's performance and the techniques used, please refer to the notebook.

