# Kaggle Challenge: Churn Prediction with XGBoost

This repository contains code for predicting customer churn in a bank dataset using the **XGBoost** algorithm. The model aims to maximize the **ROC-AUC score** as the primary evaluation metric, with extensive hyperparameter tuning implemented through `RandomizedSearchCV`.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Features](#features)
- [Modeling Approach](#modeling-approach)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Usage](#usage)
- [Submission](#submission)
- [License](#license)

---

## Project Overview

Customer churn prediction is crucial for customer retention in the banking industry. This project uses **XGBoost**, a gradient-boosted decision tree algorithm, to identify customers at risk of leaving. The model was developed and evaluated using a dataset provided by Kaggle, aiming to maximize the **ROC-AUC** score.

## Dataset

The data consists of features describing customer demographics, account balances, product holdings, and transaction activity.

- **Target**: `Exited` (1 if the customer left, 0 otherwise)
- **Features**:
  - Numerical features: `CreditScore`, `Age`, `Balance`, `NumOfProducts`, `EstimatedSalary`, etc.
  - Categorical features: `Geography`, `Gender`
  
**Note**: The dataset includes both a training set for model development and a test set for submission.

## Installation

To run this project, you need Python 3.x and the following libraries:

```bash
pip install pandas numpy scikit-learn xgboost
```

## Features

The feature engineering pipeline processes numerical and categorical data as follows:

- **Numerical Features**: Scaled with StandardScaler.
Categorical Features: Encoded using OneHotEncoder.
The final model uses the preprocessed features for training the XGBoost classifier.

Modeling Approach
The pipeline integrates the following steps:

Data Preprocessing: Categorical features are one-hot encoded, and numerical features are scaled.
Model Selection: The main classifier is XGBClassifier, part of the XGBoost library.
Model Training and Validation: Split the training data into training and validation sets, using stratified split to maintain target distribution.
Hyperparameter Tuning
RandomizedSearchCV is used to explore a wide range of hyperparameters for the XGBoost model, aiming to optimize the ROC-AUC score:

n_estimators
max_depth
learning_rate
subsample
colsample_bytree
gamma
min_child_weight
reg_alpha and reg_lambda (L1 and L2 regularization)
scale_pos_weight (to address class imbalance)
The optimal hyperparameters are selected based on 5-fold cross-validation.

Results
The best ROC-AUC score achieved on the validation set was approximately 0.93.

Usage
To reproduce the results, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/garvit1408/kaggle-churn-prediction.git
cd kaggle-churn-prediction
Download the dataset from Kaggle and place train.csv and test.csv in the project directory.

Run the main.py script to execute the training and generate predictions for submission:

bash
Copy code
python main.py
The script will output the best hyperparameters, validation ROC-AUC score, and save a submission.csv file.

Submission
The generated submission.csv file contains the id and predicted probabilities for the Exited class, formatted for submission to Kaggle.

Sample output format:
