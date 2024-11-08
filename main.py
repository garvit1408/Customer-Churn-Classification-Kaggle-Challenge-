from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pandas as pd
import numpy as np



# --------------------------------------------

# Best Score :  

# Local                                  Kaggle

# 1) 0.9335 288786096121                 0.93515

# --------------------------------------------


# Load the datasets
train_df = pd.read_csv('train.csv')  # Replace with the actual path
test_df = pd.read_csv('test.csv')  # Load your test dataset

# Separating features and target from the training data
X = train_df.drop(columns=["Exited", "id", "CustomerId", "Surname"])
y = train_df["Exited"]

# Identify categorical and numerical features
categorical_features = ["Geography", "Gender"]
numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
# numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",, "EstimatedSalary"]

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# Create a pipeline with preprocessing and XGBoost model
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier(
        eval_metric='auc',
        random_state=0
    ))
])

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    # 'classifier__n_estimators': [100],
    # 'classifier__max_depth': [3],
    # 'classifier__learning_rate': [0.085],
    # 'classifier__subsample': [0.9],
    # 'classifier__colsample_bytree': [0.95],


    'classifier__n_estimators': [200],  # Experiment with different numbers of trees
    'classifier__max_depth': [3],  # Test deeper trees
    'classifier__learning_rate': [0.05],  # Explore different learning rates
    'classifier__subsample': [0.8],  # Experiment with different fractions of data
    'classifier__colsample_bytree': [0.8],  # Test different fractions of features
    'classifier__gamma': [0.1],  # Control complexity of trees
    'classifier__min_child_weight': [3], 
}


# Configure RandomizedSearchCV
random_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=param_grid,
    n_iter=1,  # Number of parameter settings sampled
    scoring='roc_auc',
    cv=3,  # 3-fold cross-validation
    verbose=3,
    random_state=0,
    n_jobs=-1  # Use all available cores
)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# Run the random search on training data
random_search.fit(X_train, y_train)

# Best parameters and AUC-ROC score on validation set
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Predict probabilities on the validation set and calculate AUC-ROC
y_val_probs = best_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_probs)

print("Best Parameters:", best_params)
print("Best AUC-ROC Score:", roc_auc)

# Prepare to make predictions on the test dataset
X_test = test_df.drop(columns=["id", "CustomerId", "Surname"])  # Exclude irrelevant columns
# Use the best model found in RandomizedSearchCV to predict probabilities
predictions = best_model.predict_proba(X_test)[:, 1]  # Probability for class 1 (churn)

# Prepare the submission dataframe
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Exited': predictions
})

# Save the submission file
submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")
