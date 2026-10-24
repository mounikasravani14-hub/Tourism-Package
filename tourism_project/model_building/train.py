
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# MLflow configuration
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Visit_with_Us_Tourism_Prediction")

# Load training and testing data
data_dir = "tourism_project/data/prepared"

X_train = pd.read_csv(f"{data_dir}/X_train.csv")
X_test = pd.read_csv(f"{data_dir}/X_test.csv")
y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

# Define numerical and categorical features
numerical_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus","Designation"
]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

# Create pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_model)
])

# Hyperparameter grid
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.05, 0.1]
}

# Create deployment directory
os.makedirs("tourism_project/deployment", exist_ok=True)

# Train model and track experiments
with mlflow.start_run(run_name="XGBoost_Hyperparameter_Tuning"):

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    # Log each hyperparameter combination as a nested run
    results = grid_search.cv_results_

    for i, params in enumerate(results["params"]):
        with mlflow.start_run(
            run_name=f"XGBoost_Candidate_{i + 1}",
            nested=True
        ):
            mlflow.log_params(params)
            mlflow.log_metric(
                "mean_test_f1",
                float(results["mean_test_score"][i])
            )
            mlflow.log_metric(
                "std_test_f1",
                float(results["std_test_score"][i])
            )
            mlflow.log_metric(
                "rank_test_f1",
                float(results["rank_test_score"][i])
            )

    # Log best parameters and CV score
    mlflow.log_params({
        f"best_{key}": value
        for key, value in grid_search.best_params_.items()
    })
    mlflow.log_metric("best_cv_f1", float(grid_search.best_score_))

    # Select the best model
    best_model = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)
    print("Best CV F1-score:", grid_search.best_score_)

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Classification metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_precision": precision_score(
            y_train, y_pred_train, zero_division=0
        ),
        "train_recall": recall_score(
            y_train, y_pred_train, zero_division=0
        ),
        "train_f1": f1_score(
            y_train, y_pred_train, zero_division=0
        ),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(
            y_test, y_pred_test, zero_division=0
        ),
        "test_recall": recall_score(
            y_test, y_pred_test, zero_division=0
        ),
        "test_f1": f1_score(
            y_test, y_pred_test, zero_division=0
        )
    }

    mlflow.log_metrics(metrics)
    print("Evaluation metrics:", metrics)

    # Save model for deployment
    model_path = "tourism_project/deployment/best_model.joblib"
    joblib.dump(best_model, model_path)

    # Log model artifacts
    mlflow.log_artifact(model_path, artifact_path="model")

    # Log model in MLflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="xgboost_model",
        input_example=X_test.head(5)
    )

    print("Model saved successfully:", model_path)
    print("MLflow experiment tracking completed!")
