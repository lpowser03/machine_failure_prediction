# @Title: Machine Learning for Machine Failure dataset
# @Date: 6/18/2025, last mod: 6/19/2025
# @Author: Logan Powser
# @Abstract: ML functions for Machine Failure dataset from Kaggle

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from cleaning import load_data, clean_data
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

mlflow.set_tracking_uri('http://localhost:8000')
mlflow.set_experiment('Machine_Failure_Prediction')

def get_model(model_type, model_params):
    pass

def run_experiment(model_type, model_params, feature_sets=None, experiment_name="Machine_Failure_Prediction"):
    """
    Run a complete MLFlow experiment with specified model and features

    Parameters:
    -----------
    model_type : str
        Type of model to use (e.g., 'logistic', 'decision_tree', 'random_forest')
    model_params : dict
        Parameters for the model
    feature_sets : list or None
        List of feature engineering techniques to apply
    experiment_name : str
        Name of the MLFlow experiment
    """
    mlflow.set_experiment(experiment_name)

    #generate run name
    if feature_sets:
        feature_str = "_".join(feature_sets)
        run_name = f"{model_type}_{feature_str}"
    else:
        run_name = f"{model_type}_base_features"

    mlflow.sklearn.autolog()
    # Create MLFlow run
    with mlflow.start_run(run_name=run_name):

        df = load_data()
        X_raw, y = clean_data(df)

        # Apply feature engineering
        X, feature_details = create_features(X_raw, feature_sets)

        # Create model
        model = get_model(model_type, model_params)

        # Train and evaluate with cross-validation
        metrics = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)
        }

        cv_results = cross_val_score(model, X, y, cv=KFold(n_splits=5), scoring=metrics)
        # print(f"CV Results: {cv_results}")
        # print(f"Mean Accuracy: {cv_results.mean()}")
        # print(f"Standard Deviation: {cv_results.std()}")


def create_features(df:pd.DataFrame, feature_set="base"):
    """
    Create engineered features based on specified feature sets.

    Parameters:
    -----------
    df : pandas DataFrame
        Original dataframe with raw features
    feature_sets : list or None
        List of feature engineering techniques to apply.
        If None, only returns base features.

    Returns:
    --------
    X : pandas DataFrame
        DataFrame with original and engineered features
    feature_details : dict
        Dictionary with information about which features were created
    """
    # Start with a copy of the original data
    X = df.copy()

    # Initialize tracking dictionary
    feature_details = {
        "base_features": list(X.columns),
        "engineered_features": [],
        "applied_techniques": []
    }

    # If no feature sets specified, return base features only
    if feature_set is None:
        return X, feature_details

    # Apply specified feature engineering techniques
    for technique in feature_set:
        if technique == "safety_indicators":
            # You'll implement the actual feature engineering here

            # Track the technique and resulting features
            feature_details["applied_techniques"].append("safety_indicators")
            feature_details["engineered_features"].extend(["is_VOC_safe", "is_USS_safe", "is_AQ_safe"])

        elif technique == "temperature_interactions":
            # Implement temperature interaction features
            # ...

            feature_details["applied_techniques"].append("temperature_interactions")
            feature_details["engineered_features"].extend(["temp_ip_ratio", "temp_deviation"])

        # Add more techniques as needed

    return X, feature_details