# @Title: Machine Learning for Machine Failure dataset
# @Date: 6/18/2025, last mod: 6/25/2025
# @Author: Logan Powser
# @Abstract: ML functions for Machine Failure dataset from Kaggle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from cleaning import load_data, clean_data, create_features
import mlflow
import mlflow.sklearn
import mlflow.data
import pickle

print("👋 run_experiments.py is running under __main__")

def get_model(model_type:str, model_params:dict):
    params = model_params.copy()
    if model_type == 'logistic':
        return LogisticRegression(**params)
    elif model_type == 'decision_tree':
        return DecisionTreeClassifier(**params)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def run_experiment(model_type:str, model_params:dict, feature_sets:list[str]=["base"], experiment_name:str="machine_failure_prediction"):
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
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name)

    #generate run name
    if feature_sets:
        feature_str = "_".join(feature_sets)
        run_name = f"{model_type}_{feature_str}"
    else:
        run_name = f"{model_type}_base_features"

    #turn on MLFlow autologging for SKLEARN models
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

        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=0)

        cv_results = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=5), scoring='accuracy')
        print(f"CV Results: {cv_results}")
        print(f"Mean Accuracy: {cv_results.mean()}")
        print(f"Standard Deviation: {cv_results.std()}")

        model.fit(X_train, y_train)
        with open(f'prediction_api/model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return model, cv_results.mean(), (X_test, y_test)

def final_testing(model, X_test_data, y_test_data):
    with mlflow.start_run(run_name="final_testing") as run:
        full_test_data = pd.concat([X_test_data, y_test_data], axis=1)
        full_test_data.to_csv('prediction_api/data/test_data.csv', index=False)

        model.predict(X_test_data)
        y_pred = model.predict(X_test_data)

        metrics = {
            'accuracy': accuracy_score(y_test_data, y_pred),
            'f1': f1_score(y_test_data, y_pred),
            'precision': precision_score(y_test_data, y_pred),
            'recall': recall_score(y_test_data, y_pred)
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy Score: {metrics['accuracy']}")
        print(f"F1 Score: {metrics['f1']}")
        print(f"Precision Score: {metrics['precision']}")
        print(f"Recall Score: {metrics['recall']}")
        return metrics, run