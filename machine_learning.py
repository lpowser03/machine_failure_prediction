# @Title: Machine Learning for Machine Failure dataset
# @Date: 6/18/2025, last mod: 6/25/2025
# @Author: Logan Powser
# @Abstract: ML functions for Machine Failure dataset from Kaggle

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from cleaning import load_data, clean_data, create_features
import mlflow
import mlflow.sklearn

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

def run_experiment(model_type:str, model_params:dict, feature_sets:list[str]=["base"], experiment_name:str="Machine_Failure_Prediction"):
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
        metrics = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score)
        }

        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=0)

        cv_results = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=5), scoring='accuracy')
        print(f"CV Results: {cv_results}")
        print(f"Mean Accuracy: {cv_results.mean()}")
        print(f"Standard Deviation: {cv_results.std()}")

        model.fit(X_train, y_train)

        return model, cv_results.mean(), (X_test, y_test)

def final_testing(model, X_test_data, y_test_data):
    model.predict(X_test_data)
    y_pred = model.predict(X_test_data)
    print(f"Accuracy Score: {accuracy_score(y_test_data, y_pred)}")
    print(f"F1 Score: {f1_score(y_test_data, y_pred)}")
    print(f"Precision Score: {precision_score(y_test_data, y_pred)}")
    print(f"Recall Score: {recall_score(y_test_data, y_pred)}")