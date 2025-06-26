# @Title: Data Cleaning for Machine Failure dataset
# @Date: 6/18/2025, last mod: 6/19/2025
# @Author: Logan Powser
# @Abstract: Data cleaning functions for Machine Failure dataset from Kaggle

import pandas as pd

def load_data():
    return pd.read_csv('data/data.csv')

def clean_data(df:pd.DataFrame):
    df = df.drop(columns=['footfall'])
    y = df['fail']
    X = df.drop(columns=['fail'])
    return X, y


def create_features(df: pd.DataFrame, feature_sets: list[str] = "base"):
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
    if feature_sets is None:
        return X, feature_details

    # Apply specified feature engineering techniques
    for technique in feature_sets:
        if technique == "safety_indicators":
            is_AQ_safe = X["AQ"] == 1 | X["AQ"] == 2
            is_USS_safe = X["USS"] == 6 | X["USS"] == 7
            is_VOC_safe = X["VOC"] == 0 | X["VOC"] == 2
            safety_score = is_AQ_safe + is_USS_safe + is_VOC_safe

            X["is_AQ_safe"] = is_AQ_safe
            X["is_USS_safe"] = is_USS_safe
            X["is_VOC_safe"] = is_VOC_safe
            X["safety_score"] = safety_score

            # Track the technique and resulting features
            feature_details["applied_techniques"].append("safety_indicators")
            feature_details["engineered_features"].extend(["is_VOC_safe", "is_USS_safe", "is_AQ_safe", "safety_score"])

        elif technique == "temperature_interactions":
            temp_mode_ratio = X["Temperature"] / X["tempMode"]
            temp_ip_interaction = X["IP"] * X["Temperature"]

            X["temp_mode_ratio"] = temp_mode_ratio
            X["temp_ip_interaction"] = temp_ip_interaction

            feature_details["applied_techniques"].append("temperature_interactions")
            feature_details["engineered_features"].extend(["temp_ip_interaction", "temp_mode_ratio"])

        elif technique == "operational_modes":
            cs_uss_diagonal = X['CS'] == X['USS']
            rp_mode_ratio = X['RP'] / X['tempMode']

            X['cs_uss_diagonal'] = cs_uss_diagonal
            X['rp_mode_ratio'] = rp_mode_ratio

            feature_details["applied_techniques"].append("operational_modes")
            feature_details["engineered_features"].extend(["cs_uss_diagonal", "rp_mode_ratio"])

        elif technique == "electrical_system":
            high_current_load = (X['CS'] > 5)
            high_rotation_stress = (X['RP'] > 65) & (X['Temperature'] > 15)

            X['high_current_load'] = high_current_load
            X['high_rotation_stress'] = high_rotation_stress

            feature_details["applied_techniques"].append("electrical_system")
            feature_details["engineered_features"].extend(["high_current_load", "high_rotation_stress"])

    return X, feature_details