import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
import torch

from utilsdf import remove_outlier
import time

def print_context(text):
    print(' ')
    """
    Prints the given text in a styled format, mimicking SAGA GIS console output.

    Parameters:
        text (str): The text to print.
    """
    # Border symbols
    border_char = "=" * 60
    padding_char = " " * 4

    # Print formatted text
    print(border_char)
    print(f"{padding_char}ML - Process Log")
    print(border_char)

    for line in text.split("\n"):
        print(f"{padding_char}{line}")

    print(border_char)


def prepare_data(train_data, valid_data, target_col, features_col):
    """
    Prepares the training and validation data by separating features and target.

    Parameters:
        train_data (pd.DataFrame): Training data.
        valid_data (pd.DataFrame): Validation data.
        target_col (str): Target column name.
        features_col (list): List of feature column names.

    Returns:
        X_train, y_train, X_valid, y_valid: Features and target data.
    """
    X_train = train_data[features_col]
    y_train = train_data[target_col]
    X_valid = valid_data[features_col]
    y_valid = valid_data[target_col]
    return X_train, y_train, X_valid, y_valid

def try_gpu_or_cpu(model_type, X_train, y_train, X_valid, y_valid, num_rounds=10000,seed=42):
    """
    Attempts to use GPU for training. Falls back to CPU if GPU is unavailable or fails.

    Parameters:
        model_type (str): The model type to use ("xgboost", "lightgbm", "catboost").
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
        num_rounds (int): Number of training iterations.

    Returns:
        model: Trained model.
        rmse (float): Root Mean Squared Error on validation set.
        r2 (float): R-squared score on validation set.
    """
    try:
        if model_type == "xgboost":
            return train_xgboost(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=True)
        elif model_type == "lightgbm":
            return train_lightgbm(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=True)
        elif model_type == "catboost":
            return train_catboost(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=True)
    except Exception as e:
        print(f"GPU training failed for {model_type}: {e}. Falling back to CPU.")
        if model_type == "xgboost":
            return train_xgboost(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=False)
        elif model_type == "lightgbm":
            return train_lightgbm(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=False)
        elif model_type == "catboost":
            return train_catboost(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=False)

def get_dynamic_early_stopping_rounds(num_rounds):
    """
    Calculate dynamic early stopping rounds based on the total number of iterations (num_rounds).
    - Scales proportionally for larger num_rounds (> 1000).
    - Ensures minimum threshold for smaller num_rounds.
    """
    if num_rounds > 1000:
        # Reduce early stopping rounds for very large iterations
        return max(50, int(num_rounds * 0.05))  # 5% of num_rounds, but not less than 50
    elif num_rounds < 500:
        # Scale proportionally for smaller num_rounds
        return max(20, int(num_rounds * 0.1))  # 10% of num_rounds, but not less than 20
    else:
        # Default scaling
        return max(100, int(num_rounds * 0.1))  # 10% of num_rounds
    

def train_catboost(X_train, y_train, X_valid, y_valid, num_rounds=10000, seed=42, gpu=False):
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    #dynamic_early_stopping_rounds = max(100, num_rounds // 20) 
    dynamic_early_stopping_rounds = get_dynamic_early_stopping_rounds(num_rounds)
    #dynamic_early_stopping_rounds = min(1000, max(100, num_rounds // 20))

    dverbose = max(100, num_rounds // 20)

    model = CatBoostRegressor(
        iterations=num_rounds,
        task_type="GPU" if gpu else "CPU",
        devices="0:1" if gpu else "",
        early_stopping_rounds=dynamic_early_stopping_rounds,
        ##ctr_target_border_count=10, # what's this about?
        use_best_model=True,
        verbose=dverbose,  # Display progress every 100 iterations
        eval_metric="RMSE",
        random_seed=seed,
    )
    model.fit(train_pool, eval_set=valid_pool)
    y_pred = model.predict(X_valid)
    return model, np.sqrt(mean_squared_error(y_valid, y_pred)), r2_score(y_valid, y_pred)


def train_model(train_data, valid_data, target_col, features_col, output_dir, 
                model_type="catboost", num_rounds=10000, seed=42):
    # Prepare the data for training
    X_train, y_train, X_valid, y_valid = prepare_data(train_data, valid_data, target_col, features_col)
    
    # Train the model and calculate metrics
    model, rmse, r2 = try_gpu_or_cpu(model_type, X_train, y_train, X_valid, y_valid, num_rounds, seed)

    # Define paths for model and error metrics
    model_path = os.path.join(output_dir, f"{model_type}_{num_rounds}_{seed}_model.txt")
    error_path = os.path.join(output_dir, f"{model_type}_{num_rounds}_{seed}_metrics.csv")

    # Save the model
    if model_type == "catboost":
        model.save_model(model_path)
    else:
        model.save_model(model_path)

    # Create a DataFrame for error metrics and save as CSV
    metrics_df = pd.DataFrame([{"Seed": seed, "RMSE": rmse, "R2": r2, "ModelPath": model_path}])
    metrics_df.to_csv(error_path, index=False)
    
    return {
        "Seed": seed,
        "RMSE": rmse,
        "R2": r2,
        "ModelPath": model_path
    }



# Train and evaluate a model
def train_modelc(train_data, valid_data, target_col, features_col, dataset_name, 
                model_type="catboost", num_rounds=10000,seed=42):
    X_train, y_train, X_valid, y_valid = prepare_data(train_data, valid_data, target_col, features_col)
    model, rmse, r2 = try_gpu_or_cpu(model_type, X_train, y_train, X_valid, y_valid, num_rounds,seed)
    
    model_path = f"{dataset_name}_{model_type}_{str(num_rounds)}_{seed}_model.txt"
    model.save_model(model_path) if model_type == "catboost" else model.save_model(model_path)
    return {"data": dataset_name, "RMSE": rmse, "R2": r2, "modelpath": model_path}

# Compare training results across datasets
def train_and_compare(df, target_col, features_col, model_type="catboost", num_rounds=10000):
    train, valid = train_test_split(df, test_size=0.1, random_state=43)
    d1 = remove_outlier(train, target_col, approach='zscore', threshold=3)
    d2 = remove_outlier(train, target_col, approach='iqr')
    d3 = remove_outlier(train, target_col, approach='percentile', lower_percentile=0.05, upper_percentile=0.95)
    results = []
    datasets = [("Original", train), ("Z-Score", d1), ("IQR", d2), ("Percentile", d3)]
    for name, dataset in datasets:
        result = train_modelc(dataset, valid, target_col, features_col, name, model_type, num_rounds)
        results.append(result)
    return pd.DataFrame(results)

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

# XGBoost training function random_seed, init_params
def train_xgboost(X_train, y_train, X_valid, y_valid, num_rounds=10000, gpu=False):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist" if gpu else "hist",
        "gpu_id": 1 if gpu else -1,  # Use GPU 2 (index 1) or CPU (-1)
        "max_depth": 6,
    }

    model = xgb.train(params, dtrain, num_rounds, evals=[(dvalid, "eval")], early_stopping_rounds=100, verbose_eval=100)
    y_pred = model.predict(dvalid)
    return model, np.sqrt(mean_squared_error(y_valid, y_pred)), r2_score(y_valid, y_pred)

# LightGBM training function
def train_lightgbm(X_train, y_train, X_valid, y_valid, num_rounds=10000, gpu=False):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    params = {
        "objective": "regression",
        "device": "gpu" if gpu else "cpu",
        "gpu_platform_id": 1 if gpu else None,
        "gpu_device_id": 1 if gpu else None,
        "max_depth": 6,
    }

    model = lgb.train(params, train_data, num_rounds, valid_sets=[valid_data], early_stopping_rounds=100, verbose_eval=100)
    y_pred = model.predict(X_valid)
    return model, np.sqrt(mean_squared_error(y_valid, y_pred)), r2_score(y_valid, y_pred)





# Example Usage
# features_col = [col for col in df.columns if col != "target"]

