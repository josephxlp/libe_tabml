
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
import torch

from utilsdf import remove_outlier

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

def try_gpu_or_cpu(model_type, X_train, y_train, X_valid, y_valid, num_rounds=10000):
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
            return train_xgboost(X_train, y_train, X_valid, y_valid, num_rounds, gpu=True)
        elif model_type == "lightgbm":
            return train_lightgbm(X_train, y_train, X_valid, y_valid, num_rounds, gpu=True)
        elif model_type == "catboost":
            return train_catboost(X_train, y_train, X_valid, y_valid, num_rounds, gpu=True)
    except Exception as e:
        print(f"GPU training failed for {model_type}: {e}. Falling back to CPU.")
        if model_type == "xgboost":
            return train_xgboost(X_train, y_train, X_valid, y_valid, num_rounds, gpu=False)
        elif model_type == "lightgbm":
            return train_lightgbm(X_train, y_train, X_valid, y_valid, num_rounds, gpu=False)
        elif model_type == "catboost":
            return train_catboost(X_train, y_train, X_valid, y_valid, num_rounds, gpu=False)

# XGBoost training function
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

# CatBoost training function
def train_catboost(X_train, y_train, X_valid, y_valid, num_rounds=10000, gpu=False):
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    model = CatBoostRegressor(
        iterations=num_rounds,
        task_type="GPU" if gpu else "CPU",
        devices="0:1" if gpu else "",
        early_stopping_rounds=100,
        ctr_target_border_count=10 ,
        use_best_model=True,
        verbose=False,
        eval_metric='RMSE',
        #verbose=100
    )
    model.fit(train_pool, eval_set=valid_pool)
    y_pred = model.predict(X_valid)
    return model, np.sqrt(mean_squared_error(y_valid, y_pred)), r2_score(y_valid, y_pred)

# Train and evaluate a model
def train_model(train_data, valid_data, target_col, features_col, dataset_name, model_type="catboost", num_rounds=10000):
    X_train, y_train, X_valid, y_valid = prepare_data(train_data, valid_data, target_col, features_col)
    model, rmse, r2 = try_gpu_or_cpu(model_type, X_train, y_train, X_valid, y_valid, num_rounds)
    model_path = f"{dataset_name}_{model_type}_{str(num_rounds)}_model.txt"
    model.save_model(model_path) if model_type == "catboost" else model.save_model(model_path)
    return {"data": dataset_name, "RMSE": rmse, "R2": r2, "modelpath": model_path}

# Compare training results across datasets
def train_and_compare(df, target_col, features_col, model_type="catboost", num_rounds=10000):
    train, valid = train_test_split(df, test_size=0.2, random_state=43)
    d1 = remove_outlier(train, target_col, approach='zscore', threshold=3)
    d2 = remove_outlier(train, target_col, approach='iqr')
    d3 = remove_outlier(train, target_col, approach='percentile', lower_percentile=0.05, upper_percentile=0.95)
    results = []
    datasets = [("Original", train), ("Z-Score", d1), ("IQR", d2), ("Percentile", d3)]
    for name, dataset in datasets:
        result = train_model(dataset, valid, target_col, features_col, name, model_type, num_rounds)
        results.append(result)
    return pd.DataFrame(results)

# Example Usage
# features_col = [col for col in df.columns if col != "target"]

