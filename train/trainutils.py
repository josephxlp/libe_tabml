import os
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import pickle


#---------------------------------------------------------------------------------------#
# JOHN 3:16
#---------------------------------------------------------------------------------------#




#---------------------------------------------------------------------------------------#
# DATA PREPROCESSING FUNCTIONS
#---------------------------------------------------------------------------------------#



#---------------------------------------------------------------------------------------#
# MDOEL TRAINING FUNCTIONS
#---------------------------------------------------------------------------------------#


def train_model(train_data, valid_data, target_col, features_col, output_dir, 
                model_type="catboost", num_rounds=10000, seed=42,esr=None):
    # Prepare the data for training
    X_train, y_train, X_valid, y_valid = prepare_data(train_data, valid_data, target_col, features_col)
    
    # Train the model and calculate metrics
    model, rmse, r2,best_model_iter = try_gpu_or_cpu(model_type, X_train, y_train, X_valid, y_valid, 
                                                     num_rounds, seed,esr)

    fname = f"{model_type}_{num_rounds}_{seed}"
    # Define paths for model and error metrics
    model_path = os.path.join(output_dir, f"{model_type}_{num_rounds}_{seed}_model.txt")
    error_path = os.path.join(output_dir, f"{model_type}_{num_rounds}_{seed}_metrics.csv")
    save_model_params(model, outdir=output_dir,fname=fname)

    # Save the model
    if model_type == "catboost":
        model.save_model(model_path)
    else:
        model.save_model(model_path)

    # Create a DataFrame for error metrics and save as CSV
    metrics_df = pd.DataFrame([{"Seed": seed, "RMSE": rmse, "R2": r2, 
                                "BestIter": f"{best_model_iter}_of_{num_rounds}",
                                "ModelPath": model_path}])
    metrics_df.to_csv(error_path, index=False)
    
    return metrics_df,best_model_iter

def save_model_params(model, outdir, fname):
    model.get_feature_importance(prettified=True).to_csv(os.path.join(outdir,f'{fname}_feature_importance.csv'))
    model.save_model(os.path.join(outdir,f'{fname}_model.cbm'))
    evals_result = model.get_evals_result()

    # with open('evals_result.pkl', 'rb') as f:
    # evals_result = pickle.load(f)
    # Save evals_result using pickle
    with open(os.path.join(outdir,f'{fname}_evals_result.pkl'), 'wb') as f:
        pickle.dump(evals_result, f)

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


def try_gpu_or_cpu(model_type, X_train, y_train, X_valid, y_valid, 
                   num_rounds=10000,seed=42,esr=None):
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
            return train_catboost(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=True,easr=esr)
    except Exception as e:
        print(f"GPU training failed for {model_type}: {e}. Falling back to CPU.")
        if model_type == "xgboost":
            return train_xgboost(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=False)
        elif model_type == "lightgbm":
            return train_lightgbm(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=False)
        elif model_type == "catboost":
            return train_catboost(X_train, y_train, X_valid, y_valid, num_rounds, seed,gpu=False,esr=esr)

def train_catboost(X_train, y_train, X_valid, y_valid, 
                   num_rounds=10000, seed=42, gpu=False, esr=None):
    """
    Trains a CatBoostRegressor model with the given data.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_valid: Validation feature matrix.
        y_valid: Validation target vector.
        num_rounds: Maximum number of training iterations (default 10,000).
        seed: Random seed for reproducibility (default 42).
        gpu: Whether to use GPU for training (default False).
        esr: Early stopping rounds; if None, disables early stopping (default None).

    Returns:
        model: Trained CatBoostRegressor model.
        rmse: Root Mean Squared Error on the validation set.
        r2: R^2 Score on the validation set.
    """
    # Prepare data pools
    train_pool = cb.Pool(X_train, y_train)
    valid_pool = cb.Pool(X_valid, y_valid)

    # Verbosity interval
    dverbose = max(100, num_rounds // 20)

    # Handle early stopping rounds
    early_stopping_rounds = esr if esr is not None else None

    # Initialise the model
    model = cb.CatBoostRegressor(
        iterations=num_rounds,
        task_type="GPU" if gpu else "CPU",
        devices="0:1" if gpu else "",
        early_stopping_rounds=early_stopping_rounds,
        # ctr_target_border_count: Defines the number of borders for target binarization in CTR (useful for classification tasks)
        use_best_model=True,
        verbose=dverbose,  # Display progress every 100 iterations
        eval_metric="RMSE",
        random_seed=seed,
    )

    # Train the model
    model.fit(train_pool, eval_set=valid_pool)

    # Make predictions and calculate metrics
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r2 = r2_score(y_valid, y_pred)
    best_model_iter = model.get_best_iteration()

    return model, rmse, r2,best_model_iter


#---------------------------------------------------------------------------------------#
# HELPERS FUNCTIONS
#---------------------------------------------------------------------------------------#

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


def measure_time_beautifully(task_description, task_function, *args, **kwargs):
    """
    Measures the execution time of a given function and prints the elapsed time in various units.

    Parameters:
        task_description (str): A description of the task being measured.
        task_function (callable): The function to execute and measure.
        *args: Positional arguments to pass to the task_function.
        **kwargs: Keyword arguments to pass to the task_function.
    """
    # Start the timer
    start_time = time.perf_counter()

    # Execute the task function
    result = task_function(*args, **kwargs)

    # Stop the timer
    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60
    elapsed_days = elapsed_hours / 24

    # Border symbols
    border_char = "=" * 60
    padding_char = " " * 4

    # Print the execution time beautifully
    print(border_char)
    print(f"{padding_char}Task Performance Report")
    print(border_char)
    print(f"{padding_char}Task: {task_description}")
    print(f"{padding_char}Elapsed Time:")
    print(f"{padding_char * 2}{elapsed_seconds:.2f} seconds")
    print(f"{padding_char * 2}{elapsed_minutes:.2f} minutes")
    print(f"{padding_char * 2}{elapsed_hours:.2f} hours")
    print(f"{padding_char * 2}{elapsed_days:.2f} days")
    print(border_char)

    return result

