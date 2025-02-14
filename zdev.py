import os
import pickle
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

def load_parquet_files(file_list):
    return pd.read_parquet(file_list)

def prepare_data(train_files, valid_files, features_col, target_col):
    train_df = load_parquet_files(train_files)
    valid_df = load_parquet_files(valid_files)
    
    X_train, y_train = train_df[features_col], train_df[target_col]
    X_valid, y_valid = valid_df[features_col], valid_df[target_col]
    
    return X_train, y_train, X_valid, y_valid

def train_catboost(X_train, y_train, X_valid, y_valid, num_rounds=10000, seed=42, gpu=True):
    model = CatBoostRegressor(
        iterations=num_rounds, 
        random_seed=seed,
        task_type='GPU' if gpu else 'CPU', 
        verbose=500
    )
    
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), 
              early_stopping_rounds=100, verbose=500)
    
    y_pred = model.predict(X_valid)
    rmse = ((y_valid - y_pred) ** 2).mean() ** 0.5
    r2 = model.score(X_valid, y_valid)
    
    return model, rmse, r2

def save_model(model, output, fname):
    output_dir = os.path.join(output, fname)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{fname}_model.cbm')
    feature_importance_path = os.path.join(output_dir, f'{fname}_feature_importance.csv')
    evals_result_path = os.path.join(output_dir, f'{fname}_evals_result.pkl')
    
    model.save_model(model_path)
    pd.DataFrame(model.get_feature_importance(prettified=True)).to_csv(feature_importance_path, index=False)
    
    with open(evals_result_path, 'wb') as f:
        pickle.dump(model.get_evals_result(), f)
    
    return model_path

def train_and_save_model(train_files, valid_files, target_col, feature_cols, output_dir, model_name, num_rounds=10000, seed=42):
    X_train, y_train, X_valid, y_valid = prepare_data(train_files, valid_files, feature_cols, target_col)
    model, rmse, r2 = train_catboost(X_train, y_train, X_valid, y_valid, num_rounds, seed)
    model_path = save_model(model, output_dir, model_name)
    
    metrics_df = pd.DataFrame([{"Seed": seed, "RMSE": rmse, "R2": r2, "ModelPath": model_path}])
    metrics_df.to_csv(os.path.join(output_dir, f'{model_name}_metrics.csv'), index=False)
    
    return {"Seed": seed, "RMSE": rmse, "R2": r2, "ModelPath": model_path}



# pass the variables as dictionary here 
def train_all_models(path, output_dir, num_rounds=10000, seed=42):
    files = glob(f'{path}/*parquet')
    train_files, test_files = train_test_split(files, random_state=42, test_size=0.1)
    train_files, valid_files = train_test_split(train_files, random_state=42, test_size=0.1)
    
    print(f'files train {len(train_files)} valid={len(valid_files)} test={len(test_files)}')
    # pass the variables as dictionary here 
    s1_cols = ['multi_s1_band1', 'multi_s1_band2']
    s2_cols = ['multi_s2_band1', 'multi_s2_band2', 'multi_s2_band3']
    r_col = 'edem_w84'
    y_col, t_col = 'zdif', 'multi_dtm_lidar'
    
    models = {}
    models['model_s1id'] = train_and_save_model(train_files, valid_files, y_col, s1_cols + [r_col], output_dir, f'model_s1id_{y_col}_{num_rounds}', num_rounds, seed)
    models['model_s1di'] = train_and_save_model(train_files, valid_files, t_col, s1_cols + [r_col], output_dir, f'model_s1di_{t_col}_{num_rounds}', num_rounds, seed)
    models['model_s2id'] = train_and_save_model(train_files, valid_files, y_col, s2_cols + [r_col], output_dir, f'model_s2id_{y_col}_{num_rounds}', num_rounds, seed)
    models['model_s2di'] = train_and_save_model(train_files, valid_files, t_col, s2_cols + [r_col], output_dir, f'model_s2di_{t_col}_{num_rounds}', num_rounds, seed)
    
    return models
