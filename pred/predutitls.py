import os
import time
import rasterio
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from autogluon.tabular import TabularPredictor
import sys
from paths import libpath
sys.path.append(libpath)
from utilsdf import check_fillnulls 
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint


def write_predictions(predictions, tile_file, output_raster_path, block_size=(128, 128)):
    """
    Writes predictions to a new raster file using metadata from an existing tile file in blocks,
    optimised for large rasters.

    Parameters:
    - predictions (array-like): 1D array of predicted values matching the flattened raster size.
    - tile_file (str): Path to the raster file from which metadata will be read.
    - output_raster_path (str): Path where the new raster file will be saved.
    - block_size (tuple): Tuple specifying the block size for processing (default is (128, 128)).

    Returns:
    - None
    """
    with rasterio.open(tile_file) as src:
        meta = src.meta.copy()
        raster_shape = (src.height, src.width)

    try:
        predictions_reshaped = np.array(predictions).reshape(raster_shape)
    except ValueError:
        raise ValueError(f"Predictions array size {len(predictions)} does not match raster dimensions {raster_shape}.")

    meta.update({
        "dtype": rasterio.float32,
        "count": 1,
        "compress": "lzw"
    })

    with rasterio.open(output_raster_path, "w", **meta) as dst:
        print(f"Writing raster in blocks of size: {block_size}")
        for y_start in range(0, raster_shape[0], block_size[0]):
            for x_start in range(0, raster_shape[1], block_size[1]):
                y_end = min(y_start + block_size[0], raster_shape[0])
                x_end = min(x_start + block_size[1], raster_shape[1])
                window = ((y_start, y_end), (x_start, x_end))
                block_data = predictions_reshaped[y_start:y_end, x_start:x_end]
                dst.write(block_data.astype(rasterio.float32), 1, window=rasterio.windows.Window.from_slices(*window))
    print(f"Raster written successfully to {output_raster_path}")

def get_tile_file(tile_files):
    tile_file = [i for i in tile_files if i.endswith('edem_W84.tif')]
    assert len(tile_file) == 1, 'len(tile_file) != 1'
    return tile_file[0]

def load_prediction_data(fparquet, fcol):
    df = pd.read_parquet(fparquet)
    df[fcol] = check_fillnulls(df[fcol])
    return df

def load_cb_model(model_path):
    """
    Loads a CatBoost model from the given path.
    
    Parameters:
        model_path (str): Path to the CatBoost model file.
    
    Returns:
        CatBoostRegressor: Loaded CatBoost model.
    """
    assert os.path.exists(model_path), f"{model_path} does not exist"
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model


def get_prediction_df(model, df, fcol, yvar, tcol):
    df[f'ml_{yvar}'] = model.predict(Pool(df[fcol]))
    df[f'ml_{tcol}'] = df[tcol] - df[f'ml_{yvar}']
    return df

def cb_predict_workflow(outdir, modelpath, fparquet, tile_files, fcol, yvar, tcol, ps, bsize=256):
    start_time = time.time()
    tile_ifile = get_tile_file(tile_files)
    tile_odir = os.path.join(outdir, tile_ifile.split('/')[-2])
    os.makedirs(tile_odir, exist_ok=True)
    tile_ofile = os.path.join(tile_odir, tile_ifile.split('/')[-1].replace('.tif', '_ML.tif'))

    if not os.path.isfile(tile_ofile):
        print(f"Writing: {tile_ofile}...")
        df = load_prediction_data(fparquet, fcol)
        assert len(df) == ps * ps, 'Grid size does not match'
        model = load_cb_model(modelpath)
        df = get_prediction_df(model, df, fcol, yvar, tcol)

        write_predictions(predictions=df[f'ml_{tcol}'],
                          tile_file=tile_ifile,
                          output_raster_path=tile_ofile,
                          block_size=(bsize, bsize))
    else:
        print(f"{tile_ofile} already exists")

    elapsed_time = time.time() - start_time
    print(f"Time for cb_predict_workflow: {elapsed_time:.2f} seconds")
    return elapsed_time


def ag_mbest_predict_workflow(outdir, dirname, modelpath, 
                              fparquet, tile_files, 
                              fcol, yvar, tcol, ps, bsize=256):
    start_time = time.time()
    """
    Workflow to perform predictions using a pre-trained model. The function checks 
    the model before loading the data to avoid unnecessary costs if the model is invalid.

    Args:
        outdir (str): Output directory path.
        dirname (str): Directory name for organising outputs.
        modelpath (str): Path to the saved model.
        fparquet (str): Path to the input parquet file containing prediction data.
        tile_files (list): List of tile files.
        fcol (str): Feature column used for predictions.
        yvar (str): Variable to predict.
        tcol (str): Target column.
        ps (int): Grid size parameter.
        bsize (int, optional): Block size for raster output. Defaults to 256.

    Returns:
        None
    """
    # Step 1: Load and validate the model
    try:
        predictor = TabularPredictor.load(modelpath)
    except Exception as e:
        raise RuntimeError(f"Failed to load the model from {modelpath}: {e}")

    # Perform model validation checks
    # Example: Check for necessary metadata or expected feature columns
    expected_features = fcol
    model_features = predictor.feature_metadata.get_features()
    print(f"Model features: {model_features}")
    print('-'*50)
    if not all(feature in model_features for feature in expected_features):
        raise ValueError(f"Model validation failed: Missing required feature columns {expected_features}")

    # If the model passes validation, proceed
    print("Model loaded and validated successfully.")

    # Step 2: Prepare output paths
    outpath = os.path.join(outdir, dirname)
    tile_ifile = get_tile_file(tile_files)
    tile_odir = os.path.join(outpath, tile_ifile.split('/')[-2])
    os.makedirs(tile_odir, exist_ok=True)
    tile_ofile = os.path.join(tile_odir, tile_ifile.split('/')[-1].replace('.tif', '_ML.tif'))

    # Step 3: Check if output already exists
    if not os.path.isfile(tile_ofile):
        print(f"Writing: {tile_ofile} ...")
        
        # Step 4: Load prediction data
        df = load_prediction_data(fparquet, fcol)
        if len(df) != ps * ps:
            raise ValueError(f"Grid size mismatch: Expected {ps * ps}, got {len(df)}")

        # Step 5: Perform predictions and write results
        df[f'ml_{yvar}'] = predictor.predict(df[fcol])
        df[f'ml_{tcol}'] = df[tcol] - df[f'ml_{yvar}']

        write_predictions(predictions=df[f'ml_{tcol}'],  # Residuals
                          tile_file=tile_ifile,
                          output_raster_path=tile_ofile,
                          block_size=(bsize, bsize))
    else:
        print(f"{tile_ofile} already exists")

    elapsed_time = time.time() - start_time
    print(f"Time for cb_predict_workflow: {elapsed_time:.2f} seconds")
    return elapsed_time

#------------------------------------------------------------------------------------------------#
def parallel_cbe_prediction_workflow(outdir, model_list, dirname, 
                                     fparquet_list, tile_files_list, 
                                     fcol, yvar, tcol, ps, bsize=256, max_workers=4):
    """
    Parallelises the cbe_predict_workflow_vb function for multiple Parquet and tile file pairs.

    Parameters:
    - outdir (str): Output directory path.
    - model_list (list): List of paths to CatBoost model files.
    - dirname (str): Name of the subdirectory for the output.
    - fparquet_list (list): List of Parquet file paths.
    - tile_files_list (list): List of tile file paths.
    - fcol (str): Name of the feature column.
    - yvar (str): Name of the target variable.
    - tcol (str): Name of the terrain column.
    - ps (int): Grid size.
    - bsize (int): Block size for writing raster files.
    - max_workers (int): Maximum number of parallel workers.

    Returns:
    - None
    """

    def process_file_pair(fparquet, tile_files):
        """Wrapper for processing a single Parquet and tile file pair."""
        print(f'Processing: {fparquet}')
        cbe_predict_workflow_vb(
            outdir=outdir,
            model_list=model_list,
            dirname=dirname,
            fparquet=fparquet,
            tile_files=tile_files,
            fcol=fcol,
            yvar=yvar,
            tcol=tcol,
            ps=ps,
            bsize=bsize
        )
        print(f'Finished: {fparquet}')

    # Create a process pool for parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the pool
        future_to_filepair = {
            executor.submit(process_file_pair, fparquet, tile_files): (fparquet, tile_files)
            for fparquet, tile_files in zip(fparquet_list, tile_files_list)
        }

        # Monitor progress
        for future in as_completed(future_to_filepair):
            fparquet, tile_files = future_to_filepair[future]
            try:
                future.result()  # Raise exceptions if any occurred during processing
            except Exception as e:
                print(f"Error processing {fparquet}: {e}")


def cbe_predict_workflow_vb(outdir, model_list, dirname, fparquet, tile_files, 
                            fcol, yvar, tcol, ps, bsize=256):
    """
    Workflow for prediction that accepts a list of models rather than a directory.
    
    Parameters:
    - outdir (str): Output directory path.
    - model_list (list): List of paths to CatBoost model files.
    - dirname (str): Name of the subdirectory for the output.
    - fparquet (str): Path to the Parquet file containing input data.
    - tile_files (list): List of tile files.
    - fcol (str): Name of the feature column.
    - yvar (str): Name of the target variable.
    - tcol (str): Name of the terrain column.
    - ps (int): Grid size.
    - bsize (int): Block size for writing raster files. Default is 256.

    Returns:
    - None
    """
    tile_ifile = get_tile_file(tile_files)
    tile_odir = os.path.join(outdir, dirname, tile_ifile.split('/')[-2])
    os.makedirs(tile_odir, exist_ok=True)
    tile_ofile = os.path.join(tile_odir, tile_ifile.split('/')[-1].replace('.tif', '_ML.tif'))

    if not os.path.isfile(tile_ofile):
        print(f'Writing: {tile_ofile} ...')
        df = load_prediction_data(fparquet, fcol)
        assert len(df) == ps * ps, 'Grid size does not match'

        # Load each model and apply predictions sequentially
        for model_path in model_list:
            assert os.path.exists(model_path), f'{model_path} does not exist'
            model = load_cb_model(model_path)
            df = get_prediction_df(model, df, fcol, yvar, tcol)

        write_predictions(predictions=df[f'ml_{tcol}'],  #
                          tile_file=tile_ifile, 
                          output_raster_path=tile_ofile, 
                          block_size=(bsize, bsize))
    else:
        print(f'{tile_ofile} already exists')


def predict_with_models(model_dir, df, fcol, yvar, tcol):
    """
    Predicts using all CatBoost models in the specified directory, averages the predictions,
    and calculates residuals. Returns the updated dataframe.

    Parameters:
        model_dir (str): Path to the directory containing the model files.
        df (pd.DataFrame): Dataframe containing the features and true values.
        fcol (list): List of feature column names to use as predictors.
        yvar (str): Target variable name for predictions.
        tcol (str): Column name of the true target values to calculate residuals.

    Returns:
        pd.DataFrame: Updated dataframe with predictions and residuals.
    """
    # Get all model files ending with "_model.txt"
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith("_model.txt")]
    print('#--------------------------------------------------#')
    pprint(model_files)
    assert model_files, f"No model files found in {model_dir}"

    # Create a CatBoost Pool for the input data
    data_pool = Pool(df[fcol])

    # Load models and make predictions
    predictions = []
    for model_file in model_files:
        print(f"Loading model: {model_file}")
        model = load_cb_model(model_file)
        y_pred = model.predict(data_pool)
        predictions.append(y_pred)

    # Calculate the average predictions
    y_pred_avg = sum(predictions) / len(predictions)

    # Add predictions and residuals to the dataframe
    df[f'ml_{yvar}'] = y_pred_avg
    df[f'ml_{tcol}'] = df[tcol].subtract(df[f'ml_{yvar}'])

    return df


def cbe_predict_workflow(outdir,model_dir,dirname,fparquet,tile_files,
                        fcol,yvar,tcol,ps,bsize=256):
    # improve this so that it checks models before loading the data 
    tile_ifile = get_tile_file(tile_files)
    tile_odir = os.path.join(outdir,dirname,tile_ifile.split('/')[-2])
    os.makedirs(tile_odir,exist_ok=True)
    tile_ofile = os.path.join(tile_odir,tile_ifile.split('/')[-1].replace('.tif','_ML.tif'))

    if not os.path.isfile(tile_ofile):
        f'writing:: {tile_ofile} ...'
        df = load_prediction_data(fparquet,fcol)
        assert len(df) == ps * ps, 'Grid size does not match'
        #model = load_cb_model(modelpath)
        #df = get_prediction_df(model,df,fcol,yvar,tcol)
        df = predict_with_models(model_dir, df, fcol, yvar, tcol)

        
        write_predictions(predictions=df[f'ml_{tcol}'], #
                        tile_file=tile_ifile, 
                        output_raster_path=tile_ofile, 
                        block_size=(bsize, bsize))
    else:
        print(f'{tile_ofile} already exists')

 