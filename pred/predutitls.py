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

def load_cb_model(modelpath):
    assert os.path.exists(modelpath), f'{modelpath} does not exist'
    model = CatBoostRegressor()
    model.load_model(modelpath)
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


