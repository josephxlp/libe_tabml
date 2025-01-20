from utilsml import train_model
from utilsdf import list_files_by_tilenames, assign_nulls, fillna, dropnulls_bycol, check_fillnulls
from uvars import tilenames_lidar, RES_DPATH, s1_fnames, s2_fnames, aux_names
import pandas as pd
from sklearn.model_selection import train_test_split
import os

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
# X = 90: num_rounds=10000:2min
# X = 30: num_rounds=1000:3min
# X = 30: num_rounds=5000:11min
# X = 30: num_rounds=10_000:21min
# X = 12: num_rounds=100:min - failled because of memory :: try reduce variable/ no other running
def full_pipeline(model_type="catboost", num_rounds=100, X=12, yvar="zdif", tcol="edem"):
    """
    Full pipeline for training a model with given parameters.
    """
    # Step 1: List files by tilenames
    print_context('# Step 1: List files by tilenames')
    fparquet_list, tile_files_list = list_files_by_tilenames(RES_DPATH, X, tilenames_lidar)
    print(f"Found {len(fparquet_list)} parquet files and {len(tile_files_list)} tile files.")

    # Step 2: Set up output directory
    out_dpath = f'output/wa_train_ctb/{X}'
    os.makedirs(out_dpath, exist_ok=True)
    os.chdir(out_dpath)
    print(f"Output directory created or already exists: {out_dpath}")

    # Step 3: Read parquet files into dataframe
    print("Reading parquet files into dataframe...")
    tfcols = ['edem', 'ldem', 'egm08', 'egm96', 'hem', 'vv', 'vh', 'red', 'green', 'blue', 'nir', 'swir1']
    df = pd.read_parquet(fparquet_list, columns=tfcols)
    print(f"Dataframe loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Step 4: Drop nulls in the target column
    print(f"Dropping nulls from column '{tcol}'...")
    df = dropnulls_bycol(df, col=tcol)
    print(f"After dropping nulls, dataframe has {df.shape[0]} rows and {df.shape[1]} columns.")

    # Step 5: Check for nulls
    print("Checking for any remaining null values...")
    df = check_fillnulls(df)
    print("Null check complete.")

    # Step 6: Calculate yvar (difference between tcol and 'ldem')
    print(f"Calculating {yvar} as the difference between '{tcol}' and 'ldem'...")
    df[yvar] = df[tcol].subtract(df['ldem'])
    print(f"First few values of {yvar}:\n", df[yvar].head())

    # Step 7: Split dataframe into training and validation sets
    print(f"Splitting dataframe into training and validation sets...")
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=43)
    print(f"Training set size: {train_df.shape[0]} rows, Validation set size: {valid_df.shape[0]} rows.")
    del df  # Free memory

    # Step 8: Prepare for model training
    fname = f'N{len(train_df) + len(valid_df)}_{yvar}_{X}m'
    print(f"Training model with dataset name: {fname}")

    # Step 9: Train the model
    print(f"Training {model_type} model...")
    train_model(train_data=train_df, 
                valid_data=valid_df, 
                target_col=yvar, 
                features_col=['egm08', 'egm96', 'hem', 'vv', 'vh', 'red', 'green', 'blue', 'nir', 'swir1'], 
                dataset_name=fname, 
                model_type=model_type, 
                num_rounds=num_rounds)
    print("Model training complete.")
    print(fname)
    
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

# Measure time for the full pipeline
measure_time_beautifully("Full Model Training Pipeline", full_pipeline)
