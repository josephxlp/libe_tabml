import os 
import time
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from pprint import pprint
from uvars import MODEL_REPO_DPATH, RES_DPATH, tilenames_lidar
from ud_tilepartquets import dropnulls_bycol, check_fillnulls, list_files_by_tilenames
from autogluon.tabular import TabularDataset, TabularPredictor
from ud_tilepartquets import read_tiles_bysample, estimate_nsamples
import gc
import logging
import time 

# Set up logging configuration
logging.basicConfig(
    filename=os.path.join(MODEL_REPO_DPATH, 'workflow_log.txt'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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

# Function to set up time limit based on provided minutes and hours
def set_time_limit(minutes, hours):
    return 60 * minutes * hours

# Function to determine multipliers and target samples based on X
def get_multipliers_and_samples(X, num_tiles, multipliers=None):
    # If no multipliers are provided, use default ones based on X
    if multipliers is None:
        if X == 12:
            multipliers = [0.1, 0.2, 0.3, 0.5, 0.8, 1, 2, 3, 6]
            target_samples = 9000 * 9000
        elif X == 30:
            multipliers = [0.5, 1, 3, 4, 5, 6]
            target_samples = 3600 * 3600
        elif X == 90:
            multipliers = [0.01, 0.5, 1, 3, 4, 5, 6]
            target_samples = 1200 * 1200
        else:
            logging.error(f'Samples not available for {X}. Try 90, 30, or 12')
            return None, None
    else:
        # If multipliers are provided, use them and set a default target_samples
        target_samples = 9000 * 9000  # Default target sample size for a custom multiplier set

    # Estimate the number of samples based on the multipliers and number of tiles
    Nsamples = estimate_nsamples(target_samples, num_tiles, multipliers)
    assert len(Nsamples) == len(multipliers), "Multipliers do not match"
    
    return multipliers, Nsamples

# Function to read parquet files and create the dataframe
def read_parquet_data(fparquet_list, tcol, rcol, N):
    df = read_tiles_bysample(fparquet_list, tcol, rcol, N)
    logging.info(f"Dataframe loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Function to calculate the difference and add it as a new column
def calculate_difference(df, tcol, rcol, yvar):
    logging.info(f"Calculating {yvar} as the difference between '{tcol}' and '{rcol}'...")
    df[yvar] = df[tcol].subtract(df[rcol])
    return df

# Function to split the dataframe into training, validation, and test sets
def split_data(df, label):
    train_df, valid_df = train_test_split(df, test_size=0.3, random_state=43)
    valid_df, test_df = train_test_split(valid_df, test_size=0.5, random_state=43)

    logging.info(f"Training set size: {len(train_df)}")
    logging.info(f"Validation set size: {len(valid_df)}")
    logging.info(f"Test set size: {len(test_df)}")

    return train_df, valid_df, test_df

# Function to prepare datasets and train the model
def train_model(train_df, valid_df, test_df, label, labelfcol, time_limit, out_dpath, presets):
    train_data = TabularDataset(train_df[labelfcol])
    valid_data = TabularDataset(valid_df[labelfcol])
    test_data = TabularDataset(test_df[labelfcol])

    gc.collect()

    predictor = TabularPredictor(
        label=label,
        eval_metric='rmse',
        problem_type="regression",
        verbosity=3,
        path=out_dpath
    ).fit(
        train_data=train_data,
        tuning_data=test_data,
        use_bag_holdout=True,
        save_bag_folds=True,
        time_limit=time_limit,
        presets=presets
    )

    return predictor, valid_data

# Function to evaluate the model and save results
def evaluate_and_save(predictor, valid_data, label, out_dpath):
    y_pred = predictor.predict(valid_data.drop(columns=[label]))
    
    # Evaluate the model
    evaluation_results = predictor.evaluate(valid_data, silent=True)

    # Generate and save leaderboard
    leaderboard = predictor.leaderboard(valid_data)
    leaderboard_csv_path = os.path.join(out_dpath, f"{label}_leaderboard.csv")
    leaderboard.to_csv(leaderboard_csv_path, index=False)
    logging.info(f"Leaderboard saved to {leaderboard_csv_path}")

    # Save evaluation results
    evaluation_results_path = os.path.join(out_dpath, f"{label}_evaluation_results.txt")
    with open(evaluation_results_path, "w") as f:
        f.write("Evaluation Metrics:\n")
        for metric, value in evaluation_results.items():
            f.write(f"{metric}: {value}\n")
    logging.info(f"Evaluation results saved to {evaluation_results_path}")

# Main function to execute the entire workflow
def run_workflow(X, time_limit, outdir, presets, yvar, tcol, rcol, fcol, tilenames_lidar, N,m):
    logging.info("# Step 0: Instantiating the Workflow...")
    label = yvar
    labelfcol = [label] + fcol

    if m is not None:
        m = str(m).replace('.', 'p')
        mN = f'eq{m}xtile'

    logging.info('# Step 1: List files by tilenames')
    fparquet_list, tile_files_list = list_files_by_tilenames(RES_DPATH, X, tilenames_lidar)
    logging.info(f"Found {len(fparquet_list)} parquet files and {len(tile_files_list)} tile files.")
    pprint(fparquet_list)

    out_dpath = f'{outdir}/autogluon_study/{X}/{yvar}/{presets}/tlimit{str(time_limit)}_{str(N)}_{mN}'
    os.makedirs(out_dpath, exist_ok=True)
    logging.info(f"Output directory created or already exists: {out_dpath}")

    # Step 3: Read parquet files into dataframe
    df = read_parquet_data(fparquet_list, tcol, rcol, N)

    # Calculate the difference
    df = calculate_difference(df, tcol, rcol, yvar)

    # Step 7: Split dataframe into training, validation, and test sets
    train_df, valid_df, test_df = split_data(df, label)

    # Train the model
    predictor, valid_data = train_model(train_df, valid_df, test_df, label, labelfcol, time_limit, out_dpath, presets)

    # Evaluate and save results
    evaluate_and_save(predictor, valid_data, label, out_dpath)

presents_list = ["medium_quality","good_quality","high_quality","best_quality"]

start_time = time.perf_counter()
# Example usage:
X = 12 #90 
num_tiles = 6
minutes = 1#60#2
hours = 9# 4
time_limit_add = 900 # 15 mins 
outdir = MODEL_REPO_DPATH
presets = "medium_quality" #medium_quality:1@8h #good_quality:1@9h #medium_quality:#best_quality:
yvar = "zdif"
tcol = 'edem_w84'
rcol = 'multi_dtm_lidar'
fcol = ['egm08', 'egm96', 'tdem_hem', 'multi_s1_band1', 'multi_s1_band2',
        'multi_s2_band1', 'multi_s2_band2', 'multi_s2_band3']

task_description = f'Training Model Params:\nSaving @{outdir}'
task_description +=f'GRID:{X}\nTime:{hours}h{minutes}min\nMode:{presets}'
# Get the time limit
time_limit = set_time_limit(minutes, hours)

# Get multipliers and Nsamples outside the main workflow
# [0.1,0.5, 1, 3, 5, 6,10,15
multipliers, Nsamples = get_multipliers_and_samples(
    X, num_tiles,[1, 3, 6,8,10]) 

if Nsamples is None:
    exit()

## exclude some models ???
#### vovrt this into GPU, but first make sure loading big ones working just fine []???????
#N = Nsamples[0]
# 67500000
#add sample size tlimitandpresent not enough , add multiplies to the name
# run with alll dataset, no sample
print_context(task_description)
print(multipliers)
print(Nsamples)
#multipliers, Nsamples = [],[67500000]
#Logging the parameters for each attempt
for i, N in enumerate(Nsamples):
    m = multipliers[i]
    print(f'{multipliers[i]}xtile @{N} samples')
#     if i > 0:
#         break
    logging.info(f'{multipliers[i]}xtile @{N} samples')

    try: 
        run_workflow(X, time_limit, outdir, presets, 
              yvar, tcol, rcol, fcol, tilenames_lidar, N,m)
        logging.info("First attempt successful.")
    except Exception as e:
        logging.error(f"First attempt failed: {e}")
        try:
            time_limit = time_limit + time_limit_add # 15 minutes
            run_workflow(X, time_limit, outdir, presets, 
                         yvar, tcol, rcol, fcol, tilenames_lidar, N,m)
            logging.info("Second attempt successful.")
        except Exception as e:
            logging.error(f"Second attempt failed: {e}")
            logging.info("Max retries exceeded, skipping this configuration.")

# # The log will now contain all information about the workflow, including parameter values, success/failure flags, and any errors encountered.
# add one more layer of adding extra time 
# weekend do a free reing, but try gpu first 

end_time = time.perf_counter()
# Calculate elapsed time
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60
elapsed_hours = elapsed_minutes / 60
elapsed_days = elapsed_hours / 24
# Border symbols
task_description = task_description
border_char = "=" * 60
padding_char = " " * 4
print('#------------------------------------------------#')
print(border_char)
print(f"{padding_char}Task Performance Report")
print(border_char)
print_context(f"{padding_char}Task: {task_description}")
print_context(f"{padding_char}Elapsed Time:")
print(f"{padding_char * 2}{elapsed_seconds:.2f} seconds")
print(f"{padding_char * 2}{elapsed_minutes:.2f} minutes")
print(f"{padding_char * 2}{elapsed_hours:.2f} hours")
print(f"{padding_char * 2}{elapsed_days:.2f} days")
print(border_char)
# make this into a text file , in addition to pritnint it 
# send me a notification on desktop , and send email 
print(Nsamples)