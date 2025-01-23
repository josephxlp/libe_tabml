import os 
import time
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from pprint import pprint
from uvars import MODEL_REPO_DPATH, RES_DPATH,tilenames_lidar
from ud_tilepartquets import dropnulls_bycol,check_fillnulls,list_files_by_tilenames
from utilsml import train_model

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


def encode_nulls_bycol(df, col, lval, hval):
    """
    Replace invalid or out-of-range values in a specified column of a DataFrame with np.nan.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        col (str): The name of the column to process.
        lval (float): The lower bound of acceptable values.
        hval (float): The upper bound of acceptable values.

    Returns:
        pandas.DataFrame: A modified DataFrame with invalid values replaced by np.nan.
    """
    # Replace -9999 and -9999.0 with np.nan
    df[col] = df[col].replace([-9999, -9999.0], np.nan)
    
    # Replace values less than lval or greater than hval with np.nan
    df.loc[(df[col] < lval) | (df[col] > hval), col] = np.nan
    
    return df

def read_tiles_bysample(fparquet_list, tcol, rcol, N):
    """
    Processes a list of parquet files by encoding nulls, dropping rows with nulls,
    sampling rows, and concatenating the results.

    Parameters:
        fparquet_list (list of str): List of file paths to parquet files.
        tcol (str): Target column to process for null encoding and dropping.
        rcol (str): Reference column to process for null encoding and dropping.
        N (int): Number of rows to sample from each file.

    Returns:
        pd.DataFrame: A concatenated DataFrame of processed samples.
    """
    dflist = []
    for fparquet in fparquet_list:
        try:
            # Read parquet file
            df = pd.read_parquet(fparquet)
            print(f"Processing file: {os.path.basename(fparquet)}")
            
            # Encode nulls
            print(f"Encoding nulls from column {tcol}...")
            encode_nulls_bycol(df, col=tcol, lval=-40, hval=1000)
            print(f"Encoding nulls from column {rcol}...")
            encode_nulls_bycol(df, col=rcol, lval=-40, hval=1000)
            
            # Drop nulls
            print(f"Dropping nulls from column '{tcol}'...")
            df = dropnulls_bycol(df, col=tcol)
            print(f"Dropping nulls from column '{rcol}'...")
            df = dropnulls_bycol(df, col=rcol)
            
            # Sampling
            print(f"Sampling from {os.path.basename(fparquet)}...")
            L = len(df)
            if L < N:
                print(f"Warning: Requested sample size {N} exceeds available rows {L}. Sampling all rows instead.")
                #df = df.sample(L)
                df = df.sample(frac=1)
            else:
                df = df.sample(N)
            
            # Append to list
            if not df.empty:
                dflist.append(df)
            else:
                print(f"Warning: DataFrame from {os.path.basename(fparquet)} is empty after filtering.")
        
        except Exception as e:
            print(f"Error processing file {fparquet}: {e}")

    # Concatenate
    if dflist:
        df = pd.concat(dflist, ignore_index=True)
        df = check_fillnulls(df)
        print("Final DataFrame columns:", df.columns)
        return df
    else:
        print("Warning: No valid DataFrames to concatenate. Check input files or filtering logic.")
        return pd.DataFrame()  # Return an empty DataFrame if no valid data was processed


def full_pipeline(outdir,model_type="catboost", num_rounds=100, X=12, N=1000,m=None):
    os.makedirs(outdir, exist_ok=True)
    yvar="zdif"
    tcol='edem_w84' #"edem"
    rcol='multi_dtm_lidar'
    fcol = ['egm08', 'egm96', 'tdem_hem', 
            'multi_s1_band1', 'multi_s1_band2',
            'multi_s2_band1', 'multi_s2_band2', 'multi_s2_band3']
    # pass all this as dictionary 
    if m is not None:
        m = str(m).replace('.', 'p')
        mN = f'eq{m}xtile'

    """
    Full pipeline for training a model with given parameters.
    """
    #out_dpath = f'{outdir}/train_cb_bysample/{X}/{yvar}/{mN}_nsample{N}_num_rounds{num_rounds}'
    out_dpath = f'{outdir}/train_cb_bysample/{X}/{yvar}/iter{num_rounds}_n{N}_{mN}'
    os.makedirs(out_dpath, exist_ok=True)
    print(f"Output directory created or already exists: {out_dpath}")


    print_context('# Step 1: List files by tilenames')
    fparquet_list, tile_files_list = list_files_by_tilenames(RES_DPATH, X, tilenames_lidar)
    print(f"Found {len(fparquet_list)} parquet files and {len(tile_files_list)} tile files.")
    pprint(fparquet_list)
    df = read_tiles_bysample(fparquet_list, tcol, rcol, N)
    # if X == 12
    # dflist  = []
    # for fparquet in fparquet_list:
    #     df = pd.read_parquet(fparquet)#, columns=tfcols)
    #     print(f"Encoding nulls from column {tcol}...")
    #     encode_nulls_bycol(df, col=tcol, lval=-30, hval=1000)
    #     print(f"Encoding nulls from column {rcol}...")
    #     encode_nulls_bycol(df, col=rcol, lval=-30, hval=1000)
    #     #df = encode_nulls(df)

    #     print(f"Dropping nulls from column '{tcol}'...")
    #     df = dropnulls_bycol(df, col=tcol)

    #     print(f"Dropping nulls from column '{rcol}'...")
    #     df = dropnulls_bycol(df, col=rcol)
    #     print(f'Sampling from {os.path.basename(fparquet)}')
    #     L = len(df)
    #     if L < N:
    #         print(f"Warning: Requested sample size {N} exceeds available rows {L}. Sampling all rows instead.")
    #         df = df.sample(L)
    #     else:
    #         df = df.sample(N)
    #         dflist.append(df)
    # df = pd.concat(dflist, ignore_index=True)

    # df = check_fillnulls(df)
    # print(df.columns)
    
    print(f"Calculating {yvar} as the difference between '{tcol}' and 'ldem'...")
    df[yvar] = df[tcol].subtract(df[rcol])
    #print(f"First few values of {yvar}:\n", df[yvar].head())

    # Step 7: Split dataframe into training and validation sets
    print("Splitting dataframe into training and validation sets...")
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=43)
    print(f"Training set size: {train_df.shape[0]} rows, Validation set size: {valid_df.shape[0]} rows.")
    del df  # Free memory

    # Step 9: Train the model
    print(f"Training {model_type} model...")
    train_model(train_data=train_df, 
                valid_data=valid_df, 
                target_col=yvar, 
                features_col=fcol, 
                output_dir=out_dpath, # becomes outdir 
                model_type=model_type, 
                num_rounds=num_rounds)
    print("Model training complete.")
    #print(fname)

def estimate_nsamples(target_samples=81_000_000, num_tiles=6, multipliers=[0.2, 0.3, 0.5, 0.8, 1, 2, 3]):
    samples_per_tile = target_samples // num_tiles
    nsamples = [int(samples_per_tile * multiplier) for multiplier in multipliers]
    return nsamples

##==============================================================================##
ftcol = ['egm08',
 'egm96',
 'tdem_hem',
 'multi_s1_band1',
 'multi_s1_band2',
 'multi_s2_band1',
 'multi_s2_band2',
 'multi_s2_band3',
 'edem_w84',
 'tdem_dem__fw',
 'multi_dtm_lidar']


yvar="zdif"
tcol='edem_w84' #"edem"
rcol='multi_dtm_lidar'
fcol = ['egm08', 'egm96', 'tdem_hem', 
        'multi_s1_band1', 'multi_s1_band2',
        'multi_s2_band1', 'multi_s2_band2', 'multi_s2_band3']##, 'edem_w84']


#N = 10_000
##X= 90 #d:1000:1 5000:1 10000:1#20000:
#X = 30 #d:1000:1 5000:1 10000:1
X= 12 #d:1000:1 5000:1 10000:1
num_rounds = 20_000 
#-------------------------------------------------------------#
model_type="catboost"
outdir = MODEL_REPO_DPATH
num_tiles=6
if X == 12:
    #multipliers=[0.1, 0.2, 0.3, 0.5, 0.8, 1, 2, 3,6]
    multipliers=[0.1, 0.3, 0.5, 1, 2, 3,6]
    target_samples=9000*9000 #9001
    
    Nsamples = estimate_nsamples(target_samples, num_tiles, multipliers)
    print(Nsamples)
    assert len(Nsamples) == len(multipliers), "Multipliers do not much"
elif X == 30:
    multipliers= [0.5,1,3,4,5,6]
    target_samples=3600*3600
    Nsamples = estimate_nsamples(target_samples, num_tiles, multipliers)
    print(Nsamples)
    assert len(Nsamples) == len(multipliers), "Multipliers do not much"

elif X == 90:
    multipliers= [0.5,1,3,4,5,6]
    target_samples=1200*1200
    Nsamples = estimate_nsamples(target_samples, num_tiles, multipliers)
    print(Nsamples)
    assert len(Nsamples) == len(multipliers), "Multipliers do not much"

else:
    print(f'Samples not available for {X}.\n Try 90,30 or 12')

if __name__ == '__main__':
    for i, N in enumerate(Nsamples):
        #print(N)
        m = multipliers[i]
        print_context(f'Processing {N} samples\nEquivalent to {m}x TILE')
        #[] add the other wrapper here 

        measure_time_beautifully("Full Model Training Pipeline",
                                 full_pipeline,
                                 outdir=outdir,
                                 model_type=model_type, 
                                 num_rounds=num_rounds, 
                                 X=X, 
                                 N=N,
                                m=m)
        # full_pipeline(outdir=outdir,
        #                   model_type=model_type, 
        #                   num_rounds=num_rounds, 
        #                   X=X, 
        #                   N=N,
        #                   m=m)

        # try:
        #     full_pipeline(outdir=outdir,
        #                   model_type=model_type, 
        #                   num_rounds=num_rounds, 
        #                   X=X, 
        #                   N=N,
        #                   m=m)
        # except ValueError:
        #     print_context(f'The code failled due to Nsize \n{N} samples is too bigger')
        #     print(N)
        #     # add a file within the folder when it fails 