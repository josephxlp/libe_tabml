import os
from glob import glob
import pandas as pd
import numpy as np 
import rasterio
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import zscore

def get_tile_names(tile_files, tilename, X):
    """
    Extracts and cleans tile names from file paths.
    """
    tile_names = [os.path.basename(f).replace('.tif', '') for f in tile_files]
    tile_names = [name.replace(f'{tilename}_', '').lower() for name in tile_names]
    #tile_names = [name.replace(f'_{X}', '').replace(f'{tilename}_', '').lower() for name in tile_names]
    return tile_names

def get_tile_files(DPATH, X, tilename):
    """
    Retrieves tile files and the corresponding parquet file path.
    """
    TILESX = f"{DPATH}{X}"
    tile_dpath = f'{TILESX}/{tilename}'
    fparquet = f'{tile_dpath}/{tilename}_byldem.parquet'
    tile_files = glob(f'{tile_dpath}/*.tif')
    return tile_files, fparquet

def list_files_by_tilenames(RES_DPATH, X, tilenames):
    """
    Lists parquet and tile files for multiple tiles.
    """
    fparquet_list, tile_files_list = [], []
    for tilename in tilenames:
        tile_files, fparquet = get_tile_files(RES_DPATH, X, tilename)
        fparquet_list.append(fparquet)
        tile_files_list.append(tile_files)
    return fparquet_list, tile_files_list

def filter_files_by_endingwith(files, var_ending):
    """
    Filters raster files by specified endings.
    """
    filtered_files = [f for ending in var_ending for f in files if f.endswith(ending)]
    print(f"Filtered files count: {len(filtered_files)}/{len(files)}")
    return filtered_files

def read_raster(file_path, file_name):
    
    df = pd.DataFrame()
    with rasterio.open(file_path) as src:
        count = src.count
        #print(f"Number of bands: {count}")
        
        if count == 1:
            df[f'{file_name}'] = src.read(1,masked=True).flatten()
        elif count > 1:
            for i in range(1, count + 1):  # Use 1-based indexing for raster bands
                df[f'{file_name}_band{i}'] = src.read(i,masked=True).flatten()
    
    return df

def pathlist2df(tile_files, tile_names):

    dflist = []
    for filepath,filename in zip(tile_files,tile_names):
        df = read_raster(filepath,filename)
        dflist.append(df)
    da = pd.concat(dflist,axis=1)
    return da 

def tile_files_to_parquet(DPATH, X, tilename, vending_all):
    """
    Processes raster files for a tile and saves them to a parquet file.
    """
    tile_files, fparquet = get_tile_files(DPATH, X, tilename)
    tile_files = filter_files_by_endingwith(tile_files, vending_all)

    if not tile_files:
        raise ValueError(f"No matching raster files found for {tilename}")

    tile_names = get_tile_names(tile_files, tilename, X)

    if os.path.isfile(fparquet):
        print(f"Parquet file already exists: {fparquet}")
        return None, fparquet

    df = pathlist2df(tile_files, tile_names)

    if X == 90:
        assert len(df) == 1200 * 1200, 'Grid shape and df length does not match'
    elif X == 30:
        assert len(df) == 3600 * 3600, 'Grid shape and df length does not match'
    elif X == 12:
        assert len(df) == 9001 * 9001, 'Grid shape and df length does not match'

    df.to_parquet(fparquet)
    print(f"Parquet file created: {fparquet}")
    return df, fparquet


def process_tile_files_to_parquet(tilename, RES_DPATH, X, vending_all):#, nending_all, ftnames):
    """
    Processes a single tile: Reads raster files, converts them to a DataFrame, and saves them to a parquet file.
    """
    df, fparquet = tile_files_to_parquet(RES_DPATH, X, tilename, vending_all)#, nending_all, ftnames)
    return df, fparquet

def tile_files_to_parquet_parallel(tilenames, RES_DPATH, X, vending_all):#, nending_all, ftnames):
    """
    Processes multiple tiles in parallel using ThreadPoolExecutor.

    Parameters:
        tilenames (list): List of tile names to process.
        RES_DPATH (str): Base directory path.
        X (str): Subdirectory name.
        vending_all (list): List of file endings to filter input raster files.
        nending_all (list): Names corresponding to the filtered raster files.
        ftnames (list): Column names for the DataFrame.

    Returns:
        list: List of DataFrames.
        list: List of parquet file paths.
    """
    dflist = []
    fparquet_list = []

    with ThreadPoolExecutor() as executor:
        # Submit tasks for parallel execution
        futures = [
            executor.submit(process_tile_files_to_parquet, tilename, RES_DPATH, X, vending_all)#, nending_all, ftnames)
            for tilename in tilenames
        ]

        # Collect results as they complete
        for future in futures:
            try:
                df, fparquet = future.result()
                dflist.append(df)
                fparquet_list.append(fparquet)
            except Exception as e:
                print(f"Error processing tile: {e}")

    return dflist, fparquet_list



####################################################################
#####################################################################
#####################################################################

def fillna(df, list_of_variables):
    """
    Fill missing values (`NaN`) in the specified subset of columns by the nearest values' average.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        list_of_variables (list): List of column names to apply the filling.

    Returns:
        pd.DataFrame: DataFrame with filled values for the specified columns.
    """
    for col in list_of_variables:
        if col in df.columns:
            # Get indices of missing values
            nan_indices = df[df[col].isna()].index
            
            for idx in nan_indices:
                # Identify valid neighbouring values (non-NaN)
                valid_values = []
                if idx - 1 >= 0:  # Check previous row
                    valid_values.append(df[col].iloc[idx - 1])
                if idx + 1 < len(df):  # Check next row
                    valid_values.append(df[col].iloc[idx + 1])
                valid_values = [v for v in valid_values if not pd.isna(v)]  # Filter non-NaN

                # Replace NaN with the average of valid neighbouring values
                if valid_values:
                    df.at[idx, col] = np.mean(valid_values)
    return df




def remove_outlier(df, tcol, approach='zscore', threshold=3, lower_percentile=0.05, upper_percentile=0.95):
    """
    Remove outliers from the DataFrame based on the specified column and approach.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        tcol (str): The target column to identify outliers.
        approach (str): The approach to detect outliers ('zscore', 'iqr', 'percentile').
        threshold (float): The Z-score threshold for the 'zscore' approach.
        lower_percentile (float): Lower percentile for 'percentile' approach (0 to 1).
        upper_percentile (float): Upper percentile for 'percentile' approach (0 to 1).

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if tcol not in df.columns:
        raise ValueError(f"Column '{tcol}' does not exist in the DataFrame.")

    if approach == 'zscore':
        # Compute Z-scores and filter based on the threshold
        df = df[np.abs(zscore(df[tcol], nan_policy='omit')) <= threshold]

    elif approach == 'iqr':
        # Compute IQR and filter outliers
        Q1 = df[tcol].quantile(0.25)
        Q3 = df[tcol].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[tcol] >= lower_bound) & (df[tcol] <= upper_bound)]

    elif approach == 'percentile':
        # Filter values outside the specified percentile range
        lower_bound = df[tcol].quantile(lower_percentile)
        upper_bound = df[tcol].quantile(upper_percentile)
        df = df[(df[tcol] >= lower_bound) & (df[tcol] <= upper_bound)]

    else:
        raise ValueError(f"Unknown approach '{approach}'. Use 'zscore', 'iqr', or 'percentile'.")

    return df

def assign_nulls(df, lval=-50, hval=10000):
    """
    Assign `np.nan` to rows in the DataFrame based on specific conditions:
    - Replace `-inf` with `np.nan`.
    - Replace values less than `lval` with `np.nan`.
    - Replace values greater than `hval` with `np.nan`.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        lval (float): Lower threshold; values less than this are set to `np.nan`.
        hval (float): Upper threshold; values greater than this are set to `np.nan`.

    Returns:
        pd.DataFrame: DataFrame with updated values.
    """
    # Replace -inf with np.nan
    df.replace(-np.inf, np.nan, inplace=True)

    # Replace values less than lval with np.nan
    df[df < lval] = np.nan

    # Replace values greater than hval with np.nan
    df[df > hval] = np.nan

    return df

def dropnulls_bycol(df, col):
    """
    Drops rows from the DataFrame where the specified column contains null values (`NaN`).

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        col (str): The column to check for null values.

    Returns:
        pd.DataFrame: DataFrame with rows removed where the specified column contains `NaN`.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    # Drop rows where the specified column contains NaN
    df_cleaned = df.dropna(subset=[col])
    
    return df_cleaned

def check_fillnulls(df):
    """
    Checks if there are any null values across all columns in the DataFrame. 
    If found, fills them with:
    - Mean for continuous (numeric) columns.
    - Mode for categorical (non-numeric) columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to check and fill nulls.

    Returns:
        pd.DataFrame: DataFrame with null values filled.
    """
    for col in df.columns:
        if df[col].isnull().any():  # Check if the column has null values
            #print(f"Null values detected in column: {col}")
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Fill with mean for numeric columns
                mean_value = df[col].mean()
                #print(f"Filling with mean: {mean_value}")
                df[col] = df[col].fillna(mean_value)
            else:
                # Fill with mode for non-numeric (categorical) columns
                mode_value = df[col].mode()[0]
                #print(f"Filling with mode: {mode_value}")
                df[col] = df[col].fillna(mode_value)
        else:
            #print(f"No null values in column: {col}")
            continue
    
    return df
#============================================================

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


def estimate_nsamples(target_samples=81_000_000, num_tiles=6, multipliers=[0.2, 0.3, 0.5, 0.8, 1, 2, 3]):
    samples_per_tile = target_samples // num_tiles
    nsamples = [int(samples_per_tile * multiplier) for multiplier in multipliers]
    return nsamples

