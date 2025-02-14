import os
import sys
from pprint import pprint
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from trainvars import libpath
sys.path.append(libpath)
import time 

from utilsdf import read_tiles_dosample, read_tiles_nosample, list_files_by_tilenames
from trainutils import train_model, print_context, measure_time_beautifully
from uvars import MODEL_REPO_DPATH, RES_DPATH,tilenames_lidar

def train_models_pipeline(config):
    """
    Main function to execute the model training pipeline.
    
    Args:
        config (dict): Configuration dictionary with required parameters.
    """
    # Step 1: Ensure output directory exists
    os.makedirs(config['outdir'], exist_ok=True)

    # Step 2: Prepare output directory paths and format strings
    mN = f"eq{str(config['m']).replace('.', 'p')}xtile" if config['m'] is not None else 'None'
    out_dpath_template = f"{config['outdir']}/{config['dirname']}/{config['X']}/{config['yvar']}/iter{config['num_rounds']}_n{{}}_{mN}_s{len(config['seedlist'])}"

    # Step 3: List files by tilenames
    print_context("# Step 1: List files by tilenames")
    fparquet_list, tile_files_list = list_files_by_tilenames(config['RES_DPATH'], config['X'], config['tilenames_lidar'])
    print(f"Found {len(fparquet_list)} parquet files and {len(tile_files_list)} tile files.")
    pprint(fparquet_list)

    # Step 4: Read data based on sampling or no sampling
    if config['N'] is not None:
        print_context("Running: read_tiles_dosample...")
        df = read_tiles_dosample(fparquet_list, config['tcol'], config['rcol'], config['N'])
        out_dpath = out_dpath_template.format(config['N'])
    else:
        print_context("Running: read_tiles_nosample...")
        df = read_tiles_nosample(fparquet_list, config['rcol'], config['tcol'])
        out_dpath = out_dpath_template.format(len(df))

    os.makedirs(out_dpath, exist_ok=True)
    print(f"Output directory created or already exists: {out_dpath}")

    # Step 5: Calculate target variable
    print(f"Calculating {config['yvar']} as the difference between '{config['tcol']}' and '{config['rcol']}'...")
    df[config['yvar']] = df[config['tcol']].subtract(df[config['rcol']])

    # Step 6: Split data into training and validation sets
    print("Splitting dataframe into training and validation sets...")
    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=43)
    del df  # Free memory

    # Step 7: Train the model with the initial seed
    metrics_df, best_model_iter = train_model(
        train_data=train_df,
        valid_data=valid_df,
        target_col=config['yvar'],
        features_col=config['fcol'],
        output_dir=out_dpath,
        model_type=config['model_type'],
        num_rounds=config['num_rounds'],
        seed=42,
        esr=None,
    )
    print(tabulate(metrics_df, headers="keys", tablefmt="psql"))

    # Step 8: Train models for each seed
    # add plus 10 to the best_model_iter
    for i, seed in enumerate(config['seedlist']):
        print_context(f"Training {config['model_type']} model @ {seed}...\n{out_dpath}")
        print_context(
            f"{i+1}/{len(config['seedlist'])} @{config['X']}m @seed:{seed} "
            f"@num_rounds:{config['num_rounds']} @dynamic_early_stopping_rounds:{best_model_iter}"
        )

        metrics_df, best_model_iter = train_model(
            train_data=train_df,
            valid_data=valid_df,
            target_col=config['yvar'],
            features_col=config['fcol'],
            output_dir=out_dpath,
            model_type=config['model_type'],
            num_rounds=config['num_rounds'],
            seed=seed,
            esr=best_model_iter,
        )
        print(tabulate(metrics_df, headers="keys", tablefmt="psql"))

    print(f"All model training complete. Total of {len(config['seedlist'])} models.")

# Example usage:
config = {
    "RES_DPATH": RES_DPATH,
    "tilenames_lidar": tilenames_lidar,
    "outdir": MODEL_REPO_DPATH,
    "dirname": "dev",
    "X": 12,
    "N": None,
    "m": None,
    "yvar": "zdif",
    "tcol": "edem_w84",
    "rcol": "multi_dtm_lidar",
    "fcol": [
        "egm08", "egm96", "tdem_hem",
        "multi_s1_band1", "multi_s1_band2",
        "multi_s2_band1", "multi_s2_band2", "multi_s2_band3"
    ],
    "model_type": "catboost",
    "num_rounds": 30_000,
    "seedlist": [21, 43],
}
 
if __name__ == "__main__":
    measure_time_beautifully(
        "Full Model Training Pipeline",
        train_models_pipeline,
        config)
#train_models_pipeline(config)

