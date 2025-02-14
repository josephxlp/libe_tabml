import os 
import time
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from pprint import pprint
from uvars import MODEL_REPO_DPATH, RES_DPATH,tilenames_lidar
from ud_tilepartquets import dropnulls_bycol,check_fillnulls,list_files_by_tilenames
from utilsml import train_model
from utils import print_context,read_tiles_bysample,



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
X= 30 #d:1000:1 5000:1 10000:1
num_rounds = 20_000 
#-------------------------------------------------------------#
model_type="catboost"
outdir = MODEL_REPO_DPATH
num_tiles=6
if X == 12:
    #multipliers=[0.1, 0.2, 0.3, 0.5, 0.8, 1, 2, 3,6]
    #multipliers=[0.1, 0.3, 0.5, 1, 2, 3,6]
    multipliers =[5,6,10]
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