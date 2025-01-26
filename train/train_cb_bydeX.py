
import os 
import sys
import time
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from pprint import pprint
from tabulate import tabulate

from trainvars import libpath
sys.path.append(libpath)

from uvars import MODEL_REPO_DPATH, RES_DPATH,tilenames_lidar
#from ud_tilepartquets import dropnulls_bycol,check_fillnulls,list_files_by_tilenames
from utilsdf import read_tiles_dosample, read_tiles_nosample,list_files_by_tilenames
from trainutils import train_model,print_context


#-------------------------------------------------------------------------------#

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


model_type="catboost"
outdir = MODEL_REPO_DPATH
model_type="catboost"
num_rounds=100
X=12
N=None,
m=None,
seedlist=[21,43,13]
num_rounds_list, seedlist = [100], [100]
N = 1000
dirname = "dev"
#-------------------------------------------------------------------------------#
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

print_context('# Step 1: List files by tilenames')
fparquet_list, tile_files_list = list_files_by_tilenames(RES_DPATH, X, tilenames_lidar)
print(f"Found {len(fparquet_list)} parquet files and {len(tile_files_list)} tile files.")
pprint(fparquet_list)



if N is not None:
    print_context('running:: read_tiles_dosample.....')
    df = read_tiles_dosample(fparquet_list, tcol, rcol, N)
    out_dpath = f'{outdir}/{dirname}/{X}/{yvar}/iter{num_rounds}_n{N}_{mN}_s{len(seedlist)}'
    print(f"Output directory created or already exists: {out_dpath}")
else:
    print_context('running:: read_tiles_nosample ....')
    df = read_tiles_nosample(fparquet_list,rcol,tcol)
    L = len(df)
    out_dpath = f'{outdir}/{dirname}/{X}/{yvar}/iter{num_rounds}_n{L}_{mN}_s{len(seedlist)}'

os.makedirs(out_dpath, exist_ok=True)
print(f"Output directory created or already exists: {out_dpath}")
    
print(f"Calculating {yvar} as the difference between '{tcol}' and 'ldem'...")
df[yvar] = df[tcol].subtract(df[rcol])
#print(f"First few values of {yvar}:\n", df[yvar].head())

# Step 7: Split dataframe into training and validation sets
print("Splitting dataframe into training and validation sets...")
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=43)

del df  # Free memory

metrics_df,best_model_iter = train_model(train_data=train_df, 
                                         valid_data=valid_df, 
                                         target_col=yvar, 
                                         features_col=fcol, 
                                         output_dir=out_dpath, 
                                        model_type=model_type, 
                                        num_rounds=num_rounds, 
                                        seed=42,
                                        esr=None)
print(tabulate(metrics_df,headers='keys',tablefmt='psql'))

for i,seed in enumerate(seedlist):
        texta = f"Training {model_type} model @ {seed}...\n{out_dpath}"
        print_context(texta)
        textb = f'{i+1}/{len(seedlist)} @{X}m @seed:{seed}  @num_rounds:{num_rounds} @:dynamic_early_stopping_rounds{best_model_iter}'
        print_context(textb)


        metrics_df,best_model_iter = train_model(train_data=train_df, 
                                         valid_data=valid_df, 
                                         target_col=yvar, 
                                         features_col=fcol, 
                                         output_dir=out_dpath, 
                                        model_type=model_type, 
                                        num_rounds=num_rounds, 
                                        seed=42,
                                        esr=best_model_iter)
        print(tabulate(metrics_df,headers='keys',tablefmt='psql'))

print(f"All Model training complete.\nTotal of {len(seedlist)} models")
