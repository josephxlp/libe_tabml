
from uvars import tilenames_lidar,RES_DPATH,vending_all,nending_all,ftnames
from utilsdf import tile_files_to_parquet_parallel
from utilsdf import assign_nulls, fillna,dropnulls_bycol,check_fillnulls
from uvars import s1_fnames, s2_fnames,aux_names
from utilsml import train_and_compare
from utilsviz import plot_rmse_r2

target_col = tcol = 'edem'
features_col = aux_names + s1_fnames +s2_fnames
tilenames = tilenames_lidar
Xlist = [90]#[30,90,500,1000]
X=90
csv_path = 'output/outlier_effect.csv'
png_path = 'output/outlier_effect.png'

dflist, fparquet_list = tile_files_to_parquet_parallel(tilenames, RES_DPATH, X, vending_all, nending_all, ftnames)
# read the data instead of loading  that  function it to process the data 
df = dflist[0]

df = assign_nulls(df)
df = fillna(df, s1_fnames)
df = fillna(df, s2_fnames)
df =  dropnulls_bycol(df, col=tcol) # use this only latter, fill hem always
#df = check_fillnulls(df) # these two could be used later depending on the  tcol


results_df = train_and_compare(df, target_col=target_col, 
                               features_col=features_col, 
                               model_type="catboost",
                               num_rounds=1000)

plot_rmse_r2(results_df, png_path)