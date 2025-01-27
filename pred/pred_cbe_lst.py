import os
import sys
from paths import libpath
sys.path.append(libpath)
from utilsdf import get_parquets_and_geotifs_by_tile
from uvars import (tilenames_mkd, tilenames_tls,tilenames_rgn,RES_DPATH)
from uvars import aux_ending12,s1_ending12,s2_ending12,tar_ending12
from predutitls import cbe_predict_workflow#parallel_cbe_prediction_workflow
import time


ti = time.perf_counter()
yvar = "zdif"
tcol = 'edem_w84'
rcol = 'multi_dtm_lidar'
fcol = ['egm08', 'egm96', 'tdem_hem', 'multi_s1_band1', 'multi_s1_band2',
        'multi_s2_band1', 'multi_s2_band2', 'multi_s2_band3']
tar_ending,aux_ending,s1_ending,s2_ending = aux_ending12,s1_ending12,s2_ending12,tar_ending12
vending_all = tar_ending+aux_ending+s1_ending+s2_ending
X = 12 
ps = 9001
tilenames = tilenames_mkd
bsize = 256 # match with grid size X 

outdir = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/PREDICTIONS/TESTING"
dirname = "iter15000_n236435487_eqallxtile_s3"
TRAIN_MODELS_DIR ="/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS"
model_dir = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS/cb_trainbye/12/zdif/iter15000_n236435487_eqallxtile_s3/"
#f"{TRAIN_MODELS_DIR}/train_cb_bysample/12/zdif/{dirname}/"
if __name__ == "__main__":
    fparquet_list, tile_files_list = get_parquets_and_geotifs_by_tile(RES_DPATH, X, tilenames, vending_all)
    for fparquet,tile_files in zip(fparquet_list,tile_files_list):
        print(fparquet)
        


    tf = time.perf_counter() - ti 
    print(f'run.time ={tf/60} min')







# yvar = "zdif"
# tcol = 'edem_w84'
# rcol = 'multi_dtm_lidar'
# fcol = ['egm08', 'egm96', 'tdem_hem', 'multi_s1_band1', 'multi_s1_band2',
#         'multi_s2_band1', 'multi_s2_band2', 'multi_s2_band3']

# model_list_e = [
#     "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS/train_cb_bysample/12/zdif/iter20000_n81000000_eq6xtile/catboost_20000_42_model.txt",
#     "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS/train_cb_bysample/12/zdif/iter10000_n81000000_eq6xtile/catboost_10000_42_model.txt",
#     "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS/train_cb_bysample/12/zdif/iter5000_n236435487_eqallxtile/catboost_5000_13_model.txt",
#     "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS/train_cb_bysample/12/zdif/iter2000_n236435487_eqallxtile/catboost_2000_21_model.txt",
#     "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS/train_cb_bysample/12/zdif/iter5000_n236435487_eqallxtile/catboost_5000_43_model.txt",
#     "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS/train_cb_bysample/12/zdif/iter10000_n236435487_eqallxtile/catboost_10000_43_model.txt"
# ]
# dirname_model_list_e =  "model_list_e"

# outdir = '/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/PREDICTIONS/TESTING'
# bsize = 512
# ps = 9001
# X = 12
# tilenames = tilenames_mkd+ tilenames_tls+tilenames_rgn,
# tar_ending,aux_ending,s1_ending,s2_ending = aux_ending12,s1_ending12,s2_ending12,tar_ending12
# vending_all = tar_ending+aux_ending+s1_ending+s2_ending








# pass all the parameters in sing dictionmary 
# fparquet_list, tile_files_list = get_parquets_and_geotifs_by_tile(RES_DPATH, X, tilenames, vending_all)
# if __name__  == "__main__":
#     max_workers = int(os.cpu_count() * 0.5)
#     parallel_cbe_prediction_workflow(outdir, 
#                                      model_list=model_list_e, 
#                                      dirname=dirname_model_list_e, 
#                                      fparquet_list=fparquet_list, 
#                                      tile_files_list=tile_files_list, 
#                                      fcol=fcol, 
#                                      yvar=yvar, 
#                                      tcol=tcol, 
#                                      ps=ps,  
#                                      bsize=bsize, 
#                                      max_workers=max_workers)

