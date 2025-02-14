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

if __name__ == "__main__":
    fparquet_list, tile_files_list = get_parquets_and_geotifs_by_tile(RES_DPATH, X, tilenames, vending_all)
    for fparquet,tile_files in zip(fparquet_list,tile_files_list):
        print(fparquet)
        


    tf = time.perf_counter() - ti 
    print(f'run.time ={tf/60} min')

