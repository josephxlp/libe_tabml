import os
import sys
import time


from paths import libpath
sys.path.append(libpath)
from utilsdf import check_fillnulls 
from utilsdf import get_parquets_and_geotifs_by_tile
from uvars import (tilenames_mkd, tilenames_tls,tilenames_rgn,
                   tilenames_lidar,RES_DPATH)

from uvars import aux_ending12,s1_ending12,s2_ending12,tar_ending12
from predutitls import cb_predict_workflow

def main():
    start_time = time.time()

    yvar = "zdif"
    tcol = 'edem_w84'
    rcol = 'multi_dtm_lidar'
    fcol = ['egm08', 'egm96', 'tdem_hem', 'multi_s1_band1', 'multi_s1_band2',
            'multi_s2_band1', 'multi_s2_band2', 'multi_s2_band3']
    tar_ending, aux_ending, s1_ending, s2_ending = aux_ending12, s1_ending12, s2_ending12, tar_ending12
    vending_all = tar_ending + aux_ending + s1_ending + s2_ending
    X = 12
    ps = 9001
    tilenames = tilenames_mkd + tilenames_tls + tilenames_rgn # + tilenames_lidar
    bsize = 512
    #modelpath = 'path/to/model'
    #outdir = 'path/to/output'
    modelpath = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS/train_cb_bysample/12/zdif/iter10000_n81000000_eq6xtile/catboost_10000_42_model.txt"
    outdir = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/PREDICTIONS/TESTING/"

    fparquet_list, tile_files_list = get_parquets_and_geotifs_by_tile(RES_DPATH, X, tilenames, vending_all)

    total_workflow_time = 0
    for fparquet, tile_files in zip(fparquet_list, tile_files_list):
        workflow_time = cb_predict_workflow(outdir, modelpath, fparquet, tile_files, fcol, yvar, tcol, ps, bsize)
        total_workflow_time += workflow_time

    elapsed_time = time.time() - start_time

    log_file = os.path.join(outdir, 'workflow_log.txt')
    with open(log_file, 'w') as log:
        log.write(f"Total workflow time: {total_workflow_time:.2f} seconds\n")
        log.write(f"Total elapsed time: {elapsed_time:.2f} seconds\n")

    print(f"Total workflow time: {total_workflow_time:.2f} seconds")
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()