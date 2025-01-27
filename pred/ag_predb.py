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
from predutitls import ag_mbest_predict_workflow
from pred.predvars import outdir,ag_models_dpath


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
   # dirname = "tlimit120_good_quality"
    dirname = "tlimit1020_good_quality"
    modelpath = f"{ag_models_dpath}/{str(X)}/{yvar}/{dirname}"
        

    fparquet_list, tile_files_list = get_parquets_and_geotifs_by_tile(RES_DPATH, X, tilenames, vending_all)

    total_workflow_time = 0
    for fparquet, tile_files in zip(fparquet_list, tile_files_list):
        workflow_time = ag_mbest_predict_workflow(outdir, dirname, modelpath, 
                              fparquet, tile_files, 
                              fcol, yvar, tcol, ps, bsize)
        total_workflow_time += workflow_time
        print(f"Total workflow time: {total_workflow_time:.2f} seconds")
        print(f"Total workflow time: {total_workflow_time/60:.2f} mins")

    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    print(f"Total elapsed time: {elapsed_time/60:.2f} mins")

if __name__ == "__main__":
    main()


# we can run this in parallel 