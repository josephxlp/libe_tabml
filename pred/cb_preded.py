import os
import sys
from paths import libpath
sys.path.append(libpath)
from utilsdf import get_parquets_and_geotifs_by_tile
from uvars import (tilenames_mkd, tilenames_tls,tilenames_rgn,RES_DPATH)
from uvars import aux_ending12,s1_ending12,s2_ending12,tar_ending12
from predutitls import cbe_predict_workflow#parallel_cbe_prediction_workflow
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from predvars import TRAIN_MODELS_DIR,outdir

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
tilenames = tilenames_mkd+tilenames_tls+tilenames_rgn
bsize = 256 # match with grid size X 

#---------------------------------------------------------------------------------------------#
# dirname = "iter10000_n236435487_eqallxtile_s3"#"iter15000_n236435487_eqallxtile_s3"
# model_dir = f"{TRAIN_MODELS_DIR}/cb_trainbye/12/zdif/{dirname}/"

model_dir = "/media/ljp238/12TBWolf/RSPROX/OUTPUT_TILES/MODELS/cb_trainbye/12/zdif/iter15000_n236435487_eqallxtile_s3"
dirname = str(model_dir).split('/')[-1]
dirname
#---------------------------------------------------------------------------------------------#

#f"{TRAIN_MODELS_DIR}/train_cb_bysample/12/zdif/{dirname}/"
# if __name__ == "__main__":
#     fparquet_list, tile_files_list = get_parquets_and_geotifs_by_tile(RES_DPATH, X, tilenames, vending_all)
#     for fparquet,tile_files in zip(fparquet_list,tile_files_list):
#         print(fparquet)
#         #cbe_predict_workflow(outdir,model_dir,dirname,fparquet,tile_files,fcol,yvar,tcol,ps,bsize)
#     tf = time.perf_counter() - ti 
#     print(f'run.time ={tf/60} min')



def process_tile(fparquet, tile_files, outdir, model_dir, dirname, fcol, yvar, tcol, ps, bsize):
    """Wrapper for the cbe_predict_workflow function to handle a single task."""
    print(f"Processing {fparquet}")
    return cbe_predict_workflow(outdir, model_dir, dirname, fparquet, tile_files, fcol, yvar, tcol, ps, bsize)

if __name__ == "__main__":
    ti = time.perf_counter()
    # Retrieve lists of parquet and GeoTIFF files
    fparquet_list, tile_files_list = get_parquets_and_geotifs_by_tile(RES_DPATH, X, tilenames, vending_all)
    
    tasks = []
    with ProcessPoolExecutor() as ppe:
        for fparquet, tile_files in zip(fparquet_list, tile_files_list):
            # Submit tasks to the executor
            tasks.append(
                ppe.submit(process_tile, fparquet, tile_files, outdir, model_dir, dirname, fcol, yvar, tcol, ps, bsize)
            )
        
        # Monitor the completion of tasks
        for future in as_completed(tasks):
            try:
                result = future.result()
                print(f"Task completed successfully: {result}")
            except Exception as e:
                print(f"Task failed with exception: {e}")

    tf = time.perf_counter() - ti 
    print(f'run.time ={tf/60} min')
    print(f'dirname {dirname}\n fullpath:{model_dir}')





