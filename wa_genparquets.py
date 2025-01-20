from uvars import tilenames_lidar,RES_DPATH,nending_all,ftnames
from uvars import aux_ending30,s1_ending30,s2_ending30,tar_ending30
from uvars import aux_ending90,s1_ending90,s2_ending90,tar_ending90
from uvars import aux_ending12,s1_ending12,s2_ending12,tar_ending12
from utilsdf import tile_files_to_parquet_parallel

X=90
tilenames = tilenames_lidar
vending_all = tar_ending+aux_ending+s1_ending+s2_ending

if __name__ == '__main__':
    ti = time.perf_counter()

    tile_files_to_parquet_parallel(tilenames, RES_DPATH, X, vending_all, nending_all, ftnames)
    tf = time.perf_counter() - ti 
    print(f'{tf/60} min(s)')