from uvars import tilenames_lidar,RES_DPATH,vending_all,nending_all,ftnames
from utilsdf import tile_files_to_parquet_parallel

X=90
tilenames = tilenames_lidar

if __name__ == '__main__':
    tile_files_to_parquet_parallel(tilenames, RES_DPATH, X, vending_all, nending_all, ftnames)