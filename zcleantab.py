import os 
from uvars import parq_outdir, ldar_mkd, ldar_rgn, ldar_tls
from utilsdf import read_tiles_nosample
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def save_parquet_chunk(df_chunk, file_name):
    """Função auxiliar para salvar um chunk em Parquet"""
    df_chunk.to_parquet(file_name, index=False)
    print(f"Saved {file_name} with {len(df_chunk)} rows.")

def save_chunks_as_parquet_par(df, chunk_size=100_000, output_prefix="chunk", num_workers=8):
    """Divide o DataFrame em chunks e salva os arquivos em paralelo"""
    num_chunks = (len(df) // chunk_size) + 1
    tasks = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, start in enumerate(range(0, len(df), chunk_size)):
            df_chunk = df.iloc[start:start + chunk_size]
            file_name = f"{output_prefix}_{i}.parquet"
            tasks.append(executor.submit(save_parquet_chunk, df_chunk, file_name))

        # Espera todas as tarefas finalizarem
        for task in tasks:
            task.result()

def save_chunks_as_parquets(df, chunk_size=100_000, output_prefix="chunk"):
    num_chunks = (len(df) // chunk_size) + 1
    for i, start in enumerate(range(0, len(df), chunk_size)):
        df_chunk = df.iloc[start:start + chunk_size]
        df_chunk.to_parquet(f"{output_prefix}_{i}.parquet", index=False)
        print(f"Saved {output_prefix}_{i}.parquet with rows.")#{len(df_chunk)} 


def notify_send(title: str, message: str, duration: int = 5):
    """
    Displays a notification on Linux using notify-send.
    
    Parameters:
    title (str): The notification title.
    message (str): The notification message.
    duration (int): Time in seconds to display the notification.
    """
    os.system(f'notify-send -t {duration * 1000} "{title}" "{message}"')
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
n=100_000
px = 30
# fcol = ['edem_w84','multi_s1_band2', 'multi_s2_band1', 'multi_s2_band2', 'multi_s2_band3']
ldem = 'multi_dtm_lidar'
pdem = 'negroaoidtm'
# rcol = 'edem_w84'
# ycol = 'zdif'

# s1col =  ['multi_s1_band1','multi_s1_band2']
# s2col = ['multi_s2_band1', 'multi_s2_band2', 'multi_s2_band3']
roi = 'mkd' #'tls'#'rng'
roi = 'tls'
if __name__ == '__main__':
#for roi in ['mkd', 'tls', 'rng']:
    if roi == 'mkd':
        fparquet_list = ldar_mkd
        print(f'processing {roi}')
        df = read_tiles_nosample(fparquet_list,ldem,ldem)
        print(f'cols: {list(df.columns)}')
        df = df.drop(pdem,axis=1)
        print(f'cols: {list(df.columns)}')
        #df[ycol] = df[rcol].subtract(df[ldem])
        #s1tcol = [ldem,rcol]+s1col
        #s2tcol = [ldem,rcol]+s2col
        outdir = f"{parq_outdir}/{roi}"
        os.makedirs(outdir, exist_ok=True)
        pqt = f"{parq_outdir}/{roi}/clean_data_"
        save_chunks_as_parquet_par(df, chunk_size=n, output_prefix=pqt, num_workers=px)
        
    elif roi == 'tls':
        fparquet_list = ldar_tls
        print(f'processing {roi}')
        df = read_tiles_nosample(fparquet_list,ldem,ldem) #rcol>ldem  for now 
        print(f'cols: {list(df.columns)}')
        df = df.drop(pdem,axis=1)
        print(f'cols: {list(df.columns)}')
        #df[ycol] = df[rcol].subtract(df[ldem])
        # s1tcol = [ldem,rcol]+s1col
        #s2tcol = [ldem,rcol]+s2col
        outdir = f"{parq_outdir}/{roi}"
        os.makedirs(outdir, exist_ok=True)
        pqt = f"{parq_outdir}/{roi}/clean_data_"
        print(df.columns.tolist())
        save_chunks_as_parquet_par(df, chunk_size=n, output_prefix=pqt, num_workers=px)

    elif roi == 'rng':
        fparquet_list = ldar_rgn
        print(f'processing {roi}')
        df = read_tiles_nosample(fparquet_list,pdem,pdem)
        print(f'cols: {list(df.columns)}')
        df = df.drop(ldem,axis=1)
        print(f'cols: {list(df.columns)}')
        #s1tcol = [pdem,rcol]+s1col
        #s2tcol = [pdem,rcol]+s2col 
        outdir = f"{parq_outdir}/{roi}"
        os.makedirs(outdir, exist_ok=True)
        pqt = f"{parq_outdir}/{roi}/clean_data_"
        df.rename(columns={pdem:ldem}, inplace=True)
        save_chunks_as_parquet_par(df, chunk_size=n, output_prefix=pqt, num_workers=px)
        print(df.columns.tolist())

notify_send("Python Script Finished @zcleantab.py", 
        f"Your script has completed execution.\n{pqt}", 5000)