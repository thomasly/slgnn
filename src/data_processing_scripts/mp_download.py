import os
import multiprocessing as mp
from tqdm import tqdm


if __name__ == "__main__":
    with open("ZINC-downloader-3D-mol2.gz.wget") as fh:
        commands = fh.readlines()
    pool = mp.Pool(20)
    pool.map(os.system, tqdm(commands))
    pool.close()
    pool.join()
