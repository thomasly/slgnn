import os
import multiprocessing as mp
import logging

from chemreader.readers import Mol2
from tqdm import tqdm


def listener(q, outpath):
    pid = os.getpid()
    logging.info("Listener running at pid: {}".format(pid))
    outf = open(outpath, "w")
    outf.write("SMILES,mol_name\n")
    counter = 0
    while 1:
        data = q.get()
        if data == "kill":
            logging.info("Listener killed.")
            outf.close()
            return
        outf.write(data)
        counter += 1
        if counter % 100 == 0:
            logging.info("{} SMILES written to {}".format(counter, outpath))


def worker(path, q):
    mols = Mol2(path).mol2_blocks
    for m in mols:
        smiles = m.to_smiles(isomeric=True)
        name = m.mol_name
        q.put("{},{}\n".format(smiles, name))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    files = os.scandir("../ghose_filtered")
    pool = mp.Pool(10)
    manager = mp.Manager()
    que = manager.Queue()
    lsnr = mp.Process(target=listener, args=(que, "ghose_filtered_smiles.csv")) 
    for f in files:
        if not f.name.endswith(".gz"):
            continue
        pool.apply_async(worker, args=(f.path, que))
    lsnr.start()
    pool.close()
    pool.join()
    que.put("kill")
    lsnr.join()
