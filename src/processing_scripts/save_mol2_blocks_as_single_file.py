import os
import gzip
import multiprocessing as mp
import logging

from chemreader.readers.readmol2 import Mol2Reader
from tqdm import tqdm


def write_out(content, outfile, compress=True):
    if compress:
        with open(outfile, "wb") as outf:
            outf.write(gzip.compress(content.encode("utf-8")))
    else:
        with open(outfile, "w") as outf:
            outf.write(content)


def split_item(args):
    item_path, outpath, compress = args
    os.makedirs(outpath, exist_ok=True)
    blocks = Mol2Reader(item_path).blocks
    if len(blocks) == 0:
        return
    for i, block in enumerate(blocks):
        fn = item_path.split(
            os.path.sep)[-1].split(".mol2")[0] + "_" + str(i) + ".mol2"
        if compress:
            fn += ".gz"
        write_out(block, outfile=os.path.join(outpath, fn), compress=compress)


def split_mol2_in(dirpath, outpath, compress=True):
    outpath = os.path.abspath(outpath)
    items = list(os.scandir(dirpath))
    pool = mp.Pool(int(os.cpu_count() / 2))
    try:
        iterable = [
            (item.path,
             os.path.join(outpath, item.name.split(".mol2")[0]),
             compress) for item in items if
            item.name.split(".")[-1] in ["mol2", "gz"]]
        pb = tqdm(total=len(iterable))
        for _ in pool.imap_unordered(split_item, iterable):
            pb.update(1)
    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument("-i", "--dirpath")
    argparser.add_argument("-o", "--outpath")
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)
    split_mol2_in(dirpath=args.dirpath, outpath=args.outpath, compress=True)
