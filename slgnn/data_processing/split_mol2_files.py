import os
import gzip
import multiprocessing as mp
import logging

from chemreader.readers.readmol2 import Mol2Reader


def write_out(content, outfile, compress=True):
    if compress:
        with open(outfile, "wb") as outf:
            outf.write(gzip.compress(content.encode("utf-8")))
    else:
        with open(outfile, "w") as outf:
            outf.write(content)


def split_item(item_path, outpath, compress):
    logging.info("splitting {}".format(item_path))
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
    os.makedirs(outpath, exist_ok=True)
    items = list(os.scandir(dirpath))
    pool = mp.Pool(int(os.cpu_count() / 2))
    try:
        pool.starmap_async(split_item,
                           [(item.path, outpath, compress) for item in items])
    except KeyboardInterrupt:
        pool.terminate()
    else:
        pool.close()
    finally:
        pool.join()


if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument("-i", "--dirpath")
    argparser.add_argument("-o", "--outpath")
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)
    split_mol2_in(dirpath=args.dirpath, outpath=args.outpath, compress=True)
