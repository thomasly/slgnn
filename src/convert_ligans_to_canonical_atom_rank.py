import os
from shutil import copyfile
import multiprocessing as mp
import sys

from chemreader.readers import CanonicalAtomOrderConverter as CAOC
from rdkit import Chem
from tqdm import tqdm

save_path = os.path.join("..", "data", "canonical_atom_rank_ligand", "ghose")
none_path = os.path.join("..", "data", "canonical_atom_rank_ligand", "not_convertable")
os.makedirs(save_path, exist_ok=True)
os.makedirs(none_path, exist_ok=True)


def listener(q):
    while 1:
        content = q.get()
        if content == "END":
            return
        else:
            with open(content["path"], "w") as f:
                f.write(content["pdb"])

                
def converter(inpath, q):
    
    def atoms_in_file(inf):
        counter = 0
        for line in inf.readlines():
            if "HETATM" in line:
                counter += 1
        return counter
        
    print(f"working on {inpath}                                    ", end="\r")
    with open(inpath, "r") as inf:
        if not 19 < atoms_in_file(inf) < 71:
            return
    mol = Chem.MolFromPDBFile(inpath, removeHs=False)
    if mol is not None:
        new_mol = CAOC(mol).convert()
        outname = os.path.basename(inpath)
        q.put(
            {
                "path": os.path.join(save_path, outname.split(".")[0]+".pdb"),
                "pdb": Chem.MolToPDBBlock(new_mol)
            }
        )
    else:
        copyfile(inpath, os.path.join(none_path, outname))


if __name__ == "__main__":
    sys.stderr = open("stderr.txt", "w")
    sys.stdout = open("stdout.txt", "w")
    manager = mp.Manager()
    q = manager.Queue(4)

    l = mp.Process(target=listener, args=(q,))
    l.start()

    pool = mp.Pool(20)
    for f in tqdm(list(os.scandir(os.path.join("..", "data", "ligand")))):
        if not f.name.endswith(".pdb"):
            continue
        async_results = pool.apply_async(converter, args=(f.path, q))
 
    pool.close()
    _ = async_results.get()
    pool.join()
    
    q.put("END")
    l.join()
    sys.stderr.close()
    sys.stdout.close()
