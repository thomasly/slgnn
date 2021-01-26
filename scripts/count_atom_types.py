import multiprocessing as mp

from rdkit.Chem import MolFromSmiles


def get_atom_symbols(mol, q):
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    q.put(set(atoms))


def counter(q):
    atoms = set()
    while 1:
        new_atoms = q.get()
        if new_atoms == "END":
            break
        atoms = atoms.union(new_atoms)
        print(f"Current atoms: {atoms}", end="\r")
    print()
    print(atoms)
    print(len(atoms))
    with open("data/ZINC/ghose_filtered/raw/counter_atoms.txt", "w") as f:
        f.write(" ".join(atoms))
        f.write("\n")
        f.write(f"n atom types: {len(atoms)}")


if __name__ == "__main__":
    n_cpu = mp.cpu_count()
    manager = mp.Manager()
    q = manager.Queue(maxsize=n_cpu * 5)
    pool = mp.Pool(int(n_cpu / 2))
    with open("data/ZINC/ghose_filtered/raw/smiles.txt") as f:
        for line in f.readlines():
            smiles = line.strip()
            mol = MolFromSmiles(smiles)
            pool.apply_async(get_atom_symbols, args=[mol, q])
    pool.close()
    ct = mp.Process(target=counter, args=[q])
    ct.start()
    pool.join()
    q.put("END")
    ct.join()
