from rdkit.Chem import MolFromSmiles as MFS
from tqdm import tqdm

atoms = set()
with open("data/ZINC/ghose_filtered/raw/smiles.txt") as f:
  for smiles in tqdm(f.readlines()):
    mol = MFS(smiles.strip())
    if mol is None:
      continue
    mol_atoms = mol.GetAtoms()
    for atom in mol_atoms:
      atoms.add(atom.GetSymbol())
print(atoms)
print(len(atoms))
with open("data/ZINC/ghose_filtered/raw/atom_types.txt", "w") as f:
  f.write(" ".join(atoms))
