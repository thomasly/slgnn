import os
import gzip

import logging

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from tqdm import tqdm


class Mol2Reader:
    
    def __init__(self, path):
        if path.endswith(".mol2"):
            with open(path, "r") as fh:
                self.file_contents = fh.readlines()
        elif path.endswith(".gz"):
            with gzip.open(path, "r") as fh:
                try:
                    lines = fh.readlines()
                    self.file_contents = [l.decode() for l in lines]
                except OSError:  ## file not readable
                    logging.error("{} is not readable with gzip".format(path))
                    self.file_contents = []
    @property        
    def n_mols(self):
        try:
            return self._n_mols
        except AttributeError:
            self._n_mols = 0
            for l in self.file_contents:
                if "@<TRIPOS>MOLECULE" in l:
                    self._n_mols += 1
            return self._n_mols
    
    def get_blocks(self):
        r""" Read the blocks in .mol2 file based on @<TRIPOS>MOLECULE label.
        return (list): list of block contents as strings.
        """
        block_starts = [i for i, l in enumerate(self.file_contents)
                        if "@<TRIPOS>MOLECULE" in l]
        blocks = list()
        i = 0
        while i+1 < len(block_starts):
            block = "".join(
                self.file_contents[block_starts[i]: block_starts[i+1]])
            blocks.append(block)
            i += 1
        blocks.append("".join(self.file_contents[block_starts[-1]:]))
        return blocks


class Mol2Filter:
    
    no_salt_atoms = ["C", "N", "O", "H", "S", "F", "Cl", "Br"]

    def add_no_salt(cls, atom):
        r""" add atom to the no_salt_atoms list
        """
        cls.no_salt_atoms.append(atom)
        
    def set_no_salt(cls, atoms: list):
        r""" Set the whole no_salt_atoms to given atoms list
        """
        cls.no_salt_atoms = atoms
    
    def ghose_filter(self, filepath, exclude_salt=False):
        r""" Filter the given file with ghose filter. If exclude_salt is true,
        the molecule with atoms not in the no_salt_atoms list will also be
        filtered out.
        file_path (str): path to the .mol2 file or .gz file.
        exclude_salt(bool): if filter out the molecule containg salt atoms.
        =======================================================================
        return (str): filtered string in Mol2 file format. 
        """
        reader = Mol2Reader(filepath)
        blocks = reader.get_blocks()
        filtered = list()
        for block in tqdm(blocks):
            mol = Chem.rdmolfiles.MolFromMol2Block(block, sanitize=False)
            if mol is None:
                continue
            n_atoms = mol.GetNumAtoms()
            if n_atoms < 20 or n_atoms > 70:
                continue
            mw = ExactMolWt(mol)
            if mw < 180 or mw > 480:
                continue
            if exclude_salt:
                atoms = mol.GetAtoms()
                flag = 0
                for atom in atoms:    
                    if atom.GetSymbol() not in self.no_salt_atoms:
                        flag = 1
                        break
                if flag == 1:
                    continue
            filtered.append(block)

        return "\n\n".join(filtered)
    
    __filters = {"ghose": ghose_filter}
    
    def write_out(self, content, outfile, compress=True):
        r""" Write content to outfile. Compress with gzip if compress is True.
        content (str): content for writing into outfile
        outfile (str): output file path. The path must exist.
        compress (bool): if to compress the output file with gzip.
        """
        if compress:
            with open(outfile, "wb") as outf:
                outf.write(gzip.compress(content.encode("utf-8")))
        else:
            with open(outfile, "w") as outf:
                outf.write(content)
                
    def filter_all(self,
                   dirpath,
                   outpath,
                   method="ghose",
                   exclude_salt=False,
                   compress=True):
        r""" Filter all the .mol2 or .gz files with ghose filter in the
        directory. The molecules will be writen into the file with the same
        filename as the input files. Empty files will NOT be written out.
        dirpath (str): diretory contains the .mol2 or .gz files. Directory with
                       subfolders is acceptable.
        outpath (str): the output diretory path. The directory will be created
                       if it does not exist.
        method (str): method to filter the molecules. Default is Ghose filter.
        exclude_salt (bool): if to filter out the molecules containing salt
                             atoms.
        compress (bool): if True, the output file will be compressed with gzip.
        """
        outpath = os.path.abspath(outpath)
        os.makedirs(outpath, exist_ok=True)
        filter_ = self.__filters[method]
        for item in tqdm(list(os.scandir(dirpath))):
            if item.name.split(".")[-1] in ["mol2", "gz"]:
                filtered = filter_(self, item.path, exclude_salt=exclude_salt)
                if len(filtered) == 0:
                    continue
                self.write_out(filtered,
                               outfile=os.path.join(outpath, item.name),
                               compress=compress)
            elif item.is_dir():
                self.filter_all(item,
                                outpath=outpath,
                                method=method,
                                exclude_salt=exclude_salt,
                                compress=compress)
            else:
                continue


if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument("-i", "--dirpath")
    argparser.add_argument("-o", "--outpath")
    args = argparser.parse_args()
    mol2_filter = Mol2Filter()
    mol2_filter.filter_all(
        dirpath=args.dirpath,
        outpath=args.outpath,
        method="ghose",
        exclude_salt=True,
        compress=True,
    )
    