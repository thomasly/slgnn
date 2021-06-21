from random import sample
from collections import defaultdict

from chemreader.readers import Smiles
from rdkit.DataStructs import BulkTanimotoSimilarity
import numpy as np
from tqdm import tqdm


class Cluster:
    """ Cluster dataset based on Tanimoto scores.

    Args:
        threshold (float): the cutoff value used to determine if two chemicals are
            similar. Chemicals with a Tanimoto similarity score higher than the value
            will be considered similar to each other.

    Attributes:
        threshold (float): the cutoff value.
    """

    def __init__(self, threshold=0.5):
        self._threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def clustering(self, smiles: list, verbose=0):
        """ Clustering the smiles with Tanimoto similarity.

        Args:
            smiles (list): list of SMILES strings.
            verbose (bool): whether showing the progress bar.

        Returns:
            list: SMILES clusters.
        """
        self.clusters = dict()
        counter = 0
        it = tqdm(smiles) if verbose else smiles
        for s in it:
            s = Smiles(s)
            if s.rdkit_mol is None:
                continue
            idx = self._find_similarity(s)
            if idx is None:
                self.clusters[counter] = [s]
                counter += 1
            else:
                self._update_cluster(idx, s)
        self.clusters = list(self.clusters.values())
        return self.clusters

    def _find_similarity(self, smiles):
        indices = list()
        if len(self.clusters) == 0:
            return None
        for i, clu in self.clusters.items():
            clu_fps = [s.fingerprint for s in clu if s.fingerprint is not None]
            sim = BulkTanimotoSimilarity(smiles.fingerprint, clu_fps)
            best_match = max(sim)
            if best_match > self.threshold:
                indices.append(i)
        if len(indices) != 0:
            return indices
        else:
            return None

    def _update_cluster(self, idx, smiles):
        """ Helper function of clustering. If idx has only one element, add smiles to
        self.cluster accordingly. If idx has more than one element, combine theses
        clusters and add smiles to the combined cluster.
        """
        self.clusters[idx[0]].append(smiles)
        if len(idx) == 1:
            return
        for i in idx[1:]:
            self.clusters[idx[0]].extend(self.clusters[i])
            self.clusters.pop(i)


def clustering(smiles_list, threshold=0.75, verbose=False):
    smiles_list = np.array(smiles_list)
    fp_list = [Smiles(sms).fingerprint for sms in smiles_list]
    smiles_set = set(smiles_list)
    clusters = []
    n = 1
    while len(smiles_set) > 0:
        c = Smiles(sample(smiles_set, 1)[0])
        sim_scores = BulkTanimotoSimilarity(c.fingerprint, fp_list)
        clusters.append(set(smiles_list[np.array(sim_scores) > threshold]))
        smiles_set = smiles_set - clusters[-1]
        if verbose:
            print(f"Step: {n}")
            n += 1
    return clusters


def merge_common(lists):
    """ Merge function to  merge all sublist having common elements.
    """
    neigh = defaultdict(set)
    visited = set()
    for each in lists:
        for item in each:
            neigh[item].update(each)

    def comp(node, neigh=neigh, visited=visited, vis=visited.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            vis(node)
            nodes |= neigh[node] - visited
            yield node

    for node in neigh:
        if node not in visited:
            yield sorted(comp(node))
