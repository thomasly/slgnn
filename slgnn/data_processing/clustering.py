from chemreader.readers import Smiles
from rdkit.DataStructs import BulkTanimotoSimilarity


class Cluster:
    def __init__(self, threshold=0.5):
        self._threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def clustering(self, smiles: list):
        """ Clustering the smiles with Tanimoto similarity.
        """
        self.clusters = dict()
        counter = 0
        for s in smiles:
            s = Smiles(s)
            if s.rdkit_mol is None:
                continue
            idx = self._find_similarity(s)
            if idx is None:
                self.clusters[counter] = [s]
                counter += 1
            else:
                self._update_cluster(idx, s)

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
        """ If idx has only one element, add smiles to self.cluster accordingly.
        If idx has more than one element, combine theses clusters and add smiles to the
        combined cluster.
        """
        self.clusters[idx[0]].append(smiles)
        if len(idx) == 1:
            return
        for i in idx[1:]:
            self.clusters[idx[0]].extend(self.clusters[i])
            self.clusters.pop(i)
