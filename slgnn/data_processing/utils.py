import random
import functools

# import inspect


def get_pubchem_fingerprint(smiles):
    """ Generate pubchem fingerprint from SMILES.

    Args:
        smiles (str): the SMILES string
    """
    from PyFingerprint.All_Fingerprint import get_fingerprint

    return get_fingerprint(smiles, fp_type="pubchem", output="vector")


class fix_random_seed:
    """ Decorator fix the random seed within the span of a function. The random seed is
    revert to None after the function's execution.

    Args:
        seed (int): the random seed.
    """

    def __init__(self, seed=0):
        self._seed = seed

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            random.seed(self._seed)
            ret = func(*args, **kwargs)
            random.seed()
            return ret

        return wrapper
