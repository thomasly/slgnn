import random
import functools

# import inspect


def get_pubchem_fingerprint(sm):
    from PyFingerprint.All_Fingerprint import get_fingerprint

    return get_fingerprint(sm, fp_type="pubchem", output="vector")


class fix_random_seed:
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
