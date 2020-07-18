import random


def get_pubchem_fingerprint(sm):
    from PyFingerprint.All_Fingerprint import get_fingerprint

    return get_fingerprint(sm, fp_type="pubchem", output="vector")


class fix_random_seed:

    _seed = 0

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        random.seed(self._seed)
        ret = self.func(*args, **kwargs)
        random.seed()
        return ret

    @classmethod
    def seed(cls, value: int):
        cls._seed = value
        return cls
