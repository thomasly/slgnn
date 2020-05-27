from PyFingerprint.All_Fingerprint import get_fingerprint


def get_pubchem_fingerprint(sm):
    return get_fingerprint(sm, fp_type="pubchem", output="vector")
