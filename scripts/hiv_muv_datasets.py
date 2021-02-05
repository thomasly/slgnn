from slgnn.data_processing.deepchem_datasets import HIV, HIVFP, MUV, MUVFP

def main():
    idx = 99
    
    datafp = HIV(fp_type="pubchem")
    dataecfp = HIV(fp_type="ecfp")
    datafg = HIV(fragment_label=True)
    datafg_fp = HIV(fragment_label=True, fp_type="pubchem")
    datafg_ecfp = HIV(fragment_label=True, fp_type="ecfp")
    data = HIV()
    print("HIV: ", data)
    print("HIV: ", data[idx])
    print("HIV: ", datafp[idx])
    print("HIV: ", dataecfp[idx])
    print("HIV: ", datafg[idx])
    print("HIV: ", datafg_fp[idx])
    print("HIV: ", datafg_ecfp[idx])
    
    datafp = MUVFP()
    dataecfp = MUV(fp_type="ecfp")
    datafg = MUV(fragment_label=True)
    datafg_fp = MUV(fragment_label=True, fp_type="pubchem")
    datafg_ecfp = MUV(fragment_label=True, fp_type="ecfp")
    data = MUV()
    print("MUV:", data)
    print("MUV:", data[idx])
    print("MUV:", datafp[idx])
    print("MUV:", dataecfp[idx])
    print("MUV:", datafg[idx])
    print("MUV:", datafg_fp[idx])
    print("MUV:", datafg_ecfp[idx])
    
if __name__ == "__main__":
    main()