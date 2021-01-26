CUDA_VISIBLE_DEVICES=$1 \
python -m STDSED.data_processing.ecfp2ginFP \
--dataset-name chembl \
--output-path data/chembl24_smiles_GINfp.json
