FOR %%i IN (1000, 10000, 100000) DO ^
python -m slgnn.data_processing.zinc_to_graph -p data\ZINC\sampled_smiles_%%i.txt