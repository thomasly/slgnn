python -m slgnn.training.train_gcn ^
--encoder-epochs 4 ^
--classifier-epochs 10 ^
--encoder-lr 0.0001 ^
--classifier-lr 0.0001 ^
--encoder-data data/ZINC/sampled_smiles_100000.txt

python -m slgnn.training.train_gcn ^
--encoder-epochs 10 ^
--classifier-epochs 10 ^
--encoder-lr 0.0001 ^
--classifier-lr 0.0001 ^
--encoder-data data/ZINC/sampled_smiles_100000.txt ^
--no-encoder
