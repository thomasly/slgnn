python -m slgnn.training.train_gcn ^
--encoder-epochs 50 ^
--classifier-epochs 50 ^
--encoder-lr 0.00001 ^
--classifier-lr 0.00001 ^
--encoder-data data/ZINC/zinc_ghose_1000.hdf5
