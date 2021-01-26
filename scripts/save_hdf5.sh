#!/bin/bash

python -m src.save_to_hdf5 -p data/ZINC/split_ghose_filtered -n 1000 -o data/ZINC/zinc_ghose_1000.hdf5 -v &
python -m src.save_to_hdf5 -p data/ZINC/split_ghose_filtered -n 10000 -o data/ZINC/zinc_ghose_10000.hdf5 -v &
python -m src.save_to_hdf5 -p data/ZINC/split_ghose_filtered -n 100000 -o data/ZINC/zinc_ghose_100000.hdf5 -v &
