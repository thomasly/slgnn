CUDA_VISIBLE_DEVICES=$1 python \
    -m STDSED.training.pretrain_gin \
    -c ./model_configs/GIN_pretrain_drugbank_cyp450.yml