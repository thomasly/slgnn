CUDA_VISIBLE_DEVICES=$1 nohup \
python -m slgnn.training.pretrain_gin_with_zinc \
-c model_configs/GIN_pretrain.yml \
> pretrain_GIN.out 2> pretrain_GIN.err &
