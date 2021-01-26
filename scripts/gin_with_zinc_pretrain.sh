CUDA_VISIBLE_DEVICES=$1 python \
-m slgnn.training.train_gin_with_zinc_pretrained_model \
-p trained_models/pretrained_GIN_with_ZINC_epoch62.pt
#> nohup_zinc_pretrained_model_experiments_3.out
