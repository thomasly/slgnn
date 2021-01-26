CUDA_VISIBLE_DEVICES=$1 \
python -m slgnn.training.train_gin_with_zinc_pretrained_model_covid19_datasets \
-p trained_models/pretrained_GIN_with_ZINC_epoch27.pt
