#!/bin/bash

DATASET=chembl

CUDA_VISIBLE_DEVICES=$1 nohup python -m contextSub.pretrain_contextsub \
    --device 0 \
    --batch_size 1024 \
    --epochs 100 \
    --lr 0.001 \
    --decay 0.001 \
    --num_layer 5 \
    --csize 3 \
    --emb_dim 300 \
    --dropout_ratio 0 \
    --neg_samples 1 \
    --JK last \
    --context_pooling mean \
    --dataset ${DATASET} \
    --output_model_file contextSub/trained_models/contextSub_${DATASET}_nodropout_1 \
    --gnn_type gin \
    --seed 0 \
    --num_workers 1 > contextSub_pretrain_${DATASET}_nodropout.out &
