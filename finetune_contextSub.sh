#!/bin/bash

DATASET=bace

for SEED in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=$1 nohup python -m contextSub.finetune \
        --device 0 \
        --batch_size 128 \
        --epochs 100 \
        --lr 0.001 \
        --lr_scale 1 \
        --decay 0 \
        --num_layer 5 \
        --emb_dim 300 \
        --dropout_ratio 0.5 \
        --graph_pooling mean \
        --JK last \
        --gnn_type gin \
        --dataset ${DATASET} \
        --input_model_file contextSub/trained_models/$2 \
        --filename ${DATASET} \
        --seed ${SEED} \
        --runseed ${SEED} \
        --split scaffold \
        --eval_train 0 \
        --num_workers 4 >> contextSub_finetune_${DATASET}.out
done
