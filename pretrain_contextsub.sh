#!/bin/bash

DATASET=chembl

CUDA_VISIBLE_DEVICES=$1 nohup python -m contextSub.pretrain_contextsub \
    --device 0 \
    --batch_size 64 \
    --epochs 200 \
    --lr 0.001 \
    --decay 0.001 \
    --num_layer 5 \
    --csize 3 \
    --emb_dim 300 \
    --dropout_ratio 0.3 \
    --neg_samples 3 \
    --JK last \
    --context_pooling mean \
    --mode cbow \
    --dataset ${DATASET} \
    --output_model_file contextSub/trained_models/contextSub_${DATASET}_partialCharge \
    --logpath contextSub/runs/pretraining/${DATASET} \
    --gnn_type gin \
    --seed 0 \
    --num_workers 4 \
    --partial_charge > nohup_logs/contextSub_pretrain_${DATASET}_partialCharge.out &
