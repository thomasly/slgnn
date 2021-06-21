#!/bin/bash

DATASET=chembl
DEVICE=2

# increased training epochs from 200 to 300
# CUDA_VISIBLE_DEVICES=$1 python -m contextSub.pretrain_contextsub \
#     --device 0 \
#     --batch_size 64 \
#     --epochs 300 \
#     --lr 0.001 \
#     --decay 0.001 \
#     --num_layer 5 \
#     --csize 3 \
#     --emb_dim 300 \
#     --dropout_ratio 0.3 \
#     --neg_samples 1 \
#     --JK last \
#     --context_pooling mean \
#     --mode cbow \
#     --dataset ${DATASET} \
#     --output_model_file contextSub/trained_models/contextSub_${DATASET}_partialCharge_no_norm_output_mlp_epoch300 \
#     --logpath contextSub/runs/pretraining/partial_charge_no_norm_output_mlp_epoch300_${DATASET} \
#     --gnn_type gin \
#     --seed 0 \
#     --partial_charge \
#     --input_mlp \
#     --node_feat_dim 3 \
#     --edge_feat_dim 2 \
#     --num_workers 4 > nohup_logs/contextSub_pretrain_${DATASET}_partialCharge_no_norm_mlp_epoch300.out &

## mlp + norm + partial charge + 300 epochs
# CUDA_VISIBLE_DEVICES=$1 python -m contextSub.pretrain_contextsub \
#     --device 0 \
#     --batch_size 64 \
#     --epochs 300 \
#     --lr 0.001 \
#     --decay 0.001 \
#     --num_layer 5 \
#     --csize 3 \
#     --emb_dim 300 \
#     --dropout_ratio 0.3 \
#     --neg_samples 1 \
#     --JK last \
#     --context_pooling mean \
#     --mode cbow \
#     --dataset ${DATASET} \
#     --output_model_file contextSub/trained_models/contextSub_${DATASET}_partialCharge_normOutput_mlp_epoch300 \
#     --logpath contextSub/runs/pretraining/partialCharge_normOutput_mlp_epoch300_${DATASET} \
#     --gnn_type gin \
#     --seed 0 \
#     --norm_output \
#     --partial_charge \
#     --input_mlp \
#     --node_feat_dim 3 \
#     --edge_feat_dim 2 \
#     --num_workers 4 > nohup_logs/contextSub_pretrain_${DATASET}_partialCharge_normOutput_mlp_epoch300.out &

## filtered substructures, no norm, partial charge, embedding input
# python -m contextSub.pretrain_contextsub \
#     --device 0 \
#     --batch_size 64 \
#     --epochs 300 \
#     --lr 0.001 \
#     --decay 0.001 \
#     --num_layer 5 \
#     --csize 3 \
#     --emb_dim 300 \
#     --dropout_ratio 0.3 \
#     --neg_samples 3 \
#     --JK last \
#     --context_pooling mean \
#     --mode cbow \
#     --dataset ${DATASET} \
#     --output_model_file contextSub/trained_models/contextSub_${DATASET}_partialCharge_noNorm_filteredPattern_epoch300 \
#     --logpath contextSub/runs/pretraining/partialCharge_noNorm_filteredPattern_epoch300_${DATASET} \
#     --gnn_type gin \
#     --seed 0 \
#     --partial_charge \
#     --node_feat_dim 3 \
#     --edge_feat_dim 2 \
#     --num_workers 4 \
#     --pattern_path contextSub/resources/pubchemFPKeys_to_SMARTSpattern_filtered.csv > nohup_logs/contextSub_pretrain_${DATASET}_partialCharge_noNormOutput_filteredPattern_epoch300.out

## filtered substructures, normalized, partial charge, mlp input
# CUDA_VISIBLE_DEVICES=$1 python -m contextSub.pretrain_contextsub \
#     --device 0 \
#     --batch_size 64 \
#     --epochs 300 \
#     --lr 0.001 \
#     --decay 0.001 \
#     --num_layer 5 \
#     --csize 3 \
#     --emb_dim 300 \
#     --dropout_ratio 0.3 \
#     --neg_samples 1 \
#     --JK last \
#     --context_pooling mean \
#     --mode cbow \
#     --dataset ${DATASET} \
#     --output_model_file contextSub/trained_models/contextSub_${DATASET}_partialCharge_normOutput_mlp_epoch300 \
#     --logpath contextSub/runs/pretraining/partialCharge_normOutput_mlp_epoch300_${DATASET} \
#     --gnn_type gin \
#     --seed 0 \
#     --norm_output \
#     --partial_charge \
#     --input_mlp \
#     --node_feat_dim 3 \
#     --edge_feat_dim 2 \
#     --num_workers 4 \
#     --pattern_path contextSub/resources/pubchemFPKeys_to_SMARTSpattern_filtered.csv > nohup_logs/contextSub_pretrain_${DATASET}_partialCharge_normOutput_mlp_epoch300.out &

## filtered substructures, normalized, partial charge, embedding input
python -m contextSub.pretrain_contextsub \
    --device ${DEVICE} \
    --batch_size 64 \
    --epochs 300 \
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
    --output_model_file contextSub/trained_models/contextSub_${DATASET}_partialCharge_Normalized_filteredPattern_epoch300 \
    --logpath contextSub/runs/pretraining/partialCharge_Normalized_filteredPattern_epoch300_${DATASET} \
    --gnn_type gin \
    --seed 0 \
    --norm_output \
    --partial_charge \
    --node_feat_dim 3 \
    --edge_feat_dim 2 \
    --num_workers 4 \
    --pattern_path contextSub/resources/pubchemFPKeys_to_SMARTSpattern_filtered.csv > nohup_logs/contextSub_pretrain_${DATASET}_partialCharge_Normalized_filteredPattern_epoch300.out