#!/bin/bash

DEVICE=2

# contextSub
# for DATASET in bace bbbp clintox sider tox21 toxcast hiv muv
# for DATASET in tox21 toxcast hiv muv
# do
#     for SEED in 1 2 3 4 5
#     do
#         nohup python -m contextSub.finetune \
#             --device ${DEVICE} \
#             --batch_size 128 \
#             --epochs 100 \
#             --lr 0.001 \
#             --lr_scale 1 \
#             --decay 0.0001 \
#             --num_layer 5 \
#             --emb_dim 300 \
#             --dropout_ratio 0.3 \
#             --graph_pooling mean \
#             --JK last \
#             --gnn_type gin \
#             --dataset ${DATASET} \
#             --input_model_file contextSub/trained_models/contextSub_chembl_2.pth \
#             --filename contextSub_${DATASET} \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train 0 \
#             --num_workers 4 > nohup_logs/contextSub_finetune_seed${SEED}_${DATASET}.out
#         wait
#     done
# done

# contextSub_partialCharge
# for DATASET in bace bbbp clintox sider tox21 toxcast hiv muv
for DATASET in clintox sider tox21 toxcast hiv
do
    for SEED in 1 2 3 4 5
    do
        nohup python -m contextSub.finetune \
            --device ${DEVICE} \
            --batch_size 128 \
            --epochs 100 \
            --lr 0.001 \
            --lr_scale 1 \
            --decay 0.0001 \
            --num_layer 5 \
            --emb_dim 300 \
            --dropout_ratio 0.3 \
            --graph_pooling mean \
            --JK last \
            --gnn_type gin \
            --dataset ${DATASET} \
            --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge.pth \
            --filename contextSub_partialCharge_${DATASET} \
            --seed ${SEED} \
            --runseed ${SEED} \
            --split scaffold \
            --eval_train 0 \
            --num_workers 4 \
            --partial_charge > nohup_logs/contextSub_partialCharge_finetune_seed${SEED}_${DATASET}.out
        wait
    done
done

# contextPred
# for DATASET in bace bbbp clintox sider tox21 toxcast hiv muv
# for DATASET in sider tox21 toxcast hiv muv
# do
#     for SEED in 1 2 3 4 5
#     do
#         nohup python -m contextSub.finetune \
#             --device ${DEVICE} \
#             --batch_size 128 \
#             --epochs 100 \
#             --lr 0.001 \
#             --lr_scale 1 \
#             --decay 0.0001 \
#             --num_layer 5 \
#             --emb_dim 300 \
#             --dropout_ratio 0.3 \
#             --graph_pooling mean \
#             --JK last \
#             --gnn_type gin \
#             --dataset ${DATASET} \
#             --input_model_file contextSub/trained_models/contextpred.pth \
#             --filename contextPred_${DATASET} \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train 0 \
#             --num_workers 4 > nohup_logs/contextPred_finetune_seed${SEED}_${DATASET}.out
#         wait
#     done
# done
