#!/bin/bash

DEVICE=2

# contextPred
# for DATASET in bace bbbp clintox sider tox21 toxcast hiv muv
# for DATASET in sider tox21 toxcast hiv muv
# for DATASET in lightbbb
# do
#     for SEED in 1
#     do
#         python -m contextSub.finetune \
#             --device ${DEVICE} \
#             --batch_size 128 \
#             --epochs 100 \
#             --lr 0.001 \
#             --lr_scale 1 \
#             --decay 0.0001 \
#             --num_layer 5 \
#             --emb_dim 300 \
#             --dropout_ratio 0.4 \
#             --graph_pooling mean \
#             --JK last \
#             --gnn_type gin \
#             --dataset ${DATASET} \
#             --input_model_file contextSub/trained_models/supervised_contextpred.pth \
#             --filename supervised_contextPred_scaffoldsplit_${DATASET} \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train \
#             --save_model contextSub/trained_models/finetuned/contextPred/ \
#             --num_workers 4 > nohup_logs/supervised_contextPred_scaffoldsplit_finetune_seed${SEED}_${DATASET}.out 2>&1
#     done
# done


# all data sets 2-step freeze
# for DATASET in lightbbb bace bbbp clintox sider tox21 toxcast sider
# for DATASET in lightbbb
# do
#     for SEED in 1 2 3 4 5
#     do
#         python -m contextSub.finetune \
#             --device ${DEVICE} \
#             --batch_size 16 \
#             --epochs 120 \
#             --lr 0.0001 \
#             --lr_scale 1 \
#             --decay 0.0001 \
#             --num_layer 5 \
#             --emb_dim 300 \
#             --dropout_ratio 0.4 \
#             --graph_pooling mean \
#             --JK last \
#             --gnn_type gin \
#             --dataset ${DATASET} \
#             --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_noNorm_filteredPattern_epoch300.pth \
#             --filename 2stepfreeze_scaffoldsplit_${DATASET}_contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04 \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train \
#             --num_workers 1 \
#             --partial_charge \
#             --sub_input \
#             --context \
#             --pooling_indicator \
#             --contextpred \
#             --contextpred_model_file contextSub/trained_models/supervised_contextpred.pth \
#             --freeze 20 \
#             --two_step_freeze > nohup_logs/2stepfreeze_scaffoldsplit_${DATASET}_contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04_finetune_seed${SEED}.out 2>&1
#     done
# done

# for DATASET in lightbbb
# do
#     for SEED in 1
#     do
#         python -m contextSub.finetune \
#             --device ${DEVICE} \
#             --batch_size 16 \
#             --epochs 120 \
#             --lr 0.0001 \
#             --lr_scale 1 \
#             --decay 0.0001 \
#             --num_layer 5 \
#             --emb_dim 300 \
#             --dropout_ratio 0.4 \
#             --graph_pooling mean \
#             --JK last \
#             --gnn_type gin \
#             --dataset ${DATASET} \
#             --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_noNorm_filteredPattern_epoch300.pth \
#             --filename nosubpooling_2stepfreeze_scaffoldsplit_${DATASET}_contextSub_partialCharge_noNorm_subinput_doubleOutput_dropout04 \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train \
#             --num_workers 4 \
#             --partial_charge \
#             --sub_input \
#             --context \
#             --contextpred \
#             --contextpred_model_file contextSub/trained_models/supervised_contextpred.pth \
#             --freeze 20 \
#             --two_step_freeze > nohup_logs/nosubpooling_2stepfreeze_scaffoldsplit_${DATASET}_contextSub_partialCharge_noNorm_subinput_doubleOutput_dropout04_finetune_seed${SEED}.out 2>&1
#     done
# done


# mlp model, no norm
# for DATASET in bace bbbp clintox sider tox21 toxcast hiv muv
# for DATASET in bace
# do
#     for SEED in 1
#     # for SEED in 1 2 3 4
#     do
#         python -m contextSub.finetune \
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
#             --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_no_norm_output_mlp_epoch300.pth \
#             --filename contextSub_partialCharge_noNorm_mlp_300epoch_${DATASET}_test \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train 0 \
#             --num_workers 1 \
#             --partial_charge \
#             --input_mlp \
#             --node_feat_dim 3 \
#             --edge_feat_dim 2 > nohup_logs/contextSub_partialCharge_noNorm_mlp_300epoch_finetune_seed${SEED}_${DATASET}_test.out 2>&1 &
#     done
#     wait
# done


# substructure level output
# for DATASET in bace bbbp clintox sider tox21 toxcast hiv muv
# do
#     for SEED in 1 2 3 4 5
#     do
#         python -m contextSub.finetune \
#             --device ${DEVICE} \
#             --batch_size 32 \
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
#             --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_no_norm_output.pth \
#             --filename contextSub_partialCharge_noNorm_subLevel_${DATASET}_2 \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train 0 \
#             --num_workers 1 \
#             --partial_charge \
#             --sub_level > nohup_logs/contextSub_partialCharge_noNorm_subLevel_finetune_seed${SEED}_${DATASET}_2.out 2>&1
#     done
# done


# substructure level input
# for DATASET in bbbp clintox sider tox21 toxcast
# do
#     for SEED in 1 2 3 4 5
#     do
#         python -m contextSub.finetune \
#             --device ${DEVICE} \
#             --batch_size 32 \
#             --epochs 300 \
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
#             --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_no_norm_output.pth \
#             --filename contextSub_partialCharge_noNorm_subinput_context_${DATASET} \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train 0 \
#             --num_workers 1 \
#             --partial_charge \
#             --sub_input \
#             --context > nohup_logs/contextSub_partialCharge_noNorm_subinput_context_finetune_seed${SEED}_${DATASET}.out 2>&1
#     done
# done


# substructure level input and substructure level output with pooling_indicators
# for DATASET in bace bbbp clintox sider tox21 toxcast sider hiv
for DATASET in lightbbb
do
    for SEED in 1
    do
        python -m contextSub.finetune \
            --device ${DEVICE} \
            --batch_size 32 \
            --epochs 100 \
            --lr 0.0001 \
            --lr_scale 1 \
            --decay 0.0001 \
            --num_layer 5 \
            --emb_dim 300 \
            --dropout_ratio 0.4 \
            --graph_pooling mean \
            --JK last \
            --gnn_type gin \
            --dataset ${DATASET} \
            --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_noNorm_filteredPattern_epoch300.pth \
            --filename contextSub_partialCharge_noNorm_subinput_mask_subpooling_separateOutput_${DATASET} \
            --seed ${SEED} \
            --runseed ${SEED} \
            --split scaffold \
            --num_workers 1 \
            --partial_charge \
            --sub_input \
            --sub_level \
            --context \
            --pooling_indicator \
            --separate_output \
            --save_model contextSub/trained_models/finetuned/contextSub/ > nohup_logs/contextSub_partialCharge_noNorm_subinput_mask_subpooling_separateOutput_finetune_seed${SEED}_${DATASET}.out 2>&1
    done
done

# double model mode. Model 1 for molecule embedding, model 2 for substructure embeddings
# for DATASET in bace bbbp clintox sider tox21 toxcast sider hiv
# do
#     for SEED in 1 2 3 4 5
#     do
#         python -m contextSub.finetune \
#             --device ${DEVICE} \
#             --batch_size 32 \
#             --epochs 100 \
#             --lr 0.001 \
#             --lr_scale 1 \
#             --decay 0.0001 \
#             --num_layer 5 \
#             --emb_dim 300 \
#             --dropout_ratio 0.5 \
#             --graph_pooling mean \
#             --JK last \
#             --gnn_type gin \
#             --dataset ${DATASET} \
#             --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_noNorm_filteredPattern_epoch300.pth \
#             --filename contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout05_${DATASET} \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train \
#             --num_workers 1 \
#             --partial_charge \
#             --sub_input \
#             --context \
#             --pooling_indicator \
#             --contextpred \
#             --contextpred_model_file contextSub/trained_models/contextpred.pth > nohup_logs/contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout05_finetune_seed${SEED}_${DATASET}.out 2>&1
#     done
# done


# double model mode. Model 1 for molecule embedding, model 2 for substructure embeddings
# freeze GNN model weights for 20 epochs
# for DATASET in bace bbbp clintox sider tox21 toxcast sider hiv
# do
#     for SEED in 1 2 3 4 5
#     do
#         python -m contextSub.finetune \
#             --device ${DEVICE} \
#             --batch_size 32 \
#             --epochs 100 \
#             --lr 0.001 \
#             --lr_scale 1 \
#             --decay 0.0001 \
#             --num_layer 5 \
#             --emb_dim 300 \
#             --dropout_ratio 0.4 \
#             --graph_pooling mean \
#             --JK last \
#             --gnn_type gin \
#             --dataset ${DATASET} \
#             --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_noNorm_filteredPattern_epoch300.pth \
#             --filename contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04_freeze_${DATASET} \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train \
#             --num_workers 1 \
#             --partial_charge \
#             --sub_input \
#             --context \
#             --pooling_indicator \
#             --contextpred \
#             --contextpred_model_file contextSub/trained_models/contextpred.pth \
#             --freeze 20 > nohup_logs/contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04_freeze_finetune_seed${SEED}_${DATASET}.out 2>&1
#     done
# done

# experiment for lightbbb dataset
# DATASET=lightbbb
# for SEED in 1 2 3 4 5
# do
#     python -m contextSub.finetune \
#         --device ${DEVICE} \
#         --batch_size 16 \
#         --epochs 70 \
#         --lr 0.001 \
#         --lr_scale 1 \
#         --decay 0.0001 \
#         --num_layer 5 \
#         --emb_dim 300 \
#         --dropout_ratio 0.4 \
#         --graph_pooling mean \
#         --JK last \
#         --gnn_type gin \
#         --dataset ${DATASET} \
#         --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_noNorm_filteredPattern_epoch300.pth \
#         --filename contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04_freeze_${DATASET} \
#         --seed ${SEED} \
#         --runseed ${SEED} \
#         --split random_scaffold \
#         --eval_train \
#         --num_workers 1 \
#         --partial_charge \
#         --sub_input \
#         --context \
#         --pooling_indicator \
#         --contextpred \
#         --contextpred_model_file contextSub/trained_models/contextpred.pth \
#         --freeze 20 > nohup_logs/contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04_freeze_finetune_seed${SEED}_${DATASET}.out 2>&1
# done


# muv and hiv datasets
# DATASET=muv
# for SEED in 1 2 3 4 5
# do
#     python -m contextSub.finetune \
#         --device ${DEVICE} \
#         --batch_size 16 \
#         --epochs 120 \
#         --lr 0.0001 \
#         --lr_scale 1 \
#         --decay 0.0001 \
#         --num_layer 5 \
#         --emb_dim 300 \
#         --dropout_ratio 0.4 \
#         --graph_pooling mean \
#         --JK last \
#         --gnn_type gin \
#         --dataset ${DATASET} \
#         --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_noNorm_filteredPattern_epoch300.pth \
#         --filename 2stepfreeze_${DATASET}_contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04 \
#         --seed ${SEED} \
#         --runseed ${SEED} \
#         --split scaffold \
#         --num_workers 1 \
#         --partial_charge \
#         --sub_input \
#         --context \
#         --pooling_indicator \
#         --contextpred \
#         --contextpred_model_file contextSub/trained_models/contextpred.pth \
#         --freeze 20 \
#         --two_step_freeze > nohup_logs/2stepfreeze_${DATASET}_contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04_finetune_seed${SEED}.out 2>&1
# done

# DATASET=hiv
# for SEED in 1 2 3 4 5
# do
#     python -m contextSub.finetune \
#         --device ${DEVICE} \
#         --batch_size 16 \
#         --epochs 120 \
#         --lr 0.0001 \
#         --lr_scale 1 \
#         --decay 0.0001 \
#         --num_layer 5 \
#         --emb_dim 300 \
#         --dropout_ratio 0.4 \
#         --graph_pooling mean \
#         --JK last \
#         --gnn_type gin \
#         --dataset ${DATASET} \
#         --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_noNorm_filteredPattern_epoch300.pth \
#         --filename 2stepfreeze_${DATASET}_contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04 \
#         --seed ${SEED} \
#         --runseed ${SEED} \
#         --split scaffold \
#         --num_workers 1 \
#         --partial_charge \
#         --sub_input \
#         --context \
#         --pooling_indicator \
#         --contextpred \
#         --contextpred_model_file contextSub/trained_models/contextpred.pth \
#         --freeze 20 \
#         --two_step_freeze > nohup_logs/2stepfreeze_${DATASET}_contextSub_partialCharge_noNorm_subinput_mask_subpooling_doubleOutput_dropout04_finetune_seed${SEED}.out 2>&1
# done

# contextSub
# for DATASET in bace bbbp clintox sider tox21 toxcast hiv muv
# for DATASET in tox21 toxcast hiv muv
# do
#     for SEED in 1 2 3 4 5
#     do
#         python -m contextSub.finetune \
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
#             --num_workers 4 > nohup_logs/contextSub_finetune_seed${SEED}_${DATASET}.out 2>&1 &
#     done
#     wait
# done

# contextSub_partialCharge
# for DATASET in bace bbbp clintox sider tox21 toxcast hiv muv
# for DATASET in sider tox21 toxcast hiv muv
# do
#     for SEED in 1 2 3 4 5
#     do
#         python -m contextSub.finetune \
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
#             --input_model_file contextSub/trained_models/contextSub_chembl_partialCharge_no_norm_output.pth \
#             --filename contextSub_partialCharge_${DATASET} \
#             --seed ${SEED} \
#             --runseed ${SEED} \
#             --split scaffold \
#             --eval_train 0 \
#             --num_workers 1 \
#             --partial_charge > nohup_logs/contextSub_partialCharge_finetune_seed${SEED}_${DATASET}.out 2>&1 &
#     done
#     wait
# done