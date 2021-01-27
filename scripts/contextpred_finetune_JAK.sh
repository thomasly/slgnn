#!/bin/bash

split=random
# split=random

# for seed in 42 35 1829 1728 3728
# do
#     for jak in jak1 jak2 jak3
#     do
#         CUDA_VISIBLE_DEVICES=$1 python -m contextPred.chem.finetune \
#             --input_model_file ./contextPred/chem/trained_models/supervise_pretrained_model.pth \
#             --dataset ${jak} \
#             --filename ./contextPred/chem/finetune_logs/split_${split}/dataset_${jak}/seed_${seed} \
#             --split ${split}
#             --seed ${seed}
#         wait
#     done
# done

for jak in jak1 jak2 jak3
do
    CUDA_VISIBLE_DEVICES=$1 nohup python -m contextPred.chem.finetune \
        --input_model_file ./contextPred/chem/trained_models/chemblFiltered_and_supervise_pretrained_model_with_contextPred.pth \
        --dataset ${jak} \
        --filename ./contextPred/chem/finetune_logs/split_${split}/dataset_${jak}/seed_${seed} \
        --split ${split} > ./logs/contextPred_finetune_${jak}.out 2>&1 &
done
