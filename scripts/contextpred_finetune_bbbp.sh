#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
split=random
DATA=bbbp

python -m contextPred.chem.finetune \
    --input_model_file ./contextPred/chem/trained_models/chemblFiltered_and_supervise_pretrained_model_with_contextPred.pth \
    --dataset ${DATA} \
    --filename ./contextPred/chem/finetune_logs/split_${split}/dataset_${DATA}/seed_${seed} \
    --save_model_to ./contextPred/chem/trained_models/chemblFiltered_and_supervise_pretrained_${DATA}_finetuned_model_with_contextPred.pth \
    --split ${split} > ./logs/contextPred_finetune_${DATA}.out 2>&1 &
