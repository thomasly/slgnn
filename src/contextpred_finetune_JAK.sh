#!/bin/bash

split=scaffold
# split=random

for jak in jak1 jak2 jak3
do
    CUDA_VISIBLE_DEVICES=$1 python -m contextPred.chem.finetune \
        --input_model_file ./contextPred/chem/trained_models/supervise_pretrained_model.pth \
        --dataset ${jak} \
        --filename ./contextPred/chem/finetune_logs/${split}/${jak} \
        --split ${split}
    wait
done