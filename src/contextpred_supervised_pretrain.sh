CUDA_VISIBLE_DEVICES=$1 \
    python -m contextPred.chem.pretrain_supervised \
        --output_model_file ./contextPred/chem/trained_models/supervise_pretrained_model \
        --input_model_file ./contextPred/chem/trained_models/context_pretrained_model_2