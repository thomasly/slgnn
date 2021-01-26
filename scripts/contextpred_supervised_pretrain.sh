CUDA_VISIBLE_DEVICES=2 \
    python -m contextPred.chem.pretrain_supervised \
        --output_model_file ./contextPred/chem/trained_models/chemblFiltered_and_supervise_pretrained_model_with_contextPred \
        --input_model_file ./contextPred/chem/trained_models/chemblFiltered_pretrained_model_with_contextPred > ./logs/contextPred_supervised_pretrain_output.log 2>&1 &