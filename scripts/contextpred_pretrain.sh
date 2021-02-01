# CUDA_VISIBLE_DEVICES=2 \
#     nohup python -m contextPred.chem.pretrain_contextpred \
#         --output_model_file ./contextPred/chem/trained_models/chemblFiltered_pretrained_model_with_contextPred \
#         --dataset contextPred/chem/dataset/chembl_filtered > ./logs/contextPred_chemblFiltered_pretrain_output.log 2>&1 &


CUDA_VISIBLE_DEVICES=$1 \
    nohup python -m contextPred.chem.pretrain_contextpred \
        --output_model_file ./contextPred/chem/trained_models/zincStandard_pretrained_model_with_contextPred \
        --dataset contextPred/chem/dataset/zinc_standard_agent > ./logs/contextPred_zincStandard_pretrain_output.log 2>&1 &
