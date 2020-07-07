# for conf in config_GIN_CPAN_JAK1.yml config_GIN_CPAN_JAK2.yml config_GIN_CPAN_JAK3.yml
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m slgnn.training.train_gin \
#     -c model_configs/$conf
# done

for conf in config_GIN_BACE.yml config_GIN_BBBP.yml config_GIN_ClinTox.yml config_GIN_HIV.yml config_GIN_Sider.yml
do
    CUDA_VISIBLE_DEVICES=$1 \
    python -m slgnn.training.train_gin \
    -c model_configs/$conf
done

# CUDA_VISIBLE_DEVICES=$1 \
# python -m slgnn.training.train_gin \
# -c model_configs/config_GIN_CPAN_JAK$2.yml &

# for dataset in BACE BBBP ClinTox HIV
# do
#     echo $dataset
#     CUDA_VISIBEL_DEVICES=$1 python -m slgnn.training.train_gin -c model_configs/config_GIN_$dataset.yml
# done