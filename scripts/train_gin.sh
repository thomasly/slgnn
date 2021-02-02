# for conf in config_GIN_CPAN_JAK1.yml config_GIN_CPAN_JAK2.yml config_GIN_CPAN_JAK3.yml
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m slgnn.training.train_gin \
#     -c model_configs/$conf
# done

# for conf in ToxCast.yml MUV.yml Sider.yml HIV.yml
# for conf in BBBP.yml
# for dataset in BACE BBBP ClinTox HIV
for dataset in BACE BBBP ClinTox Sider HIV
do
    CUDA_VISIBLE_DEVICES=$1 nohup \
        python -m slgnn.training.train_gin \
        -c model_configs/${dataset}.yml > logs/train_ginfe_with_${dataset}_fragment.out 2>&1 &
done

# for conf in config_GIN_Amu.yml config_GIN_Ellinger.yml config_GIN_Mpro.yml
# do
#     CUDA_VISIBLE_DEVICES=$1 \
#     python -m slgnn.training.train_gin \
#     -c model_configs/$conf
# done

# CUDA_VISIBLE_DEVICES=$1 \
# python -m slgnn.training.train_gin \
# -c model_configs/config_GIN_CPAN_JAK$2.yml &

# for dataset in BACE BBBP ClinTox HIV
# do
#     echo $dataset
#     CUDA_VISIBEL_DEVICES=$1 python -m slgnn.training.train_gin -c model_configs/config_GIN_$dataset.yml
# done