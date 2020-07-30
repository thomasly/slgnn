conda install -y pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch

CUDA=cpu
TORCH=1.4.0

pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric

conda install -y -c rdkit rdkit
conda install -y pyyaml
conda install -y matplotlib
conda install -y -c thomasly chemreader
pip install -U Sphinx
