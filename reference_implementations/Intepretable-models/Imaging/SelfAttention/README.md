## Method
- This method integrates a trainable self-attention module into the last layer of ResNet architecture, modifying its feature extraction process.

## Resources
- [Self_attention](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/SelfAttention/Self_attention.ipynb) contains an example of Self attention architecture on ResNet, along with demonstrations of heatmap generated on chest xray images.
- [selfattention.yaml](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/SelfAttention/selfattention.yaml) contains configuration for training selfattention model.
- [resnetattention.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/SelfAttention/resnetattention.py) is the model file containing the definition of ResNet Attention.
- [train_nih.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/SelfAttention/train_nih.py) is a training script for ResNetAttention to NIH dataset.
- [utils](https://github.com/VectorInstitute/interpretability-bootcamp/tree/main/reference_implementations/Intepretable-models/Imaging/utils) contains utility scripts for all imaging techniques.

## Running the scripts
- Change file paths and parameters values within the [selfattention.yaml](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/SelfAttention/selfattention.yaml) file.
- Login to a GPU node on the cluster
```bash
srun --pty --mem=100GB -c 20 --gres=gpu:4 --qos=normal -t 8:00:00 /bin/bash
```
- Run script via `torchrun` command
```bash
 torchrun --standalone --nproc-per-node=4 --nnodes=1 ~/interpretability-bootcamp/reference_implementations/Intepretable-models/Imaging/SelfAttention/train_nih.py
```

>[Note]
-< # of GPUs in `srun` command == # nproc-per-node in `torchrun` command
-< Max available is 4 GPUs on 1 node.

## Dataset
- [NIH Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data/data)
    -  This dataset is present on the cluster at the following path `/ssd003/projects/aieng/public/interp_bootcamp/datasets/NIH`. 
    - If you are running scripts on your laptop, you need to download the dataset from the link above.


