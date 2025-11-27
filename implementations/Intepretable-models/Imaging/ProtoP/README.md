## Method
Prototypical Networks creates a classifier by learning a new metric space that tries to generalize to new classes not seen in the training set, given only a small number of examples of each new class.

## Resources
- [prototypical_networks](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/ProtoP/prototypical_networks.ipynb) notebook shows an architechture for prototypical neworks, trains it on MNIST dataset and visualizes explanations based on nearest prototypes.
- [prototype_heatmap.png](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/ProtoP/prototype_heatmap.png) is an image depicting heatmap output of prototypical network for a handwritten digit.

## Datasets
- [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
    - This dataset is present on the Vector cluster at the following path `/ssd003/projects/aieng/public/interp_bootcamp/datasets/MNIST`.
    - If you are running scripts on your laptop, you need to download the dataset from the link above and preprocess it. Else, you can copy the dataset from the cluster to your machine.

>[!NOTE]
>- The prototypical network training code isn't available at the moment. It will be added to the repository after the bootcamp. Please contact the facilitators if this is needed urgently.

>- If you are running the notebook on your laptop, you will need to copy the checkpoint from the cluster using the following command:

```bash
scp user@vremote.vectorinstitute.ai:/ssd003/projects/aieng/public/interp_bootcamp/checkpoints/proto_mnist.pth .
```
