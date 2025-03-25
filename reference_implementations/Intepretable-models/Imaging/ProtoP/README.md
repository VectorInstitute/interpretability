## Method
Prototypical Networks creates a classifier by learning a new metric space that tries to generalize to new classes not seen in the training set, given only a small number of examples of each new class.

## Resources
[prototypical_networks](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/ProtoP/prototypical_networks.ipynb) notebook shows an architechture for prototypical neworks, trains it on MNIST dataset and visualizes explanations based on nearest prototypes.
[prototype_heatmap.png](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Imaging/ProtoP/prototype_heatmap.png) is an image depicting heatmap output of prototypical network for a handwritten digit.

## Datasets
- [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
    - This dataset is present on the Vector cluster at the following path `/ssd003/projects/aieng/public/interp_bootcamp/datasets/MNIST`.
    - If you are running scripts on your laptop, you need to download the dataset from the link above.

>[!NOTE]
>- The prototypical network training code isn't available at the moment. Please contact the facilitators if this is needed urgently.