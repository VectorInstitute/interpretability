## Method
NBM is a new model archiecture where it tries to mitigate the limitation of GAMs by learning a single Neural network to learn the basis of shape functions.

## Resources
- [NBM-jupyter](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NBM/NBM-jupyter.ipynb) notebook contains training of NBM method on California housing dataset to predict house prices.
- [Basis_Functions](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NBM/Basis_Functions.ipynb) notebook shows basis function plots for Polynomial curves and fourier transform.
- [nbm_regression.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NBM/nbm_regression.py)
- [model](https://github.com/VectorInstitute/interpretability-bootcamp/tree/main/reference_implementations/Intepretable-models/Tabular/NBM/model) folder contains NBM model file.
- [utils](https://github.com/VectorInstitute/interpretability-bootcamp/tree/main/reference_implementations/Intepretable-models/Tabular/NBM/utils) contains visualization functions.

## Running the scripts
- Change file paths and parameters values within the [nam.yaml](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/nam.yaml) file.
- Login to a GPU node on the cluster
```bash
srun --pty --mem=5GB -c 2 --gres=gpu:1 --qos=normal -t 8:00:00 /bin/bash
```
- Source the virtual environment
```bash
source /ssd003/projects/aieng/public/interp_bootcamp/venv/bin/activate
```
- Run script via `python` command
```bash
python nbm_regression.py
```

## Datasets
- [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)