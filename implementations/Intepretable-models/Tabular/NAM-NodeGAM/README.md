## Method
NAM is an extension of GAMs where the functions are a linear combination of neural networks. They are more popular, scalable and can extend to multi-task learning.

NodeGAM is another neural-based GAM that improves from NAM in 2 ways: (1) it uses differentiable trees which allow quick, non-linear changes, (2) it uses attention to do feature selection that can scale to a large number of features, and (3) model pairwise feature interactions called NodeGA2M.

## Resources
- [nam-jupyter](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/nam-jupyter.ipynb) notebook contains examples of classification and regression using the DNAMITE library.
- [nodegam](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/nodegam.ipynb) notebook contains a classification example of credit default using NodeGAM method.
- [nam](https://github.com/VectorInstitute/interpretability-bootcamp/tree/main/reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/nam) folder contains utility scripts for NAM model such as visualization, model definition, metrics etc.
- [train_nam.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/train_nam.py)
- [compare_models.py](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/compare_models.py) script compares EBM and XGB model on readmission classification and plots feature functions for both.
- [nam.yaml](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/nam.yaml) contains parameters for loading data, training NAM and storing plots.
- [nam.pkl](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/nam.pkl) is a pretrained pickle file for trained NAM classifier on diabetic data.
- [scalar.pkl](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Intepretable-models/Tabular/NAM-NodeGAM/scaler.pkl) is a pretrained pickle file using NodeGAM classifier on creadit card default dataset.

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
python train_nam.py
```

## Datasets
- [Bank dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [Gas turbine dataset](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)
- [Default of Credit Card clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## References
@inproceedings{chang2021node,
  title={NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning},
  author={Chang, Chun-Hao and Caruana, Rich and Goldenberg, Anna},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

The official full implementation of NODE-Gam is available from [Github](https://github.com/zzzace2000/nodegam)
