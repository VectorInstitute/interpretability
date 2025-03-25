## Method
A Counterfactual explanation can be used to explain predictions of individual instances by describing a smallest change to feature values that changes the prediction to a predefined output.

## Resources
- [Counter_imaging](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Post-hoc/counterfactual/Counter_imaging.ipynb) notebook shows an example of counterfactual explanation on chest xray images using perturbation-based explainability technique via CAPTUM library.
- [Counter_tabular](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Post-hoc/counterfactual/Counter_tabular.ipynb) notebook trains an EBM on the diabetic dataset for classification and uses [dice-ml](https://interpret.ml/DiCE/notebooks/DiCE_getting_started.html) for training a counterfactual explanation.

## Datasets
- [Diabetic dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- [NIH Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data/data)
    -  This dataset is present on the cluster at the following path `/ssd003/projects/aieng/public/interp_bootcamp/datasets/NIH`. 
    - If you are running scripts on your laptop, you need to download the dataset from the link above.