## Method
Local Interpretable Model Agnostic Explanations (LIME) method attempts to explain individual predictions by fitting a surrogate interpretable model that gives good approximate predictions of the underlying black box model locally.

## Resources
- [LIME-jupyter.ipynb](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Post-hoc/LIME/LIME-jupyter.ipynb) trains a Gradient Boosting Regressor model to predict turbine energy yield, and shows local and global explanations provided by LIME.
- [finalized_model.sav](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Post-hoc/LIME/finalized_model.sav) is a pretrained model.
- [lime_oi.png](https://github.com/VectorInstitute/interpretability-bootcamp/blob/main/reference_implementations/Post-hoc/LIME/lime_oi.png) is a visualization of local explanation of features of the dataset.

## Datasets
- [Gas sensor turbine data](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)