# GOSDT (Generalized Optimal Sparse Decision Tree)

GOSDT is a type of decision tree model designed to find the best possible decision tree while keeping it small and interpretable. It works by efficiently searching through all possible decision trees and selecting the one that gives the best balance between accuracy and simplicity (i.e., the number of splits in the tree).

This is a demonstration of GOSDT.

## Setup

If you are running the notebook on your laptop:
1. Install [graphviz](https://www.graphviz.org/download/) to render the images.
2. In the file `.venv/lib/python3.10/site-packages/gosdt/_threshold_guessing.py` (or path to your environment), replace 

```python
from sklearn.base import BaseEstimator, TransformerMixin, check_X_y 
```

to 

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y
```

The  gosdt package tries to import `check_X_y from sklearn.base`, but in newer versions of scikit-learn, 
`check_X_y` is located in `sklearn.utils.validation`. We prefer this solution so as not the break the dependency on scikit-learn version for other examples.
