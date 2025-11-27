# AI Engineering Interpretability Utils

Helper modules for AI Engineering Interpretability Bootcamp implementations.

This package provides reusable utilities for:
- Loading and preprocessing common datasets (Adult, Diabetes, Bank Marketing, Gas Turbine)
- Model training and evaluation helpers
- Visualization utilities for interpretability

## Installation

```bash
pip install aieng-interp-utils
```

## Usage

```python
from aieng.interp.data import load_adult_dataset, preprocess_adult_data
from aieng.interp.models import train_pytorch_model, get_device
from aieng.interp.visualization import plot_confusion_matrix, plot_roc_curve

# Load and preprocess data
df = load_adult_dataset("path/to/data")
X_train, X_test, y_train, y_test = preprocess_adult_data(df)

# Train model
device = get_device()
model = train_pytorch_model(model, X_train, y_train, device=device)

# Visualize results
plot_roc_curve(y_test, y_pred_proba)
plot_confusion_matrix(y_test, y_pred)
```

## License

MIT
