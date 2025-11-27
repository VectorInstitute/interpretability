#!/usr/bin/env python
# coding: utf-8

"""
Generalized Additive Model (GAM) for Regression Tasks on the Insurance Dataset

This script:
- Loads the insurance dataset.
- Preprocesses the data:
   • For numerical features: scales them using StandardScaler and saves the original values.
   • For categorical features: applies LabelEncoder and stores the mapping for plotting.
- Trains a LinearGAM using a sum-of-terms specification (smooth for numericals, factor for categoricals).
- Plots partial dependence (shape functions) for all features,
  converting the x‑axes back to the original scale for numericals and applying original category labels for categoricals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pygam import LinearGAM, s, f

np.random.seed(42)


# -----------------------------
# 1. Load and Preprocess Data
# -----------------------------
def load_data():
    # Adjust the path as needed
    return pd.read_csv("../datasets/insurance/insurance.csv")


def preprocess_data(data):
    # Make copies to preserve original numerical values
    data_processed = data.copy()
    data_numerical_orig = data_processed.copy()

    # Identify categorical and numerical columns.
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = data.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    # Remove the target variable 'charges' from numerical_columns if present.
    if "charges" in numerical_columns:
        numerical_columns.remove("charges")

    # For categorical features, use LabelEncoder and store mapping.
    cat_mapping = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data_processed[col] = le.fit_transform(data_processed[col])
        # Save the original class labels (in order of encoding)
        cat_mapping[col] = list(le.classes_)

    # For numerical features, store original values.
    data_numerical_orig = data_numerical_orig[numerical_columns]

    # Scale numerical features
    scaler = StandardScaler()
    data_processed[numerical_columns] = scaler.fit_transform(
        data_processed[numerical_columns]
    )

    return data_processed, scaler, cat_mapping, data_numerical_orig, numerical_columns


# Load and preprocess the data
df = load_data()
(
    df_processed,
    scaler,
    cat_mapping,
    data_numerical_orig,
    numerical_columns,
) = preprocess_data(df)
print("Processed data head:")
print(df_processed.head())

# -----------------------------
# 2. Split Data for Training
# -----------------------------
# Assume target is 'charges'
X = df_processed.drop("charges", axis=1)
y = df_processed["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Build the GAM Model
# -----------------------------
# Build model specification: smooth for numerical, factor for categorical.
terms = None
for i, col in enumerate(X_train.columns):
    if col in numerical_columns:
        new_term = s(i)
    else:
        new_term = f(i)
    terms = new_term if terms is None else terms + new_term

gam = LinearGAM(terms, n_splines=5, lam=0.01).fit(X_train.values, y_train.values)
print(gam.summary())

# -----------------------------
# 4. Evaluate the Model
# -----------------------------
y_pred = gam.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# -----------------------------
# 5. Plot Partial Dependence for All Features
# -----------------------------
feature_names = X_train.columns
n_features = len(feature_names)

# Create a mapping for numerical features: feature -> (mean, std)
num_stats = {}
for col in numerical_columns:
    j = numerical_columns.index(col)
    mean = scaler.mean_[j]
    std = np.sqrt(scaler.var_[j])
    num_stats[col] = (mean, std)

# Baseline row: use the mean of the scaled X_train for all features.
baseline_row = X_train.values.mean(axis=0)

# Determine subplot grid dimensions
ncols = math.ceil(np.sqrt(n_features))
nrows = math.ceil(n_features / ncols)

fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axs = axs.ravel()

# For each feature, create a full X matrix based on baseline values.
for idx in range(n_features):
    feature = feature_names[idx]
    term_idx = int(idx)

    if feature in numerical_columns:
        grid_points = 100
        # Create baseline matrix with grid_points rows.
        X_baseline = np.tile(baseline_row, (grid_points, 1))
        # Get original range for this numerical feature
        orig_vals = data_numerical_orig[feature]
        orig_min, orig_max = orig_vals.min(), orig_vals.max()
        grid_orig = np.linspace(orig_min, orig_max, grid_points)
        # Convert grid to scaled values using (mean, std)
        mean, std = num_stats[feature]
        grid_scaled = (grid_orig - mean) / std
        X_baseline[:, term_idx] = grid_scaled
        pdep, confi = gam.partial_dependence(term=term_idx, X=X_baseline, width=0.95)
        axs[idx].plot(grid_orig, pdep)
        axs[idx].fill_between(grid_orig, confi[:, 0], confi[:, 1], color="r", alpha=0.3)
        axs[idx].set_xlabel(feature)
    else:
        # For categorical features
        levels = cat_mapping[feature]
        grid_points = len(levels)
        X_baseline = np.tile(baseline_row, (grid_points, 1))
        grid_factor = np.arange(grid_points)
        X_baseline[:, term_idx] = grid_factor
        pdep, confi = gam.partial_dependence(term=term_idx, X=X_baseline, width=0.95)
        axs[idx].plot(grid_factor, pdep, marker="o")
        axs[idx].fill_between(
            grid_factor, confi[:, 0], confi[:, 1], color="r", alpha=0.3
        )
        axs[idx].set_xticks(grid_factor)
        axs[idx].set_xticklabels(levels)
        axs[idx].set_xlabel(feature)

    axs[idx].set_ylabel("Target")
    axs[idx].set_title(f"Shape Function of {feature}")

# Hide any extra subplots if n_features is not a perfect grid.
for k in range(n_features, len(axs)):
    fig.delaxes(axs[k])

plt.tight_layout()
plt.savefig("GAM_regression.png")
