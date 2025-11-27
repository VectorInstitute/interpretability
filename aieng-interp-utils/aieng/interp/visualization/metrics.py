"""Visualization utilities for model evaluation metrics."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    display_labels: list[str] | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
) -> ConfusionMatrixDisplay:
    """
    Plot confusion matrix for classification results.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    display_labels : list of str, optional
        Labels to display on the confusion matrix axes.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    cmap : str, default="Blues"
        Colormap for the confusion matrix.
    title : str, default="Confusion Matrix"
        Title for the plot.

    Returns
    -------
    ConfusionMatrixDisplay
        The confusion matrix display object.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=cmap, values_format="d", ax=ax)
    if ax is None:
        plt.title(title)
    else:
        ax.set_title(title)
    return disp


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "ROC Curve",
    **kwargs,
) -> RocCurveDisplay:
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Target scores (probability estimates or decision function).
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    title : str, default="ROC Curve"
        Title for the plot.
    **kwargs
        Additional keyword arguments passed to RocCurveDisplay.from_predictions.

    Returns
    -------
    RocCurveDisplay
        The ROC curve display object.
    """
    disp = RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, **kwargs)
    if ax is None:
        plt.title(title)
    else:
        ax.set_title(title)
    return disp


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ax: plt.Axes | None = None,
    title: str = "Precision-Recall Curve",
    **kwargs,
) -> PrecisionRecallDisplay:
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Target scores (probability estimates or decision function).
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    title : str, default="Precision-Recall Curve"
        Title for the plot.
    **kwargs
        Additional keyword arguments passed to PrecisionRecallDisplay.from_predictions.

    Returns
    -------
    PrecisionRecallDisplay
        The precision-recall curve display object.
    """
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, **kwargs)
    if ax is None:
        plt.title(title)
    else:
        ax.set_title(title)
    return disp
