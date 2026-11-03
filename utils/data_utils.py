"""Data splitting, normalisation and column-filtering helpers."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(features, labels, train_size=0.7, random_state=0):
    """Stratified train/test split. Returns dicts X, y with keys 'train' and 'test'."""
    X, y = {}, {}
    X["train"], X["test"], y["train"], y["test"] = train_test_split(
        features, labels, stratify=labels, train_size=train_size, random_state=random_state
    )
    return X, y


def normalization(X, y):
    """Fit a StandardScaler on X['train'] and apply it to every split.

    Returns
    -------
    X_normalized, X_unnormalized, y, preprocess
    """
    preprocess = StandardScaler().fit(X["train"])
    X_normalized = {k: preprocess.transform(v) for k, v in X.items()}
    X_unnormalized = dict(X)
    y = dict(y)
    return X_normalized, X_unnormalized, y, preprocess


def columns_mostly_equal(arr, threshold=0.97):
    """Indices of columns where a single value makes up more than `threshold` of the rows."""
    n_rows, n_cols = arr.shape
    indices = []
    for col in range(n_cols):
        _, counts = np.unique(arr[:, col], return_counts=True)
        if counts.max() / n_rows > threshold:
            indices.append(col)
    return indices
