"""Performance metrics with bootstrap confidence intervals."""
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

METRICS = {"auc": roc_auc_score, "brier": brier_score_loss}
METRIC_LABELS = {"auc": "AUC", "brier": "Brier Score"}


def bootstrap(metric, y_true, y_prob, sample_weight=None, num_bootstrap=5, ci=95):
    """Return [point estimate, lower, upper] for `metric` using a percentile bootstrap.

    NOTE: `sample_weight` is passed through unchanged in every bootstrap replicate
    (it is not resampled together with the indices). This reproduces the behaviour of
    the original analysis; see README for details.
    """
    overall = metric(y_true, y_prob, sample_weight=sample_weight)

    boot_samples = []
    for _ in range(num_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        boot_samples.append(metric(y_true[indices], y_prob[indices], sample_weight=sample_weight))

    lower = np.percentile(boot_samples, (100 - ci) / 2)
    upper = np.percentile(boot_samples, 100 - (100 - ci) / 2)
    return [overall, lower, upper]
