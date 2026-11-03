"""Figures for the external-validation analysis."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import METRIC_LABELS

LEGEND_LABELS = ["Internal", "Vanilla External", "ID-like", "OOD", "Matched External"]


def plot_performance_by_decile(performance_by_deciles, metric_name, out_path=None, show=False):
    """One panel per external site: performance (with 95% CI) across dissimilarity deciles."""
    intervals = [f"{i}-{i + 10}" for i in range(0, 100, 10)]

    plt.figure(figsize=(15, 9))
    for site, perf in enumerate(performance_by_deciles):
        perf = np.asarray(perf)
        mean, low, high = pd.Series(perf[:, 0]), perf[:, 1], perf[:, 2]

        plt.subplot(2, 3, site + 1)
        ax = mean.plot(marker="o", label=str(site + 1))
        ax.fill_between(mean.index, low, high, alpha=0.2)
        plt.xlabel("Dissimilarity Score Decile", fontsize=11)
        plt.ylabel(METRIC_LABELS[metric_name], fontsize=11)
        plt.ylim([-0.005, mean.max() + 0.016])
        plt.xticks(ticks=np.arange(len(intervals)), labels=intervals, rotation=45)
        plt.grid(False)
        plt.title(f"External ID: {site + 1}")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_performance_by_site(all_results, id_sizes, metric_name, out_path=None, show=False,
                             ylim=(0, 0.07)):
    """Grouped bars per external site: vanilla external, ID-like, OOD (with population share)."""
    results = np.array(all_results)  # (n_sites, n_conditions, 3)
    n_rows, n_cols, _ = results.shape
    x = np.arange(n_rows)
    bar_width = 0.23

    plt.figure()
    for i in range(1, n_cols - 1):  # skip 'Internal' (0) and 'Matched External' (last)
        bars = plt.bar(x + (i - 0.5) * bar_width, results[:, i, 0], width=bar_width,
                       label=LEGEND_LABELS[i])
        if i == 1:
            continue
        share = np.array(id_sizes) if i == 2 else 1 - np.array(id_sizes)
        for j, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                     f"{100 * share[j]:.0f}%", ha="center", va="bottom", fontsize=7)

    plt.ylabel(METRIC_LABELS[metric_name], fontsize=11)
    plt.xlabel("External Site ID", fontsize=11)
    plt.xticks(x + bar_width * 1.5, [f"{i + 1}" for i in range(n_rows)])
    plt.ylim(list(ylim))
    plt.legend(framealpha=0.5, fontsize=10, loc="upper left")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()
