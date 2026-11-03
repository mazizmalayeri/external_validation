#!/usr/bin/env python
"""Run the leave-one-site-out external validation and produce the figures.

Example
-------
    python scripts/run_external_validation.py --data TAVI_data.csv --metric brier --out results
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.metrics import METRICS
from utils.pipeline import load_data, run_external_validation
from utils.plotting import plot_performance_by_decile, plot_performance_by_site


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="TAVI_data.csv", help="Path to the TAVI CSV file.")
    p.add_argument("--metric", choices=sorted(METRICS), default="brier")
    p.add_argument("--out", default="results", help="Output directory for figures / results.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-sites", type=int, default=5, help="Number of largest sites used as external folds.")
    p.add_argument("--show", action="store_true", help="Display figures interactively.")
    args = p.parse_args()

    #setting seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    #loading data
    features, labels, hospitals = load_data(args.data)
    print("Outcome rate:", (labels == 1).mean())

    #running pipeline
    results = run_external_validation(features, labels, hospitals, METRICS[args.metric],
                                  n_sites=args.n_sites)

    #plotting the results
    plot_performance_by_decile(results["performance_by_deciles"], args.metric,
                               os.path.join(args.out, "performance_by_decile.png"), args.show)
    plot_performance_by_site(results["all_results"], results["id_sizes"], args.metric,
                             os.path.join(args.out, "performance_by_site.png"), args.show)
    print(f"Figures written to {args.out}/")


if __name__ == "__main__":
    main()
