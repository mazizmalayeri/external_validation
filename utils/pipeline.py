"""Leave-one-site-out external validation with dissimilarity (OOD) analysis."""
import numpy as np
import pandas as pd
import torch
from miceforest import ImputationKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .autoencoder import AE
from .data_utils import columns_mostly_equal, normalization, split_data
from .metrics import bootstrap

AE_KWARGS = dict(hidden_sizes=[20, 20], latent_dim=30, device="cpu", lr=1e-3)
AE_BATCH_SIZE = 64
AE_EPOCHS = 10


def load_data(path, site_col="centrum", label_col="30-day-mort"):
    """Read the CSV and split into (features, labels, sites)."""
    df = pd.read_csv(path)
    features = df.drop([site_col, label_col], axis=1)
    return features, df[label_col], df[site_col]


def _impute(kernel, array, columns):
    """Impute new data with a fitted MICE kernel and drop rows that are still NaN."""
    imputed = kernel.impute_new_data(new_data=pd.DataFrame(array, columns=columns), random_state=0)
    imputed = imputed.complete_data(dataset=0).to_numpy(dtype="float32")
    keep = ~np.isnan(imputed).any(axis=1)
    return imputed, keep


def _fit_ae(X, input_size):
    ae = AE(input_size=input_size, **AE_KWARGS)
    ae.train(torch.tensor(X), AE_BATCH_SIZE, AE_EPOCHS)
    return ae


def run_external_validation(features, labels, hospitals, performance_metric,
                            n_sites=5, n_decile_bootstrap=50):
    """Hold out each of the `n_sites` largest sites in turn as the external set.

    For every held-out site this: imputes (MICE), checks dataset separability (Debray c-statistic),
    fits a logistic regression on the remaining sites, evaluates it internally and externally,
    scores external patients with an autoencoder dissimilarity score, and splits the external
    performance into ID-like / OOD / weight-matched / per-decile estimates.

    Returns
    -------
    dict with keys:
        all_results            list (per site) of [internal, external, ID-like, OOD, matched],
                               each entry being [estimate, ci_low, ci_high]
        id_sizes               fraction of external patients that are ID-like, per site
        ess_matching           normalised effective sample size of the matching weights, per site
        performance_by_deciles list (per site) of arrays of shape (10, 3): estimate/low/high
        separability_auc       Debray AUC between development and external data, per site
    """
    unique_hospitals = hospitals.value_counts()
    print('Each hospital patients count:\n', unique_hospitals, unique_hospitals.sum())

    all_results, id_sizes, ess_matching = [], [], []
    performance_by_deciles, separability_auc = [], []

    for external_id in range(n_sites):
        internal_ids = [i for i in range(n_sites) if i != external_id]
        mask_internal = np.zeros(hospitals.shape, dtype=bool)
        for i in internal_ids:
            mask_internal |= (hospitals == unique_hospitals.index[i]).to_numpy()
        mask_external = (hospitals == unique_hospitals.index[external_id]).to_numpy()

        print("\nInternal Hospitals ID:", "".join(map(str, internal_ids)), "#instances:", mask_internal.sum())
        print("External Hospital ID:", external_id, "#instances:", mask_external.sum())

        # --- development / external data ---------------------------------------------------
        dev_X = features[mask_internal].to_numpy(dtype="float32")
        dev_y = labels[mask_internal].to_numpy().reshape(-1)
        ext_X = features[mask_external].to_numpy(dtype="float32")
        ext_y = labels[mask_external].to_numpy().reshape(-1)

        # remove near-constant columns (decided on the development data only)
        drop_idx = columns_mostly_equal(dev_X, threshold=0.99)
        dev_X = np.delete(dev_X, drop_idx, axis=1)
        ext_X = np.delete(ext_X, drop_idx, axis=1)
        print("Removed columns:", drop_idx)

        dev_X, dev_y = split_data(dev_X, dev_y)

        # --- imputation (MICE fitted on the development training split only) ---------------
        columns = np.delete(features.columns.values, drop_idx)
        kernel = ImputationKernel(data=pd.DataFrame(dev_X["train"], columns=columns), random_state=0)
        kernel.mice(50)
        dev_X["train"] = kernel.complete_data(dataset=0).to_numpy(dtype="float32")

        dev_X["test"], keep = _impute(kernel, dev_X["test"], columns)
        dev_X["test"], dev_y["test"] = dev_X["test"][keep], dev_y["test"][keep]

        ext_X, keep = _impute(kernel, ext_X, columns)
        ext_X, ext_y = ext_X[keep], ext_y[keep]

        # --- Debray separability: can a classifier tell development from external? ---------
        X_all = np.concatenate((dev_X["train"], dev_X["test"], ext_X), axis=0)
        y_all = np.concatenate((np.zeros(len(dev_X["train"]) + len(dev_X["test"])),
                                np.ones(len(ext_X))))
        X_all = StandardScaler().fit(X_all).transform(X_all)
        sep_model = LogisticRegression(random_state=0, max_iter=500).fit(X_all, y_all)
        auc_diff = roc_auc_score(y_all, sep_model.predict_proba(X_all)[:, 1])
        print("AUC of difference between datasets:", auc_diff)
        separability_auc.append(auc_diff)

        # --- prediction model ---------------------------------------------------------------
        dev_X, _, dev_y, scaler = normalization(dev_X, dev_y)
        model = LogisticRegression(random_state=0, max_iter=1000).fit(dev_X["train"], dev_y["train"])
        
        train_scores = model.predict_proba(dev_X["train"])
        print('Checking performane ...')
        print("train validation:", performance_metric(dev_y["train"], train_scores[:, 1]))

        internal_scores = model.predict_proba(dev_X["test"])
        internal_validation = bootstrap(performance_metric, dev_y["test"], internal_scores[:, 1])
        print("internal validation:", internal_validation[0])

        ext_X_scaled = scaler.transform(ext_X)
        ext_scores = model.predict_proba(ext_X_scaled)[:, 1]
        external_validation = bootstrap(performance_metric, ext_y, ext_scores)
        print("external validation:", external_validation[0])

        # --- dissimilarity (OOD) score: AE trained on the development test split -----------
        ae = _fit_ae(dev_X["test"], dev_X["test"].shape[1])
        re_dev = ae.dissimilarity_score(dev_X["test"])
        re_ext = ae.dissimilarity_score(ext_X_scaled)

        # ID-like vs OOD external patients (threshold = 90th percentile of dev dissimilarity)
        ood_threshold = np.quantile(re_dev, 0.9)
        is_ood, is_id = re_ext > ood_threshold, re_ext < ood_threshold
        print("Num OOD and ID:", is_ood.sum(), is_id.sum())
        ood_performance = bootstrap(performance_metric, ext_y[is_ood], ext_scores[is_ood])
        id_performance = bootstrap(performance_metric, ext_y[is_id], ext_scores[is_id])
        id_size = is_id.mean()
        print("OOD-like and ID-like:", ood_performance[0], id_performance[0], id_size)

        # --- matching: sanity check + weights ------------------------------------------------
        scaler_ext = StandardScaler().fit(ext_X)
        ae_ext = _fit_ae(scaler_ext.transform(ext_X), ext_X.shape[1])
        re_ext_prime = ae_ext.dissimilarity_score(scaler_ext.transform(ext_X))
        re_dev_prime = ae_ext.dissimilarity_score(scaler_ext.transform(dev_X["train"]))  # note: see README
        print("Check if development is subset of external:", re_dev_prime.mean(), re_ext_prime.mean())

        # Weights: external patients that look MORE like development data get larger weight.
        weights = np.clip(re_ext, np.percentile(re_ext, 2.5), np.percentile(re_ext, 97.5))
        weights = weights.max() - weights
        ess = (weights.sum() ** 2) / (weights ** 2).sum() / len(weights)
        matched_validation = bootstrap(performance_metric, ext_y, ext_scores, sample_weight=weights)
        print("Matching:", matched_validation[0])

        all_results.append([internal_validation, external_validation, id_performance,
                            ood_performance, matched_validation])
        id_sizes.append(float(id_size))
        ess_matching.append(float(ess))

        # --- performance by dissimilarity decile -------------------------------------------
        scores_df = pd.DataFrame({"prob": ext_scores, "label": ext_y, "distance": re_ext})
        scores_df["decile"] = pd.qcut(scores_df["distance"], 10, labels=False)
        decile_results = [
            bootstrap(performance_metric, g["label"].to_numpy(), g["prob"].to_numpy(),
                      num_bootstrap=n_decile_bootstrap, ci=95)
            for _, g in scores_df.groupby("decile")
        ]
        performance_by_deciles.append(np.array(decile_results))

    return dict(all_results=all_results, id_sizes=id_sizes, ess_matching=ess_matching,
                performance_by_deciles=performance_by_deciles, separability_auc=separability_auc)
