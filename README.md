# Similarity-aware external validation of a TAVI 30-day mortality model

Code accompanying:

> Azizmalayeri M, Abu-Hanna A, Houterman S, Vis MM, Cinà G.
> **Rethinking external validation for the target population: Capturing patient-level similarity with a generative model.**
> arXiv:2605.11284 (2026). <https://arxiv.org/abs/2605.11284>

External validation is usually summarised by a single number, which mixes two things: how good the model
really is, and how different the external patients are from the development patients. This repository
implements the paper's framework, which separates the two by scoring **every external patient's similarity
to the development data** with an autoencoder and then evaluating performance in subgroups of the external
data. It is demonstrated on 30-day mortality after transcatheter aortic valve implantation (TAVI) using data
from the Netherlands Heart Registration (NHR).

## The framework in brief

The right analysis depends on where the model will be deployed.

| Deployment | Question | What is computed | Code output |
|---|---|---|---|
| Development population | **Q1** – Does the model keep its internal performance on external data *matched* to the development distribution? | Performance on external data reweighted to the development distribution, compared with internal validation | `matched` entry of `all_results` |
| Development population | **Q2** – How robust is the model to OOD data? | Performance across deciles of the similarity score | `performance_by_decile.png` |
| External population | **Q3** – How does performance differ between ID-like and OOD patients? | Performance in ID-like vs OOD subgroups of the external data | `performance_by_site.png` |

**Similarity score.** An autoencoder is trained on development data only; a patient's reconstruction error is
their dissimilarity to the development population (higher = less similar). Unlike a logistic-regression
"membership model", it needs no external data for training and can in principle be reused across external
datasets (see *Notes*). The AE is trained on the development **test** split, kept separate from the data used to
fit the prediction model, to avoid biasing the evaluation.

**ID-like vs OOD.** The threshold is the 90th percentile of the similarity scores of the development data
(paper, Appendix B). External patients above it are OOD; the rest are ID-like.

**Metric.** The Brier score is the primary metric because it is additive over disjoint subsets: the overall
Brier score equals the size-weighted average of the ID-like and OOD Brier scores (paper, Appendix C), so the
conventional external result decomposes exactly into the two subgroup results. AUC is also supported
(`--metric auc`), but it is not additive.

## Case-study setup (as in the paper)

- The 5 largest hospitals are used in a leave-one-centre-out scheme: each hospital in turn is the external site and the other 4 form the development set. Hospital numbers in the figures follow centre size.
- Development data are split into train / test; a logistic regression is fitted on train and evaluated on test (internal validation) and on the held-out hospital (external validation).
- Missing values are imputed with MICE (`miceforest`), fitted on the development training data only to avoid leakage.
- Near-constant columns (>99% identical in the development data) are removed.
- Confidence intervals are percentile bootstrap intervals.
- A Debray-style separability AUC (logistic regression distinguishing development from external patients) is printed per fold as a population-level pre-check. The paper notes that population-level differences are a useful check on whether the matching and shift analyses are meaningful at all.

## Installation and usage

```bash
pip install -r requirements.txt
python scripts/run_external_validation.py --data TAVI_data.csv --metric brier --out results
```

Options: `--metric {auc,brier}` (default `brier`), `--seed`, `--n-sites`, `--show`.

Outputs in `results/`:
- `performance_by_decile.png` – performance by similarity decile, one panel per external hospital (Q2; paper Fig. 4)
- `performance_by_site.png` – vanilla external vs ID-like vs OOD performance, with the share of patients in each subgroup (Q3; paper Fig. 5)

The matched-external estimates (Q1; paper Fig. 3) are computed and printed for each fold, but this repository
does not yet include a plotting function for that figure.

## Data

The patient-level NHR data are **not** included and cannot be redistributed here (`*.csv` is git-ignored).
The script expects a CSV with a hospital column `centrum`, a binary outcome column `30-day-mort`, and numeric
predictor columns. Predictor selection and pre-processing in the paper follow Yordanov et al.,
*J Clin Epidemiol* 2023;157:13–21.

## Repository layout

```
tavi/data_utils.py    split / scaling / constant-column helpers
tavi/metrics.py       bootstrap CIs, metric registry (AUC, Brier)
tavi/autoencoder.py   autoencoder used as the similarity (dissimilarity) model
tavi/vae.py           VAE from the original notebook (not used by the pipeline)
tavi/pipeline.py      leave-one-centre-out validation
tavi/plotting.py      figures
scripts/run_external_validation.py   entry point
```

## Citation

```bibtex
@article{azizmalayeri2026rethinking,
  title   = {Rethinking external validation for the target population: Capturing patient-level similarity with a generative model},
  author  = {Azizmalayeri, Mohammad and Abu-Hanna, Ameen and Houterman, Saskia and Vis, Marije M. and Cin{\`a}, Giovanni},
  journal = {arXiv preprint arXiv:2605.11284},
  year    = {2026}
}
```
