# AIMS

**Thermal-aware residual pointing prediction for the Sardinia Radio Telescope.**

This is a curated thesis-review repository, not a deployed telescope-control
system. It contains the selected real-only pipeline, the derived supervised
dataset, frozen predictions, experimental notebooks and the figures used to
discuss the work. AIMS is the final system name. The public repository is
[alessandrocabras/aims-thesis](https://github.com/alessandrocabras/aims-thesis).
Module and dataset filenames use the `aims_*` prefix consistently.

## Start here

1. Read [the methodology and limitations](docs/METHODOLOGY.md).
2. Verify the frozen numerical evidence with `python scripts/check_release.py`.
3. Inspect [reference metrics](results/reference/final_metrics.csv) and
   [the data dictionary](docs/DATA_DICTIONARY.md).
4. Browse [the figure overview](thesis_figures/00_contact_sheet.png).
5. Use [the experiment index](docs/EXPERIMENTS.md) for historical ablations and
   the controlled FEM v1/v2 comparison.

## Reported retrospective performance

The dataset contains **668 events and 43 predictors** from October 2023 to
August 2025. The chronological split is 400 training, 133 validation and
135 historical-test events. The reported Huber/Ridge predictions and their
learned preprocessing use the **400 training events**, not a 533-event refit.

| Axis | Selected model | Historical-test MAE | Training-median baseline MAE | Errors within 5 arcsec |
| --- | --- | ---: | ---: | ---: |
| XEL | Huber | 4.891609 arcsec | 5.077449 arcsec | 62.22% |
| EL | Ridge | 7.780369 arcsec | 13.171300 arcsec | 50.37% |

The constant baselines are reconstructed from the released training targets
and evaluated on exactly the same historical-test rows. A check script
reproduces these numbers; it does not train or select a model.

**The historical test was inspected during development.** These are
retrospective estimates requiring confirmation on a new prospective interval.
Freezing predictions does not make that interval retrospectively independent.
The 5-arcsec threshold is a reference, not a universal receiver specification.

## Repository map

```text
src/                    Original selected pipeline and FEM label-join utility
scripts/                Verification, isolated retraining, preparation and audit
notebooks/review/       Short runnable reviewer entry point
notebooks/experiments/  Preserved development notebooks, with cleared outputs
notebooks/restricted/  Raw-data preparation notebook; external inputs required
data/processed/        Released 668-event feature matrix and dataset report
data/raw/README.md     Required raw inputs, deliberately not distributed
results/reference/     Frozen validation/test predictions and metrics
thesis_figures/        13 vector PDFs, PNGs, renderer and frozen plotting inputs
docs/                  Protocol, experiment index, provenance and publication guide
tests/                 Data/metric integrity and safety checks
reproduced/            New local runs only; ignored by Git
```

## Installation

Python 3.11 is the reference interpreter for this review package.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_release.py
python -m unittest discover -s tests -v
```

The integrity checker and its tests require only the Python standard library.
The dependency ranges above are not claimed to reconstruct the original
training environment exactly. See [reproducibility levels](docs/REPRODUCIBILITY.md).

## Reproduce the final fits from released features

```bash
python scripts/retrain_from_features.py
```

This fits Huber and Ridge on the 400 training rows and evaluates validation
and the historical test. New metrics, figures and serialized models go under
`reproduced/final/`, never into `results/reference/`. The original function
also creates separate models fitted on all 668 rows; these are not the models
used to produce the reported evaluation scores. The shared feature builder
was fitted on the original 400 events and is not refitted by this command.

Re-running this comparison does not turn the historical test into a fresh test.
Do not select new models by repeatedly optimising against these 135 events.

## Reproduce the thesis figures

```bash
python scripts/render_figures.py
```

All plotting inputs are included. This renders saved results without model
inference, training or bootstrap recomputation. See the
[figure guide](thesis_figures/README.md). Regenerated PDFs/PNGs go under
`reproduced/thesis_figures/`; published originals are unchanged.

For notebooks, install `requirements-notebooks.txt`, start `jupyter lab` and
open `notebooks/review/00_review_results.ipynb`. Local Colab can use the same
repository through its mounted volume; set its working directory to the
repository root. No Google Drive mount is required.

## What is and is not reproducible here

Included: final feature-level fitting, validation/test metric verification,
constant baselines, receiver metrics, all 13 thesis figures, and code inspection
of the development experiments.

Not included: raw sensor histories, control logs, `.dat` files, FEM workbooks,
FEM-v2 label files, fitted neural experts, physical response operators, or
serialized deployment models. Reconstructing PCA/window inputs from raw data
and rerunning the FEM/SSL experiments requires those materials. Their absence
is explicit, not replaced with synthetic stand-ins. No access commitment for
institutional data is implied by this repository.

## Main methodological conclusions

- Target-coordinate verification and control-state audit precede modelling.
- XEL is a projected coordinate, not the longitudinal azimuth offset.
- Compact real-only models are the retained reference under the tested protocols.
- In the controlled EL study, F2A (8.608 arcsec) did not achieve the specified
  improvement over F1A (8.736 arcsec) and remained worse than Ridge (7.151 arcsec).
- The best neural residual correction had a small gain with an interval
  including zero. No new historical-test evaluation was triggered by these gates.
- Strict online causality and correction-application semantics remain to be
  verified before control integration. See [limitations](docs/METHODOLOGY.md).

## Rights and attribution

Code/data licensing and institutional clearance must be confirmed by the
repository owner before public release. No open-source licence is assigned
automatically by this packaging step. See [RIGHTS.md](RIGHTS.md) and
[publication instructions](docs/PUBLISHING.md). Authors and a permanent citation
should be added once the repository identity and institutional attribution are
confirmed; no DOI, author list or affiliation has been invented.
