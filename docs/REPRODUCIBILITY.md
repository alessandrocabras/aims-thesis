# Reproducibility levels

1. **Frozen evidence verification:** Python standard library only. Checks
   dataset structure, chronological membership, numerical alignment, saved
   metrics, baselines, notebook cleanliness and the allowlisted release.
2. **Figure regeneration:** NumPy/Matplotlib; uses bundled snapshots. No model
   fitting, new inference or bootstrap recomputation.
3. **Feature-level final-model retraining:** published 43-feature matrix,
   pandas/scikit-learn stack; 400-event fitting population. New outputs go
   under `reproduced/final/` and are compared against reference scores.
4. **Raw reconstruction and FEM/SSL experiments:** source code is included,
   but external raw data and experiment artefacts are required. This level
   is not independently reproducible from this public package alone.

The original production environment was not exported as a complete lockfile.
The dependency ranges are compatible installation constraints, not evidence of
bitwise reproducibility. A locally validated environment report is written
with each retraining run. Floating-point and optimisation differences may
change fitted coefficients or predictions; retain the reference values and
report discrepancies rather than overwriting them.

To run the public checks:

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
```

Publication checks expect clean notebook outputs. The interactive review
notebook uses `--allow-executed-review`, which allows its own saved outputs
but still verifies data, metrics and the preserved experimental sources.
Clear the review notebook's outputs before committing it after execution.

To fit the final estimators in an isolated output directory:

```bash
python scripts/retrain_from_features.py
```

To regenerate all figures without changing published originals:

```bash
python scripts/render_figures.py
```

The legacy figure launcher and renderer remain available as original source.
Running them directly overwrites the published figure copies and consequently
fails their provenance hashes. Use the wrapper above for reviewer runs.

## Local packaging verification

Feature-level retraining was exercised with Python 3.11, NumPy 2.4.6,
pandas 2.3.3, SciPy 1.17.1, scikit-learn 1.6.1, joblib 1.5.3 and
Matplotlib 3.11.1. These are verification versions, not a reconstruction of
the original production environment.

| Axis / split | Reference MAE | Local rerun MAE | Rerun minus reference |
| --- | ---: | ---: | ---: |
| XEL / validation | 4.210947 | 4.207657 | -0.003290 |
| XEL / historical test | 4.891609 | 4.889165 | -0.002444 |
| EL / validation | 8.155283 | 8.155283 | less than 0.000001 |
| EL / historical test | 7.780369 | 7.780369 | less than 0.000001 |

These reruns are numerical checks, not new evidence of generalisation. The
small Huber difference is recorded rather than relabelled as an improvement.
The original reference CSVs and source files were not modified. Git is set
to preserve file bytes across platforms so that provenance SHA-256 checks
remain valid after cloning.

The original feature-builder source is preserved. Sensor imputation/binning,
PCA fitting scope and the observational availability caveats are documented
in `METHODOLOGY.md`; feature-level retraining does not fix or independently
validate those upstream choices.

Public module names, dataset filenames and notebook references use AIMS.
These are naming-only changes: feature values, targets, saved predictions and
model settings are unchanged. `docs/source_manifest.json` retains the original
source SHA-256 identities and the current release hashes; original source
filenames are omitted to keep the public naming consistent.
