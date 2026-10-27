# Released data dictionary

## Final feature matrix

`data/processed/aims_clean_features.csv`: 668 rows, 51 columns, UTF-8 CSV.
Rows are cross-scan events, not individual telemetry samples. Times are
interpreted as UTC. Coverage: 2023-10-23 through 2025-08-27, non-continuous.

| Columns | Meaning / units |
| --- | --- |
| `Timestamp` | Event timestamp, interpreted as UTC |
| `Split` | Fixed `train`, `validation`, `test`; test means historical test |
| `Elevation`, `Azimuth` | Telescope geometry in degrees |
| `ReceiverCode`, `ReceiverBand` | Audited context; bands K, C_high, C_low, Q |
| `target_XEL_arcsec` | Audited physical cross-elevation target in arcseconds |
| `target_EL_arcsec` | Elevation target in arcseconds |
| `feature__*` | 43 predictors consumed by the regression pipeline |

Only columns prefixed `feature__` enter Huber/Ridge. The first eight columns
are targets, context and provenance, not an invitation to train on `Split`
or either target.

Feature groups: 13 thermal summaries; elevation and its sine/cosine (3);
cyclic hour, day-of-year and azimuth (6); current-state PCA (8); endpoint
thermal-change PCA (8); five receiver indicators including unknown (5).
PCA values depend on the training-fitted scaler and components; they are not
temperatures or physical deformation amplitudes. Endpoint changes are not
time-normalised rates. The exact implementation is in `src/aims_pipeline.py`.

The matrix supports feature-level retraining. It cannot reconstruct the
original 60 x 295 histories or independently refit their PCA from scratch.
Three active-commanded-offset cases and 197 invalid thermal windows were
removed from the 868 mapped targets to obtain the 668 events.

## Frozen results

`results/reference/final_predictions.csv` contains BOTH validation and
historical-test predictions. Join on `Timestamp` to the feature matrix and
filter `Split == 'test'` for the 135-event results. Each event has one row
per axis. There is no train-prediction population in this table.

| Column | Meaning |
| --- | --- |
| `axis` | XEL or EL |
| `target`, `prediction` | Observed and predicted offsets, arcseconds |
| `residual` | `target - prediction`, arcseconds |
| `abs_error` | Absolute residual, arcseconds |
| `Timestamp`, geometry, receiver | Join/context fields |

`final_metrics.csv` gives count, MAE, RMSE, R2, bias, within_5_percent and
p90_abs_error by axis/split. The receiver/elevation tables refer to the
historical test. The constant baseline is computed as the median target of
the 400 training rows, held constant across the evaluation population.

## Plot snapshots and provenance

`thesis_figures/source_data/` contains the minimal frozen plotting inputs,
including an eight-column event projection and saved experiment summaries.
Some FEM summaries are six-decimal printed outputs, not row-level predictions.
Their intervals are imported, not reconstructed from the summary means.

`docs/source_manifest.json` records original workspace-relative paths and
hashes. Original paths are provenance, not evidence that all restricted
original files are distributed. No absolute personal workstation paths are
needed to run the public quickstart.
