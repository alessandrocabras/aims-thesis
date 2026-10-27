# Experiment index

Read the short reviewer notebook first. The development notebooks below
are preserved for inspection, with outputs, execution counts and nonessential
metadata removed. The original cells mostly retain Italian commentary;
system names and imports have been updated to AIMS. A preserved result or wording is not endorsed
as the final methodological conclusion; see `METHODOLOGY.md`.

| Notebook | Purpose | Required external inputs |
| --- | --- | --- |
| `10_real_data_baselines_and_diagnostics.ipynb` | Baselines, thermal/geometry ablations, window experiments | Real telemetry and audit data; some sections also FEM |
| `20_fem_real_calibration_and_domain_shift.ipynb` | Domain support, calibration and drift diagnostics | Real and FEM data, comparison exports |
| `30_control_logs_true_xel_and_receiver_audit.ipynb` | Target audit, receiver/quality tests, MoE diagnostics | Audit/telemetry data; optional saved neural experts |
| `40_fem_transfer_and_real_ssl.ipynb` | Matched FEM v1/v2 learning curves and exploratory transfer | Raw real histories, FEM workbook, v2 labels/manifest |
| `41_fem_v2_nested_cv_controls.ipynb` | Paired-head EL controls, nested epoch selection and cross-fitted residual correction | Same raw inputs; physical-operator subtest additionally needs both operators |

These notebooks are not an automated public-data-only benchmark. Several
require institutionally held inputs or neural checkpoints that are deliberately
not redistributed. The historical prototype and duplicate root-level notebooks
are omitted. No third-party concept-drift repository is vendored.

For original-window reconstruction, the restricted preparation notebook and
`scripts/prepare_final_dataset.py` expect the paths documented in
`data/raw/README.md`. Do not substitute synthetic dummy inputs to make the
experiment appear reproduced. Leave unavailable tests marked unavailable.

The final-model review path is intentionally separate: it uses the released
derived matrix and writes new runs outside the frozen reference directory.
