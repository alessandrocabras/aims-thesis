# Thesis figures

All 13 chapter figures are supplied as vector PDF and 300-dpi PNG, with
English labels. `00_contact_sheet.png` is an overview, not a chapter figure.

```bash
python scripts/render_figures.py
```

The bundled `source_data/` snapshots make this command self-contained.
New figures go to `reproduced/thesis_figures/`; the published originals and
their provenance hashes remain unchanged. The legacy direct renderer and
Colab launcher overwrite files in this directory, so prefer the wrapper
or `notebooks/review/00_review_results.ipynb` when reviewing this repository.
No model inference or training occurs. Metrics are checked against the saved
predictions; historical-test rows are selected by joining to the fixed split.
Bootstrap intervals are read from saved experiment output, not recalculated.

The `--refresh-sources` option is for the original full workspace layout,
whose paths are recorded in the source manifest. Do not use it in this review
copy unless deliberately supplying that original layout.

Optional rendering QA: install `pymupdf` and `pillow`, then run
`python reproduced/thesis_figures/verify_figures.py` after regeneration.
Generated previews under `qa/` are ignored by Git.

Copy the `thesis_figures/` directory beside the main thesis TeX file to match
the revised chapter's figure paths. The thesis manuscript and its bibliography
are not distributed in this repository.

| Stem | Content |
| --- | --- |
| `01_target_coordinate_audit` | Provenance and coordinate conventions |
| `02_final_data_pipeline` | Sample construction and exclusions |
| `13_dataset_timeline` | Targets and chronological split |
| `14_temporal_evaluation_design` | Historical and nested evaluation roles |
| `10_fem_real_thermal_shift` | Domain classification and support coverage |
| `04_final_prediction_pipeline_parallel` | Independent XEL/EL branches |
| `15_matched_transfer_protocol` | Paired encoder transfer and head reset |
| `16_fem_transfer_controls` | Fold/seed results and paired intervals |
| `17_observed_vs_predicted` | Both-axis retrospective predictions |
| `18_absolute_error_ecdf` | Central distribution and complete error tail |
| `09_el_residual_vs_elevation` | Signed residuals by elevation |
| `07_mae_by_receiver` | Receiver-conditioned MAE and counts |
| `12_selective_confidence_curve` | Expert-stack selective curve, not final Huber |
