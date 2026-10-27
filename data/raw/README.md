# Raw inputs intentionally excluded

The default preparation script expects:

```text
data/raw/real/EffectiveDataset_Tseries_with_azimuth.csv
data/raw/control/control_crosscheck_effective_target_projection.csv
data/raw/control/control_crosscheck_pointing_audit.csv
```

The audit scripts additionally require the pointing master, logs and DAT
directories. FEM experiments require their workbook, v2 labels and manifest,
and optionally expert checkpoints/physical operators. Consult the notebook
configuration cells for their exact schemas and paths.

These are not bundled or downloadable automatically. Their publication
permission is distinct from the permission for the 668-row derived matrix.
Do not commit raw logs, DAT records or private metadata after running audits.
The repository `.gitignore` excludes these materials defensively.
