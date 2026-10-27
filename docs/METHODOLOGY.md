# Methodology and interpretation

## Populations and artefacts

The final derived dataset has 668 chronological events (400/133/135).
The original feature builder fits current-state and endpoint-difference
scalers/PCA on the 400 training events. The regression pipelines fit their
imputation, standardisation and regressors on those same rows. The reported
validation/historical-test results do not come from a 533-event refit.

The original training function subsequently fits separate downstream
deployment models on all 668 rows using the existing representation.
Those models are not independently evaluated by the reported test scores.
The review wrapper redirects all generated artefacts to `reproduced/final/`.

## Targets and error conventions

The final XEL target is the audited longitudinal target multiplied by the
cosine of the corresponding elevation. The historical CSV label `XEl Offset`
was misleading: do not reuse it without checking its coordinate convention.
The published numerical target is `target_XEL_arcsec`; no second projection
must be applied to it. EL is `target_EL_arcsec`.

Residual = observed target - prediction. MAE, RMSE, signed bias and p90 are
in arcseconds. Within-5 is an event fraction, not radial-error compliance.
Receiver subsets are 109 K, 11 C-high and 15 C-low historical-test events.

For a held constant azimuth correction c_A, its cross-elevation effect is
c_XEL(t) = c_A cos(E(t)). Holding XEL constant instead is a different policy.
Sign, units, state persistence and replacement versus incremental command
semantics must be confirmed before comparing with an actual control procedure.

## Historical evaluation, not prospective certification

The 135-event historical test was inspected during development. The release
preserves retrospective results; it does not claim that freezing them creates
an independent test. Three later transfer outer folds use the preceding 533
events: 266/355/444 training events and 89 evaluation events per fold.

Notebook 40 contains exploratory screening, including configurations where
outer evaluation data guided early stopping. Notebook 41 uses inner
chronological selection of label-dependent epoch counts and residual strength.
Its thermal projector is still fitted on the entire outer training population,
including inner-validation covariates. This protects outer evaluation targets
but is not fully nested preprocessing at every level.

RidgeCV uses its default internal leave-one-out regularisation procedure,
not an inner chronological split, with transformations fitted upstream.

## FEM transfer

FEM v2 `common_only` changes synthetic targets, not the matched input sensor
histories. Comparisons use the same rows/windows at 30, 60 and 85 degrees.
F1A/F2A transfer only the encoder and both reset the supervised head.

The notebook-41 EL mean per-run MAEs are Ridge 7.151379, F1A 8.736162,
F2A 8.607756 arcsec. The F2A-minus-F1A paired bootstrap contrast is
-0.133388 with 95% interval [-0.554601, 0.290898]. The matched transfer gate
fails. Ridge plus an F1A residual correction has mean MAE 7.039861, but its
paired interval [-0.339333, 0.043706] also includes zero and the required
absolute gain is not met.

Mean per-run MAE and MAE of seed-averaged predictions are different quantities.
The day-block bootstrap contrasts use the latter and report the bootstrap
mean delta; they need not equal a subtraction of the per-run summary means.
The intervals are conditional on fitted predictions, not a complete accounting
of model-selection uncertainty. Favourable resample fractions are not posterior
probabilities. Gate-dependent tests not executed, and operator tests blocked
by missing inputs, must not be counted as negative experimental results.

## Known online/procedural limitations

The offline loader floors timestamps to two-minute bins, averages within a
bin, and allows limited backward filling. The last-bin average or a filled
value can include an observation unavailable at the nominal prediction time.
An as-of timestamp audit or a strictly availability-aware loader is required
before claiming causal online operation. The package preserves this behaviour
for traceability rather than silently changing the historical experiment.

All 668 selected events have fit flags 1,1 in the source audit and zero
nonzero-azel command count under its reconstruction. This is not proof that
every fitted correction was accepted and applied. The audit parser records
`Command issued`, not a certified execution acknowledgement. Initial state,
implicit receiver/session resets and command accumulation semantics require
further verification; missing state must not be silently treated as zero.

The public repository does not establish commissioned integration in DISCoS,
elimination of astronomical calibration, or compliance with all receiver needs.
