# Public evaluation policy

This is a policy-only derivative, not the original real-dataset audit or frozen
instantiation. No real-data row counts, zero counts, sensor counts, timestamps,
checksums, dataset paths or instantiated partition boundaries are published here.
The original research files remain unchanged outside this source export.

The companion configs/evaluation_protocol.json preserves the operational window,
selection, interval, fault-grid, controller and resource settings used by the
existing synthetic modules. Removed fit-row endpoints are expressed as train-only
fit scope; the dataset-sized contiguous selector is expressed symbolically as
floor(rate * sensor_count). The separately documented small synthetic fixture
uses its existing ceil adaptation; neither injection implementation is changed.

Chronological outer split is70/10/20, with validation roles40/30/30 for tuning,
interval calibration and controller selection. Each window contains12 historical
steps and12 future targets; scored horizons3/6/12 mean15/30/60 minutes. No window
crosses a partition. Fit preprocessing and models on train only. Tune only on
tuning; calibrate only on calibration; select joint controller thresholds only
on controller-selection data. Final-test scoring remains disabled and cannot be
used for development.

Zeros are ambiguous, not confirmed faults. Primary metrics use unchanged finite
positive targets. Inputs use causal within-window forward fill and train-fitted
nonzero median fallbacks, with frozen scaling and missingness indicators. Fault
overlays change historical inputs only and preserve targets/masks across models.
Seeds, rates and severity stay in the companion configuration unchanged.

Coverage diagnostics on calibration are in-sample. No general distribution-free
time-series/fault coverage guarantee is claimed. The existing API only exposes
calibration-derived intervals for supported clean fixed-XGBoost predictions.
Fallback/switched/faulty intervals remain unavailable. Four CPU threads maximum,
no GPU, and1800-second experiment bounds remain unchanged.
