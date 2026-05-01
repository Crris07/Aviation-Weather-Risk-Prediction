# Aviation Weather Risk Prediction

Predict whether airport weather will become operationally unsafe within the next 2 hours using NOAA hourly observations and time-aware machine learning.

## Start Here

If you are viewing this on GitHub and want the walkthrough, open:

- `notebooks/03_baseline.ipynb`

If you want the exported final artifacts directly:

- `reports/figures/phase4_vs_phase5_test_pr_auc.png`
- `reports/figures/phase4_vs_phase5_loro_pr_auc.png`
- `reports/figures/phase5_feature_importance_top20.png`

## Problem

For each hourly timestamp `t`, predict:

- `y_risk = 1` if in `(t+1h, t+2h]` either
- visibility drops below `3 km`, or
- sustained wind exceeds `25 knots`

This is a deliberately hard generalization setup:

- Train airports: `KDEN`, `KDFW`, `KSFO`
- Test airports: `KATL`, `KBOS`
- Train years: `2018-01-01` to `2023-12-31`
- Test years: `2024-01-01` to `2025-12-31`

## What This Project Shows

- End-to-end ingest from NOAA ISD global-hourly data
- Careful label construction on an hourly grid
- Time-series feature engineering with causal lags and rolling windows
- Honest evaluation on unseen airports
- Domain robustness via leave-one-regime-out (LORO)
- Calibration with isotonic regression
- Iterative experimentation, including rejected ideas and corrected data bugs

## Modeling Journey

The final story in this repo is not just "here is the best score." It is:

1. Build an honest single-head baseline.
2. Test more ambitious architectures.
3. Retire the ones that do not solve the real bottleneck.
4. Pivot to features that encode the missing physics.

### Champion System

The current champion remains:

- `Single-head LightGBM`
- `Pooled isotonic calibration`
- `Physics-driven feature enrichment`

The architecture stayed simple. The gains came from making the inputs more informative.

## Phase 4 Clean Baseline

`Phase 4 clean` is the decontaminated baseline after removing duplicated unit columns that had silently inflated effective model capacity.

This is the honest reference model:

- one binary risk head
- calibrated with pooled isotonic on validation data
- evaluated on unseen airports and with LORO domain transfer

Key calibrated test PR-AUC:

- `KATL: 0.727`
- `KBOS: 0.630`

Key calibrated LORO PR-AUC:

- `KSFO: 0.430`
- `KBOS: 0.643`
- `KDEN: 0.614`
- `KDFW: 0.454`
- `KATL: 0.723`

## Architecture Experiment: Multi-Head Specialist vs Single-Head Generalist

In Phase 5, I tested a multi-head LightGBM architecture.

The hypothesis was straightforward:

- visibility risk and wind risk are physically different phenomena
- a visibility specialist head could focus on fog / low-visibility structure
- a wind specialist head could focus on storm / convective structure
- the final safety score could then be formed by combining the two heads

### Result: Null Improvement

The multi-head model did not convincingly beat the single-head baseline, and in some LORO settings it performed worse.

Representative results:

- `KATL` combined PR-AUC: `0.723 -> 0.738` (`+0.015`)
- `KBOS` combined PR-AUC: `0.631 -> 0.640` (`+0.009`)
- `KSFO` LORO PR-AUC: `0.441 -> 0.419` (`-0.022`)

### Why Specialization Failed

Post-experiment diagnostics pointed to two real bottlenecks:

1. `Label dominance and data sparsity`
   Visibility events dominated the combined label. Wind-only events were too rare for the wind head to benefit much from specialization, so splitting the heads diluted supervision instead of sharpening it.

2. `Feature insufficiency`
   Both specialist heads still looked at essentially the same general feature bank. A wind head cannot discover new signal for convective risk if the inputs do not describe the underlying physics well enough.

The lesson was important:

- the bottleneck was not `how we predict`
- the bottleneck was `what the model gets to see`

## Calibration Experiment: Per-Airport Isotonic

I also tested per-airport isotonic calibration against the pooled isotonic calibrator from Phase 4.

### Result: Retired

Per-airport calibration lost across the board.

Calibrated test PR-AUC:

- `KATL: pooled 0.723 vs per-airport 0.708` (`-0.016`)
- `KBOS: pooled 0.631 vs per-airport 0.620` (`-0.010`)

Calibration quality was already excellent with pooled isotonic:

- `ECE ~ 0.006` on both test airports

Why it failed:

- the test airports used transferred calibrators rather than native ones
- pooled isotonic already had very little headroom left to improve
- per-airport isotonic fit on smaller positive counts and transferred worse

Conclusion:

- multi-head architecture was not the fix
- per-airport calibration was not the fix

That narrowed the bottleneck to feature insufficiency, especially for:

- `KSFO` marine-layer fog
- `KDFW` convective wind / severe-weather structure

## Phase 5 Physics-Driven Features

After retiring the architecture and calibration branches, the project pivoted to generalizable physics-driven features rather than airport-specific hacks.

The main additions were:

1. `dew_depression_min_3h` and `dew_depression_min_6h`
   Persistence of near-saturation, useful for fog formation.

2. `wind_dir_sin/cos x coastal`
   Lets the model learn onshore-flow structure on coastal airports without hard-coded airport rules.

3. `calm_and_saturated`
   A compact fog-ingredient indicator: `wind_knots < 5` and `dew_depression < 2`.

4. `dew_depression x hour_sin`
   A diurnal interaction aimed at dawn / overnight fog timing.

5. `pressure_drop_3h x wind_speed`
   A frontal-passage / squall proxy.

This is the part that ties the project together: the physics features were not random additions. They were the direct response to the failure analysis from the multi-head and calibration experiments.

## Final Comparison: Phase 4 Clean vs Phase 5 Physics

The final comparison in this repo is:

1. `Phase 4 clean`
   Single-head LightGBM + pooled isotonic on the decontaminated feature set.
2. `Phase 5 physics`
   The same baseline plus the physics-driven features above.

### Unseen Test Airports

Calibrated test PR-AUC:

- `KATL: 0.727 -> 0.719` (`-0.0076`)
- `KBOS: 0.630 -> 0.640` (`+0.0096`)

Interpretation:

- the physics features did not produce a uniform lift on pooled unseen-airport test performance
- but they did help on the coastal test airport, which is directionally consistent with the marine-layer hypothesis

### LORO Domain Robustness

Calibrated LORO PR-AUC improved consistently:

- `KSFO +0.0063`
- `KDEN +0.0059`
- `KDFW +0.0059`
- `KBOS +0.0034`
- `KATL +0.0029`

Interpretation:

- the physics features were more valuable for `robustness across regimes` than for brute-force pooled test lift
- that is a meaningful result in an aviation setting, where transfer across airports matters

### Feature Importance

The most influential Phase 5 physics features were:

- `dew_depression_x_hour_sin`
- `dew_depression_min_6h`
- `pressure_drop_x_wind`

This supports the core thesis that the new features were not decorative. The model actually used them.

## Repository Layout

```text
aviation-weather/
|-- config.yaml
|-- README.md
|-- notebooks/
|   |-- 01_eda.py
|   |-- 02_features.py
|   |-- 03_baseline.py
|   `-- 03_baseline.ipynb
|-- reports/
|   `-- figures/
|-- src/
|   |-- ingest.py
|   |-- label.py
|   |-- features.py
|   |-- baseline.py
|   |-- compare_variants.py
|   |-- export_presentation_notebook.py
|   `-- _feature_validate.py
`-- requirements.txt
```

## Reproducing The Final Outputs

Run the final experiment chain:

```bash
python -m src.features --variant phase4_clean
python -m src._feature_validate --variant phase4_clean
python -m src.baseline --variant phase4_clean

python -m src.features --variant phase5_physics
python -m src._feature_validate --variant phase5_physics
python -m src.baseline --variant phase5_physics

python -m src.compare_variants
python -m src.export_presentation_notebook
```

Then open:

- `notebooks/03_baseline.ipynb`

## Notes

- Raw NOAA and processed parquet files are not committed.
- The recruiter-facing notebook and final comparison figures are committed.
- The repo intentionally preserves negative results and rejected experiments, because the quality of the reasoning is part of the work product.
