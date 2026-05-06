# Astrolabe Technical Notes

For installation, API usage, and citation, see the root [README.md](../README.md).

This document focuses on software and technical details.

## Code structure

- `core.py`: main pipeline (`Astrolabe`), SHAP, network build, ablation, persistence.
- `network.py`: shared network construction (nodes, links, percentiles).
- `widget.py`: anywidget backend, traitlets sync, explicit plot preparation.
- `static/widget.js`: frontend graph and plot interactions.
- `static/widget.css`: widget styles.

## Pipeline outline

1. Validate and clean input (`df_X`, `df_y`).
2. Fit XGBoost (or use a supplied pre-trained model).
3. Compute SHAP values and SHAP interactions.
4. Compute linear baseline coefficients.
5. Build feature metadata.
6. Build network data (`nodes`, `links`, percentiles).
7. Optionally run ablation experiments.

## Model defaults

By default, Astrolabe uses the XGBoost library defaults. Astrolabe only injects:

- `random_state=<user value, default 42>`
- for classification: `use_label_encoder=False`, `eval_metric="logloss"`

Use `xgb_kwargs` when you want explicit reproducible hyperparameters.

e.g.)

```python
xgb_kwargs = {
    "learning_rate": 0.09,
    "n_estimators": 300,
    "max_depth": 3,
    "reg_alpha": 0.3,
    "reg_lambda": 1.3,
}
astro = astrolabe.model(df_X, df_y, xgb_kwargs=xgb_kwargs)
```

Project dependency pin currently uses `xgboost==3.0.1`.

## Network schema

Node fields:

- `id`
- `importance` (mean absolute SHAP main effect)
- `direction` (corr(feature, SHAP))
- `linear_coefficient`

Link fields:

- `source`, `target`
- `importance` (normalized interaction)
- `direction` (signed interaction ratio)
- `corr_spearman`, `corr_pearson`
- `mutual_information`
- percentile fields: `corr_percentile`, `corr_pearson_percentile`, `mi_percentile`, `interaction_percentile`

## Edge typing config

Global thresholds are configured through `EdgeConfig` / `astrolabe.config(...)`:

- `redundant`
- `model_driven`
- `nonlinear_mi`
- `nonlinear_corr`

`status()` reports redundant relationships from absolute Pearson correlation (`|r| >= 0.7`).

## Widget data model

`AstrolabeWidget` syncs these core traitlets:

- `network_data`
- `selected_node_ids`
- `plot_data`
- `edge_config`
- `ablation_candidates`
- `ablation_results`
- `node_threshold`, `link_threshold`
- `plot_mode`, `plot_smooth`, `plot_smooth_frac`, `plot_only`

Plot-only rendering is available through `Astrolabe.plot(...)`, which forwards to `prepare_explicit_plot(...)`.

## Persistence

`Astrolabe.save(path)` uses `joblib.dump` and `astrolabe.load(path)` uses `joblib.load` with type checking.

Recommended file extension: `.astrolabe`.