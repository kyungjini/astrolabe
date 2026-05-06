# Astrolabe

Astrolabe is a visual analytics system that reframes tabular model interpretation as an interactive feature relationship network. It bridges the gap between granular SHAP-based attributions and complex structural dependencies to uncover insights often missed by individual feature rankings.

This work is currently under review (VIS 2026 Short Paper).

## Installation

```bash
git clone https://github.com/kyungjini/astrolabe
cd astrolabe
pip install .
```

Python: **3.10+**

## Quick start

```python
import astrolabe

astro = astrolabe.model(df_X, df_y, task="regression")
astro.explore()
```

Save/load analysis objects:

```python
astro.save("run.astrolabe")
reloaded = astrolabe.load("run.astrolabe")
```

## API

- `astrolabe.model(df_X, df_y, task="regression", **kwargs)`
- `astrolabe.load_model(df_X, df_y, task="regression", model=..., **kwargs)`
- `astrolabe.load(path)`
- `astrolabe.config(...)`

Main object methods:

- `Astrolabe.explore()`
- `Astrolabe.save(path)`
- `Astrolabe.plot(...)` (optional plot-only helper)
- `Astrolabe.status()`
- `Astrolabe.show_multicollinearity(threshold=0.7)`

## Citation

N/A

## License

Apache License 2.0. See [LICENSE](LICENSE).
