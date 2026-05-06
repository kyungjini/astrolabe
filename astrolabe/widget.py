from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import anywidget
import traitlets
import pandas as pd
import numpy as np


class AstrolabeWidget(anywidget.AnyWidget):
    """AnyWidget-based interactive explorer for an :class:`Astrolabe` object.

    This wraps the D3.js network graph (in ``static/widget.js``) and
    renders vega-lite charts via vega-embed in response to node selections.
    """

    # Frontend bundle
    _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "widget.css"

    # Data and selection traitlets
    network_data = traitlets.Dict({}).tag(sync=True)
    selected_node_ids = traitlets.List([]).tag(sync=True)
    plot_data = traitlets.Dict({}).tag(sync=True)  # Data for vega-embed plots
    edge_config = traitlets.Dict({}).tag(sync=True)  # Edge classification thresholds
    ablation_candidates = traitlets.List([]).tag(
        sync=True
    )  # Features with ablation experiments
    ablation_results = traitlets.Dict({}).tag(sync=True)  # Ablation experiment results
    # When set, scatter/SHAP plots use cached ablation SHAP for this removed-feature key.
    ablation_view_target = traitlets.Unicode("").tag(sync=True)

    # Threshold traitlets (tunable from frontend)
    node_threshold = traitlets.Float(0.05).tag(sync=True)
    link_threshold = traitlets.Float(0.05).tag(sync=True)

    # Plot mode for bivariate plots (2 features selected)
    # "shap_dependence": Feature vs SHAP, colored by other feature
    # "feature_scatter": Feature vs Feature, colored by SHAP
    plot_mode = traitlets.Unicode("shap_dependence").tag(sync=True)
    # Plot smoothing controls (for scatter trendline overlay)
    plot_smooth = traitlets.Bool(False).tag(sync=True)
    plot_smooth_frac = traitlets.Float(0.3).tag(sync=True)
    # Plot-only mode hides the network/sidebar and renders the scatter panel only
    plot_only = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        astro: "Astrolabe",
        plot_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Core data from Astrolabe (lazy, in-memory only)
        self.network_data = astro.network_data
        self.df_X: pd.DataFrame = astro.df_X
        self.shap_value: np.ndarray = astro.shap_values
        self.shap_interaction: np.ndarray = astro.shap_interaction_values
        self.feature_metadata: Dict[str, Dict[str, Any]] = astro.feature_metadata
        self.plot_only = plot_only
        self.plot_fixed_range = None
        self.plot_manual_range_enabled: bool = False
        self.plot_manual_range: Optional[
            tuple[tuple[float, float], tuple[float, float]]
        ] = None

        # Pass edge config to JavaScript
        from .core import _EDGE_CONFIG

        self.edge_config = _EDGE_CONFIG.to_dict()

        # Pass ablation study data to JavaScript
        self.ablation_candidates = astro.ablation_candidates_
        # Convert ablation results to serializable format
        ablation_results_serializable = {}
        for feature, result in astro.ablation_results_.items():
            if result is not None:
                ablation_results_serializable[feature] = {
                    "nodes": result["nodes"],
                    "links": result["links"],
                    "feature_names": result["feature_names"],
                }
        self.ablation_results = ablation_results_serializable

        # Per-ablation numpy cache (not synced to JS — avoids huge JSON)
        self._ablation_plot_cache: Dict[str, Dict[str, Any]] = {}
        for fname, result in (astro.ablation_results_ or {}).items():
            if result is None or "shap_values" not in result:
                continue
            rfnames = list(result["feature_names"])
            sv = np.asarray(result["shap_values"])
            p = len(rfnames)
            n = sv.shape[0]
            if sv.ndim != 2 or sv.shape[1] != p:
                continue
            si = result.get("shap_interaction_values")
            if si is None:
                # Older .astrolabe / ablation runs without interaction tensor — still use ablated SHAP
                si = np.zeros((n, p, p), dtype=sv.dtype)
            else:
                si = np.asarray(si)
                if si.shape != (n, p, p):
                    si = np.zeros((n, p, p), dtype=sv.dtype)
            self._ablation_plot_cache[fname] = {
                "feature_names": rfnames,
                "df_X": astro.df_X[rfnames].copy(),
                "shap_values": sv,
                "shap_interaction_values": si,
            }

        # Wire up JS<->Python messaging
        self.on_msg(self._handle_custom_msg)
        self._last_displayed_selection: tuple[str, ...] | None = None
        self._current_theme: str = "light"  # Track current theme for charts

        # Observe selection changes on the Python side if needed later
        try:
            self.observe(self._on_selection_change_py, names=["selected_node_ids"])
        except Exception:
            # Some AnyWidget versions may differ; ignore
            pass

        # Observe plot_mode changes to update plot when mode toggles
        try:
            self.observe(self._on_plot_mode_change, names=["plot_mode"])
        except Exception:
            # Some AnyWidget versions may differ; ignore
            pass

        try:
            self.observe(self._on_plot_smooth_change, names=["plot_smooth"])
        except Exception:
            pass

        try:
            self.observe(self._on_ablation_view_change, names=["ablation_view_target"])
        except Exception:
            pass

        # Initialize plot_data
        self.plot_data = {}

    def prepare_explicit_plot(
        self,
        mode: str,
        x: str,
        y: Optional[str] = None,
        color: Optional[str] = None,
        ablation: Optional[str] = None,
        smooth: bool = False,
        smooth_frac: float = 0.3,
        fixed_range: Optional[list[list[float]]] = None,
    ) -> None:
        """Prepare a standalone plot without relying on the network selection UI.

        Parameters
        ----------
        mode : str
            Either ``SHAPdependence`` or ``FeatureScatter``.
        x : str
            Primary feature for the x-axis.
        y : str, optional
            Secondary feature for feature-scatter mode.
        color : str, optional
            Optional feature used as the color dimension for SHAP dependence.
        ablation : str, optional
            Name of the ablation target (removed feature) to visualize.
            If omitted, uses the original non-ablated state.
        smooth : bool, default False
            If True, overlay a LOWESS trend line on continuous scatter plots.
        smooth_frac : float, default 0.3
            Fraction of points used in each LOWESS local window.
        fixed_range : list[list[float]], optional
            Fixed axis ranges as ``[[x_min, x_max], [y_min, y_max]]``.
            When set, zoom/pan via scroll is disabled.
        """
        self.plot_smooth = bool(smooth)
        self.plot_smooth_frac = float(smooth_frac)
        self.plot_fixed_range = self._normalize_fixed_range(fixed_range)
        cache = getattr(self, "_ablation_plot_cache", {}) or {}
        if ablation is None or str(ablation).strip() == "":
            self.ablation_view_target = ""
        else:
            target = str(ablation).strip()
            if target not in cache:
                available = sorted(cache.keys())
                if available:
                    raise ValueError(
                        f"Unknown ablation target '{target}'. Available targets: {available}"
                    )
                raise ValueError(
                    "No ablation cache is available. Build/load an Astrolabe object with ablation results first."
                )
            self.ablation_view_target = target

        normalized = (mode or "").strip().lower().replace("_", "").replace(" ", "")
        if normalized in {"shapdependence", "dependence"}:
            if x not in self.df_X.columns:
                raise ValueError(f"Feature '{x}' not found in data.")
            selected = [x]
            if color:
                if color not in self.df_X.columns:
                    raise ValueError(f"Feature '{color}' not found in data.")
                selected.append(color)
            self.plot_mode = "shap_dependence"
            self._last_displayed_selection = tuple(selected)
            self._prepare_plot_data(selected)
            if self.plot_data:
                self.plot_data["plot_mode"] = "shap_dependence"
                self.plot_data.setdefault("main_feature", x)
                self.plot_data["color_feature"] = color
            return

        if normalized in {"featurescatter", "scatter"}:
            if x not in self.df_X.columns:
                raise ValueError(f"Feature '{x}' not found in data.")
            if y is None:
                raise ValueError("FeatureScatter mode requires both x and y.")
            if y not in self.df_X.columns:
                raise ValueError(f"Feature '{y}' not found in data.")
            selected = [x, y]
            self.plot_mode = "feature_scatter"
            self._last_displayed_selection = tuple(selected)
            self._prepare_plot_data(selected)
            if self.plot_data:
                self.plot_data["plot_mode"] = "feature_scatter"
                self.plot_data.setdefault("main_feature", x)
                self.plot_data["color_feature"] = y
            return

        raise ValueError("mode must be 'SHAPdependence' or 'FeatureScatter'")

    def _normalize_fixed_range(
        self, fixed_range: Optional[list[list[float]]]
    ) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
        """Validate and normalize fixed axis range input."""
        if fixed_range is None:
            return None
        if not isinstance(fixed_range, (list, tuple)) or len(fixed_range) != 2:
            raise ValueError("fixed_range must be [[x_min, x_max], [y_min, y_max]]")

        x_range, y_range = fixed_range
        if (
            not isinstance(x_range, (list, tuple))
            or not isinstance(y_range, (list, tuple))
            or len(x_range) != 2
            or len(y_range) != 2
        ):
            raise ValueError("fixed_range must be [[x_min, x_max], [y_min, y_max]]")

        x0, x1 = float(x_range[0]), float(x_range[1])
        y0, y1 = float(y_range[0]), float(y_range[1])
        if x0 >= x1 or y0 >= y1:
            raise ValueError("fixed_range must satisfy min < max for both axes")
        return ((x0, x1), (y0, y1))

    def _apply_fixed_range_to_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fixed x/y domains and disable zoom selection when requested."""
        fixed = None
        if getattr(self, "plot_manual_range_enabled", False):
            fixed = getattr(self, "plot_manual_range", None)
        if not fixed:
            fixed = getattr(self, "plot_fixed_range", None)
        if not fixed:
            return spec

        (x0, x1), (y0, y1) = fixed

        def _set_quant_domain(encoding: Dict[str, Any], axis_key: str, domain: list[float]) -> None:
            if axis_key not in encoding:
                return
            axis_cfg = encoding.get(axis_key, {})
            if axis_cfg.get("type") != "quantitative":
                raise ValueError("fixed_range can only be used with quantitative x/y axes")
            scale_cfg = dict(axis_cfg.get("scale", {}))
            scale_cfg["domain"] = domain
            axis_cfg["scale"] = scale_cfg
            encoding[axis_key] = axis_cfg

        if "layer" in spec and isinstance(spec["layer"], list) and spec["layer"]:
            point_layer = dict(spec["layer"][0])
            point_encoding = dict(point_layer.get("encoding", {}))
            _set_quant_domain(point_encoding, "x", [x0, x1])
            _set_quant_domain(point_encoding, "y", [y0, y1])
            point_layer["encoding"] = point_encoding
            point_mark = point_layer.get("mark", {"type": "circle"})
            if isinstance(point_mark, str):
                point_mark = {"type": point_mark}
            point_mark = dict(point_mark)
            point_mark["clip"] = True
            point_layer["mark"] = point_mark

            # Ensure layered overlays (e.g., LOWESS) are clipped too.
            for idx in range(1, len(spec["layer"])):
                layer = dict(spec["layer"][idx])
                layer_mark = layer.get("mark")
                if layer_mark is not None:
                    if isinstance(layer_mark, str):
                        layer_mark = {"type": layer_mark}
                    layer_mark = dict(layer_mark)
                    layer_mark["clip"] = True
                    layer["mark"] = layer_mark
                    spec["layer"][idx] = layer

            if "selection" in point_layer:
                point_layer.pop("selection", None)
            spec["layer"][0] = point_layer
            spec.pop("selection", None)
            return spec

        encoding = dict(spec.get("encoding", {}))
        _set_quant_domain(encoding, "x", [x0, x1])
        _set_quant_domain(encoding, "y", [y0, y1])
        spec["encoding"] = encoding
        mark_cfg = spec.get("mark")
        if mark_cfg is not None:
            if isinstance(mark_cfg, str):
                mark_cfg = {"type": mark_cfg}
            mark_cfg = dict(mark_cfg)
            mark_cfg["clip"] = True
            spec["mark"] = mark_cfg
        spec.pop("selection", None)
        return spec

    def _compute_lowess_line(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        frac: float = 0.3,
        n_points: int = 200,
    ) -> Optional[list[Dict[str, float]]]:
        """Compute a simple LOWESS-style smooth line for a 2D scatter plot.

        This uses a local weighted linear regression with tricube weights.
        The implementation is dependency-free so the plot-only API does not
        require statsmodels.
        """
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if x.size < 3:
            return None

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if np.allclose(x.min(), x.max()):
            return None

        frac = float(min(max(frac, 0.05), 1.0))
        n = x.size
        k = max(3, int(np.ceil(frac * n)))
        grid_size = int(min(max(n_points, 25), max(25, n)))
        x_grid = np.linspace(float(x.min()), float(x.max()), grid_size)
        y_grid = []

        for x0 in x_grid:
            distances = np.abs(x - x0)
            if k < n:
                nearest = np.argpartition(distances, k - 1)[:k]
            else:
                nearest = np.arange(n)
            x_local = x[nearest]
            y_local = y[nearest]
            d_local = distances[nearest]

            d_max = float(np.max(d_local))
            if d_max <= 0:
                weights = np.ones_like(d_local)
            else:
                u = np.clip(d_local / d_max, 0.0, 1.0)
                weights = (1.0 - u ** 3) ** 3

            design = np.column_stack([np.ones_like(x_local), x_local])
            sqrt_w = np.sqrt(weights)
            weighted_design = design * sqrt_w[:, None]
            weighted_target = y_local * sqrt_w

            try:
                beta, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
                y_grid.append(float(beta[0] + beta[1] * x0))
            except np.linalg.LinAlgError:
                y_grid.append(float(np.average(y_local, weights=weights)))

        return [{"x": float(xv), "y": float(yv)} for xv, yv in zip(x_grid, y_grid)]

    def _add_lowess_layer(
        self,
        spec: Dict[str, Any],
        x_field: str,
        y_field: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
    ) -> Dict[str, Any]:
        """Overlay a LOWESS trendline on a Vega-Lite scatter spec."""
        if not self.plot_smooth:
            return spec

        trend = self._compute_lowess_line(
            x_values,
            y_values,
            frac=getattr(self, "plot_smooth_frac", 0.3),
        )
        if not trend:
            return spec

        point_layer: Dict[str, Any] = {
            "mark": spec.get("mark", {"type": "circle"}),
            "encoding": spec.get("encoding", {}),
        }
        if "selection" in spec:
            point_layer["selection"] = spec["selection"]

        return {
            k: v
            for k, v in {
                **spec,
                "layer": [
                    point_layer,
                    {
                        "data": {"values": trend},
                        "mark": {
                            "type": "line",
                            "color": "#111827",
                            "strokeWidth": 2.25,
                            "opacity": 0.95,
                        },
                        "encoding": {
                            "x": {"field": "x", "type": "quantitative"},
                            "y": {"field": "y", "type": "quantitative"},
                        },
                    },
                ],
            }.items()
            if k not in {"mark", "encoding", "selection"}
        }

    # ------------------------------------------------------------------
    # Messaging from JS frontend
    # ------------------------------------------------------------------
    def _handle_custom_msg(self, msg: Dict[str, Any], buffers: Any) -> None:
        mtype = msg.get("type")
        if mtype == "selection":
            selected = msg.get("payload", [])
            if not isinstance(selected, list):
                selected = [selected]

            tuple_selected = tuple(selected)
            if tuple_selected == self._last_displayed_selection:
                return
            self._last_displayed_selection = tuple_selected

            # Keep traitlet in sync
            try:
                self.selected_node_ids = list(selected)
            except Exception:
                try:
                    self.set_trait("selected_node_ids", list(selected))
                except Exception:
                    pass

            # Prepare plot data for vega-embed
            if len(selected) == 0:
                self.plot_data = {}
            else:
                self._prepare_plot_data(selected)
        elif mtype == "ablation_context":
            # Set synchronously on the Python side (avoids trait/selection message reordering in the comm).
            raw = msg.get("payload")
            if raw is None or raw == "":
                self.ablation_view_target = ""
            elif isinstance(raw, str):
                self.ablation_view_target = raw
            else:
                self.ablation_view_target = ""
        elif mtype == "manual_range":
            payload = msg.get("payload", {}) or {}
            enabled = bool(payload.get("enabled", False))
            raw_range = payload.get("range")
            normalized = None
            if raw_range is not None:
                try:
                    normalized = self._normalize_fixed_range(raw_range)
                except Exception:
                    normalized = None
            self.plot_manual_range_enabled = enabled
            self.plot_manual_range = normalized
            if self._last_displayed_selection:
                self._prepare_plot_data(list(self._last_displayed_selection))
        elif mtype == "theme":
            # Update theme for charts
            new_theme = msg.get("payload", "light")
            if new_theme != self._current_theme:
                self._current_theme = new_theme
            # If there's a current selection, redraw charts with new theme
            if self._last_displayed_selection:
                self._prepare_plot_data(list(self._last_displayed_selection))

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _get_feature_type(self, feature_name: str) -> str:
        """Return ``"categorical"`` or ``"continuous"`` for a feature."""
        if self.plot_only and feature_name in self.df_X.columns:
            series = self.df_X[feature_name].dropna()
            if pd.api.types.is_numeric_dtype(series):
                return "continuous"

        if self.feature_metadata:
            meta = self.feature_metadata.get(feature_name, {})
            return meta.get("type", "continuous")

        # Fallback: infer from data
        if feature_name not in self.df_X.columns:
            return "continuous"
        unique_vals = self.df_X[feature_name].dropna().unique()
        return "categorical" if len(unique_vals) <= 10 else "continuous"

    def _active_plot_arrays(
        self, selected: List[str]
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Optional[Dict[str, int]]]:
        """Data for Vega plots: original model, or ablated model when ``ablation_view_target`` matches."""
        target = (self.ablation_view_target or "").strip()
        cache = getattr(self, "_ablation_plot_cache", {}) or {}
        if not target or target not in cache:
            return self.df_X, self.shap_value, self.shap_interaction, None
        c = cache[target]
        names = c["feature_names"]
        if not all(s in names for s in selected):
            return self.df_X, self.shap_value, self.shap_interaction, None
        idx = {n: i for i, n in enumerate(names)}
        return c["df_X"], c["shap_values"], c["shap_interaction_values"], idx

    def _on_ablation_view_change(self, change: Dict[str, Any]) -> None:  # noqa: ARG002
        if self._last_displayed_selection:
            self._prepare_plot_data(list(self._last_displayed_selection))

    def _prepare_plot_data(self, selected: List[str]) -> None:
        """Prepare plot data for vega-embed based on 1 or 2 selected features.

        - 1 feature: univariate (scatter for continuous, boxplot for categorical)
        - 2 features: bivariate (scatter/heatmap depending on types)
        """
        if self.df_X is None or self.shap_value is None:
            self.plot_data = {}
            return

        df_p, shap_p, shap_i_p, idx_map = self._active_plot_arrays(selected)

        def _fidx(f: str) -> int:
            if idx_map is not None:
                return idx_map[f]
            return df_p.columns.tolist().index(f)

        if len(selected) == 1:
            feature = selected[0]
            if feature not in df_p.columns:
                print(f"Feature '{feature}' not found in data.")
                return

            feature_idx = _fidx(feature)
            feature_type = self._get_feature_type(feature)

            plot_df = pd.DataFrame(
                {
                    "Feature Value": df_p[feature].values,
                    "SHAP Value": shap_p[:, feature_idx],
                }
            )

            # Convert to list of dicts for JSON serialization
            data = plot_df.to_dict("records")

            # Prepare vega-lite spec
            if feature_type == "categorical":
                plot_df["Category"] = plot_df["Feature Value"].astype(str)
                data = plot_df[["Category", "SHAP Value"]].to_dict("records")

                spec = {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "description": f"SHAP values by {feature} (Categorical)",
                    # Width/height will be set dynamically in JS to avoid ResizeObserver issues
                    "data": {"values": data},
                    "mark": {"type": "boxplot", "extent": "min-max", "size": 40},
                    "encoding": {
                        "x": {
                            "field": "Category",
                            "type": "nominal",
                            "title": f"{feature} (Categories)",
                            "axis": {
                                "labelAngle": -45,
                                "titleFontSize": 12,
                                "labelFontSize": 10,
                            },
                        },
                        "y": {
                            "field": "SHAP Value",
                            "type": "quantitative",
                            "title": "SHAP Value",
                            "axis": {"titleFontSize": 12, "labelFontSize": 10},
                        },
                        "tooltip": [
                            {"field": "Category", "type": "nominal", "title": feature},
                            {
                                "field": "SHAP Value",
                                "type": "quantitative",
                                "aggregate": "mean",
                                "format": ".3f",
                                "title": "Mean SHAP",
                            },
                            {
                                "field": "SHAP Value",
                                "type": "quantitative",
                                "aggregate": "count",
                                "title": "Count",
                            },
                        ],
                    },
                    "selection": {"grid": {"type": "interval", "bind": "scales"}},
                    "config": {
                        "view": {"stroke": None},
                        "background": self._get_chart_background(),
                        "axis": self._get_chart_config(),
                        "title": {"color": self._get_chart_config()["titleColor"]},
                    },
                }
            else:
                spec = {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "description": f"SHAP values vs {feature}",
                    # Width/height will be set dynamically in JS to avoid ResizeObserver issues
                    "data": {"values": data},
                    "mark": {
                        "type": "circle",
                        "size": 30,
                        "opacity": 0.6,
                        "color": "#6b7280",
                    },
                    "encoding": {
                        "x": {
                            "field": "Feature Value",
                            "type": "quantitative",
                            "title": f"{feature} (Feature Value)",
                            "axis": {"titleFontSize": 12, "labelFontSize": 10},
                        },
                        "y": {
                            "field": "SHAP Value",
                            "type": "quantitative",
                            "title": f"SHAP Value",
                            "axis": {"titleFontSize": 12, "labelFontSize": 10},
                        },
                        "tooltip": [
                            {
                                "field": "Feature Value",
                                "type": "quantitative",
                                "title": feature,
                            },
                            {
                                "field": "SHAP Value",
                                "type": "quantitative",
                                "format": ".3f",
                            },
                        ],
                    },
                    "selection": {"grid": {"type": "interval", "bind": "scales"}},
                    "config": {
                        "view": {"stroke": None},
                        "background": self._get_chart_background(),
                        "axis": self._get_chart_config(),
                        "title": {"color": self._get_chart_config()["titleColor"]},
                    },
                }

            if feature_type == "continuous":
                spec = self._add_lowess_layer(
                    spec,
                    "Feature Value",
                    "SHAP Value",
                    plot_df["Feature Value"].values,
                    plot_df["SHAP Value"].values,
                )
                spec = self._apply_fixed_range_to_spec(spec)

            self.plot_data = {
                "spec": spec,
                "feature_type": feature_type,
                "main_feature": feature,
                "color_feature": None,
                "manual_range_supported": feature_type == "continuous",
                "manual_range_enabled": self.plot_manual_range_enabled,
                "manual_range": self.plot_manual_range,
            }

        elif len(selected) == 2:
            feature1, feature2 = selected[0], selected[1]

            if feature1 not in df_p.columns or feature2 not in df_p.columns:
                print("One or more features not found in data.")
                return

            feature1_idx = _fidx(feature1)
            feature1_type = self._get_feature_type(feature1)
            feature2_type = self._get_feature_type(feature2)

            feature2_idx = _fidx(feature2)
            if feature1_idx < feature2_idx:
                interaction_values = shap_i_p[:, feature1_idx, feature2_idx]
            else:
                interaction_values = shap_i_p[:, feature2_idx, feature1_idx]

            plot_df = pd.DataFrame(
                {
                    "Feature1": df_p[feature1].values,
                    "SHAP Value": shap_p[:, feature1_idx],
                    "Feature2": df_p[feature2].values,
                    "Interaction": interaction_values,
                }
            )

            # Convert to list of dicts for JSON serialization
            data = plot_df.to_dict("records")

            # Case A: continuous vs continuous
            if feature1_type == "continuous" and feature2_type == "continuous":
                bg_color = self._get_chart_background()
                plot_mode = self.plot_mode

                # Mode 1: SHAP Dependence (default) - Feature vs SHAP, colored by other feature
                if plot_mode == "shap_dependence":
                    min_val = float(plot_df["Feature2"].min())
                    max_val = float(plot_df["Feature2"].max())

                    spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "description": f"SHAP values vs {feature1}, colored by {feature2}",
                        # Width/height will be set dynamically in JS to avoid ResizeObserver issues
                        "data": {"values": data},
                        "mark": {"type": "circle", "size": 40, "opacity": 0.7},
                        "params": [
                            {"name": "colorMin", "value": min_val},
                            {"name": "colorMax", "value": max_val},
                        ],
                        "encoding": {
                            "x": {
                                "field": "Feature1",
                                "type": "quantitative",
                                "title": f"{feature1} (Feature Value)",
                                "axis": {"titleFontSize": 12, "labelFontSize": 10},
                            },
                            "y": {
                                "field": "SHAP Value",
                                "type": "quantitative",
                                "title": f"SHAP Value",
                                "axis": {"titleFontSize": 12, "labelFontSize": 10},
                            },
                            "color": {
                                "field": "Feature2",
                                "type": "quantitative",
                                "title": feature2,
                                "scale": {
                                    "scheme": "viridis",
                                    "domain": {"expr": "[colorMin, colorMax]"},
                                    "clamp": True,
                                },
                                "legend": None,
                            },
                            "tooltip": [
                                {
                                    "field": "Feature1",
                                    "type": "quantitative",
                                    "title": feature1,
                                },
                                {
                                    "field": "SHAP Value",
                                    "type": "quantitative",
                                    "format": ".3f",
                                },
                                {
                                    "field": "Feature2",
                                    "type": "quantitative",
                                    "title": feature2,
                                    "format": ".3f",
                                },
                                {
                                    "field": "Interaction",
                                    "type": "quantitative",
                                    "format": ".3f",
                                },
                            ],
                        },
                        "selection": {"grid": {"type": "interval", "bind": "scales"}},
                        "config": {
                            "view": {"stroke": None},
                            "background": bg_color,
                            "axis": self._get_chart_config(),
                            "title": {"color": self._get_chart_config()["titleColor"]},
                        },
                    }

                    spec = self._add_lowess_layer(
                        spec,
                        "Feature1",
                        "SHAP Value",
                        plot_df["Feature1"].values,
                        plot_df["SHAP Value"].values,
                    )
                    spec = self._apply_fixed_range_to_spec(spec)

                    self.plot_data = {
                        "spec": spec,
                        "feature_type": "continuous_continuous",
                        "plot_mode": "shap_dependence",
                        "main_feature": feature1,
                        "color_feature": feature2,
                        "color_min": min_val,
                        "color_max": max_val,
                        "manual_range_supported": True,
                        "manual_range_enabled": self.plot_manual_range_enabled,
                        "manual_range": self.plot_manual_range,
                    }

                # Mode 2: Feature Scatter - Feature vs Feature, colored by SHAP
                else:  # plot_mode == "feature_scatter"
                    shap_min = float(plot_df["SHAP Value"].min())
                    shap_max = float(plot_df["SHAP Value"].max())

                    spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "description": f"{feature1} vs {feature2}, colored by SHAP value of {feature1}",
                        # Width/height will be set dynamically in JS to avoid ResizeObserver issues
                        "data": {"values": data},
                        "mark": {"type": "circle", "size": 40, "opacity": 0.7},
                        "params": [
                            {"name": "colorMin", "value": shap_min},
                            {"name": "colorMax", "value": shap_max},
                        ],
                        "encoding": {
                            "x": {
                                "field": "Feature1",
                                "type": "quantitative",
                                "title": f"{feature1}",
                                "axis": {"titleFontSize": 12, "labelFontSize": 10},
                            },
                            "y": {
                                "field": "Feature2",
                                "type": "quantitative",
                                "title": f"{feature2}",
                                "axis": {"titleFontSize": 12, "labelFontSize": 10},
                            },
                            "color": {
                                "field": "SHAP Value",
                                "type": "quantitative",
                                "title": f"SHAP({feature1})",
                                "scale": {
                                    "scheme": "viridis",
                                    "domain": {"expr": "[colorMin, colorMax]"},
                                    "clamp": True,
                                },
                                "legend": None,
                            },
                            "tooltip": [
                                {
                                    "field": "Feature1",
                                    "type": "quantitative",
                                    "title": feature1,
                                },
                                {
                                    "field": "Feature2",
                                    "type": "quantitative",
                                    "title": feature2,
                                    "format": ".3f",
                                },
                                {
                                    "field": "SHAP Value",
                                    "type": "quantitative",
                                    "title": f"SHAP({feature1})",
                                    "format": ".3f",
                                },
                                {
                                    "field": "Interaction",
                                    "type": "quantitative",
                                    "format": ".3f",
                                },
                            ],
                        },
                        "selection": {"grid": {"type": "interval", "bind": "scales"}},
                        "config": {
                            "view": {"stroke": None},
                            "background": bg_color,
                            "axis": self._get_chart_config(),
                            "title": {"color": self._get_chart_config()["titleColor"]},
                        },
                    }

                    spec = self._add_lowess_layer(
                        spec,
                        "Feature1",
                        "Feature2",
                        plot_df["Feature1"].values,
                        plot_df["Feature2"].values,
                    )
                    spec = self._apply_fixed_range_to_spec(spec)

                    self.plot_data = {
                        "spec": spec,
                        "feature_type": "continuous_continuous",
                        "plot_mode": "feature_scatter",
                        "main_feature": feature1,
                        "color_feature": f"SHAP({feature1})",
                        "color_min": shap_min,
                        "color_max": shap_max,
                        "manual_range_supported": True,
                        "manual_range_enabled": self.plot_manual_range_enabled,
                        "manual_range": self.plot_manual_range,
                    }

            # Case B: categorical vs categorical
            elif feature1_type == "categorical" and feature2_type == "categorical":
                plot_df["Cat1"] = plot_df["Feature1"].astype(str)
                plot_df["Cat2"] = plot_df["Feature2"].astype(str)

                heatmap_data = (
                    plot_df.groupby(["Cat1", "Cat2"])
                    .agg({"Interaction": ["mean", "count"]})
                    .reset_index()
                )
                heatmap_data.columns = ["Cat1", "Cat2", "Mean_Interaction", "Count"]
                heatmap_data_list = heatmap_data.to_dict("records")

                spec = {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "title": {
                        "text": f"SHAP Interaction Heatmap: {feature1} × {feature2}",
                        "fontSize": 14,
                        "anchor": "start",
                    },
                    "description": f"Mean SHAP Interaction: {feature1} × {feature2}",
                    # Width/height will be set dynamically in JS to avoid ResizeObserver issues
                    "data": {"values": heatmap_data_list},
                    "mark": {"type": "rect"},
                    "encoding": {
                        "x": {
                            "field": "Cat1",
                            "type": "nominal",
                            "title": feature1,
                            "axis": {
                                "labelAngle": -45,
                                "titleFontSize": 12,
                                "labelFontSize": 10,
                            },
                        },
                        "y": {
                            "field": "Cat2",
                            "type": "nominal",
                            "title": feature2,
                            "axis": {"titleFontSize": 12, "labelFontSize": 10},
                        },
                        "color": {
                            "field": "Mean_Interaction",
                            "type": "quantitative",
                            "title": "Mean Interaction",
                            "scale": {"scheme": "viridis"},
                            "legend": {"titleFontSize": 12, "labelFontSize": 10},
                        },
                        "tooltip": [
                            {"field": "Cat1", "type": "nominal", "title": feature1},
                            {"field": "Cat2", "type": "nominal", "title": feature2},
                            {
                                "field": "Mean_Interaction",
                                "type": "quantitative",
                                "format": ".3f",
                            },
                            {"field": "Count", "type": "quantitative"},
                        ],
                    },
                    "config": {
                        "view": {"stroke": None},
                        "background": self._get_chart_background(),
                        "axis": self._get_chart_config(),
                        "title": {"color": self._get_chart_config()["titleColor"]},
                    },
                }

                self.plot_data = {
                    "spec": spec,
                    "feature_type": "categorical_categorical",
                    "main_feature": feature1,
                    "color_feature": feature2,
                    "manual_range_supported": False,
                    "manual_range_enabled": self.plot_manual_range_enabled,
                    "manual_range": self.plot_manual_range,
                }

            # Case C: one categorical, one continuous
            else:
                if feature1_type == "categorical":
                    plot_df["Feature1_str"] = plot_df["Feature1"].astype(str)
                    data = plot_df[
                        ["Feature1_str", "SHAP Value", "Feature2", "Interaction"]
                    ].to_dict("records")
                    min_val = float(plot_df["Feature2"].min())
                    max_val = float(plot_df["Feature2"].max())

                    spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "description": f"SHAP values vs {feature1}, colored by {feature2}",
                        # Width/height will be set dynamically in JS to avoid ResizeObserver issues
                        "data": {"values": data},
                        "mark": {"type": "circle", "size": 40, "opacity": 0.7},
                        "params": [
                            {"name": "colorMin", "value": min_val},
                            {"name": "colorMax", "value": max_val},
                        ],
                        "encoding": {
                            "x": {
                                "field": "Feature1_str",
                                "type": "nominal",
                                "title": f"{feature1} (Categories)",
                                "axis": {
                                    "labelAngle": -45,
                                    "titleFontSize": 12,
                                    "labelFontSize": 10,
                                },
                            },
                            "y": {
                                "field": "SHAP Value",
                                "type": "quantitative",
                                "title": f"SHAP Value",
                                "axis": {"titleFontSize": 12, "labelFontSize": 10},
                            },
                            "color": {
                                "field": "Feature2",
                                "type": "quantitative",
                                "title": feature2,
                                "scale": {
                                    "scheme": "viridis",
                                    "domain": {"expr": "[colorMin, colorMax]"},
                                    "clamp": True,
                                },
                                "legend": None,
                            },
                            "tooltip": [
                                {
                                    "field": "Feature1_str",
                                    "type": "nominal",
                                    "title": feature1,
                                },
                                {
                                    "field": "SHAP Value",
                                    "type": "quantitative",
                                    "format": ".3f",
                                },
                                {
                                    "field": "Feature2",
                                    "type": "quantitative",
                                    "title": feature2,
                                    "format": ".3f",
                                },
                                {
                                    "field": "Interaction",
                                    "type": "quantitative",
                                    "format": ".3f",
                                },
                            ],
                        },
                        "selection": {"grid": {"type": "interval", "bind": "scales"}},
                        "config": {
                            "view": {"stroke": None},
                            "background": self._get_chart_background(),
                            "axis": self._get_chart_config(),
                            "title": {"color": self._get_chart_config()["titleColor"]},
                        },
                    }

                    self.plot_data = {
                        "spec": spec,
                        "feature_type": "categorical_continuous",
                        "main_feature": feature1,
                        "color_feature": feature2,
                        "color_min": min_val,
                        "color_max": max_val,
                        "manual_range_supported": False,
                        "manual_range_enabled": self.plot_manual_range_enabled,
                        "manual_range": self.plot_manual_range,
                    }
                else:
                    plot_df["Feature2_str"] = plot_df["Feature2"].astype(str)
                    data = plot_df[
                        ["Feature1", "SHAP Value", "Feature2_str", "Interaction"]
                    ].to_dict("records")

                    spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "description": f"SHAP values vs {feature1}, colored by {feature2}",
                        # Width/height will be set dynamically in JS to avoid ResizeObserver issues
                        "data": {"values": data},
                        "mark": {"type": "circle", "size": 40, "opacity": 0.7},
                        "encoding": {
                            "x": {
                                "field": "Feature1",
                                "type": "quantitative",
                                "title": f"{feature1} (Feature Value)",
                                "axis": {"titleFontSize": 12, "labelFontSize": 10},
                            },
                            "y": {
                                "field": "SHAP Value",
                                "type": "quantitative",
                                "title": f"SHAP Value",
                                "axis": {"titleFontSize": 12, "labelFontSize": 10},
                            },
                            "color": {
                                "field": "Feature2_str",
                                "type": "nominal",
                                "title": feature2,
                                "scale": {"scheme": "category20"},
                                "legend": {
                                    "title": feature2,
                                    "titleFontSize": 12,
                                    "labelFontSize": 10,
                                },
                            },
                            "tooltip": [
                                {
                                    "field": "Feature1",
                                    "type": "quantitative",
                                    "title": feature1,
                                    "format": ".3f",
                                },
                                {
                                    "field": "SHAP Value",
                                    "type": "quantitative",
                                    "format": ".3f",
                                },
                                {
                                    "field": "Feature2_str",
                                    "type": "nominal",
                                    "title": feature2,
                                },
                                {
                                    "field": "Interaction",
                                    "type": "quantitative",
                                    "format": ".3f",
                                },
                            ],
                        },
                        "selection": {"grid": {"type": "interval", "bind": "scales"}},
                        "config": {
                            "view": {"stroke": None},
                            "background": self._get_chart_background(),
                            "axis": self._get_chart_config(),
                            "title": {"color": self._get_chart_config()["titleColor"]},
                        },
                    }
                    spec = self._apply_fixed_range_to_spec(spec)

                    self.plot_data = {
                        "spec": spec,
                        "feature_type": "continuous_categorical",
                        "main_feature": feature1,
                        "color_feature": feature2,
                        "manual_range_supported": True,
                        "manual_range_enabled": self.plot_manual_range_enabled,
                        "manual_range": self.plot_manual_range,
                    }

        else:
            self.plot_data = {}

    def _get_chart_background(self) -> str:
        """
        Just use transparent background to seamlessly integrate with IDE them
        """
        return "transparent"

    def _get_chart_config(self) -> dict:
        """Get chart configuration based on current theme."""
        if self._current_theme == "dark":
            return {
                "gridColor": "#3a3f54",
                "domainColor": "#6c757d",
                "labelColor": "#e4e6eb",
                "titleColor": "#e4e6eb",
            }
        else:
            return {
                "gridColor": "#e5e7eb",
                "domainColor": "#6b7280",
                "labelColor": "#374151",
                "titleColor": "#212529",
            }

    def _on_selection_change_py(self, change: Dict[str, Any]) -> None:  # noqa: ARG002
        """Hook for additional Python-side logic when selection changes."""
        pass

    def _on_plot_mode_change(self, change: Dict[str, Any]) -> None:  # noqa: ARG002
        """When plot mode changes, regenerate plot data for current selection."""
        if self._last_displayed_selection and len(self._last_displayed_selection) == 2:
            # Regenerate plot with new mode
            self._prepare_plot_data(list(self._last_displayed_selection))

    def _on_plot_smooth_change(self, change: Dict[str, Any]) -> None:  # noqa: ARG002
        """When smoothing toggles, regenerate plot data for current selection."""
        if self._last_displayed_selection:
            self._prepare_plot_data(list(self._last_displayed_selection))
