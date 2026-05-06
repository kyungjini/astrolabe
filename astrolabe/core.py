from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor
import shap

from .network import build_network, percentile_rank


def _merge_xgb_kwargs(
    task: str, random_state: int, user: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(random_state=random_state)
    if task == "classification":
        merged["use_label_encoder"] = False
        merged["eval_metric"] = "logloss"
    merged.update(user or {})
    return merged


# Global configuration for edge classification thresholds (percentile-based)
class EdgeConfig:
    """Configuration for edge classification thresholds.

    All thresholds are percentile-based (0.0 to 1.0):
    - redundant: Top % for high correlation (collinearity)
    - model_driven: Top % for high interaction scores
    - nonlinear_mi: Top % for mutual information in non-linear detection
    - nonlinear_corr: Maximum correlation percentile for non-linear edges
    """

    def __init__(
        self,
        redundant: float = 0.9,
        model_driven: float = 0.97,
        nonlinear_mi: float = 0.75,
        nonlinear_corr: float = 0.4,
    ):
        self.redundant = redundant
        self.model_driven = model_driven
        self.nonlinear_mi = nonlinear_mi
        self.nonlinear_corr = nonlinear_corr

    def to_dict(self) -> Dict[str, float]:
        """Convert config to dictionary for serialization."""
        return {
            "redundant": self.redundant,
            "model_driven": self.model_driven,
            "nonlinear_mi": self.nonlinear_mi,
            "nonlinear_corr": self.nonlinear_corr,
        }

    def update(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{key} must be between 0.0 and 1.0, got {value}")
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    def __repr__(self) -> str:
        """String representation for display."""
        return (
            f"Edge Classification Config:\n"
            f"  Redundant:      > {self.redundant:.2f} (top {(1-self.redundant)*100:.0f}%)\n"
            f"  Model-driven:   ≥ {self.model_driven:.2f} (top {(1-self.model_driven)*100:.0f}%)\n"
            f"  Non-linear MI:  ≥ {self.nonlinear_mi:.2f} (top {(1-self.nonlinear_mi)*100:.0f}%)\n"
            f"  Non-linear corr: < {self.nonlinear_corr:.2f} (bottom {(1-self.nonlinear_corr)*100:.0f}%)"
        )


# Global edge configuration instance
_EDGE_CONFIG = EdgeConfig()


def config(
    redundant: Optional[float] = None,
    model_driven: Optional[float] = None,
    nonlinear_mi: Optional[float] = None,
    nonlinear_corr: Optional[float] = None,
    reset: bool = False,
) -> EdgeConfig:
    """Configure edge classification thresholds globally.

    Parameters
    ----------
    redundant : float, optional
        Percentile threshold for redundant edges (high correlation).
        Default: 0.9 (top 10%)
    model_driven : float, optional
        Percentile threshold for model-driven edges (high interaction).
        Default: 0.97 (top 3%)
    nonlinear_mi : float, optional
        Percentile threshold for MI in non-linear edge detection.
        Default: 0.75 (top 25%)
    nonlinear_corr : float, optional
        Maximum correlation percentile for non-linear edges.
        Default: 0.4 (bottom 60%)
    reset : bool, optional
        Reset to default values. Default: False

    Returns
    -------
    EdgeConfig
        Current configuration object

    Examples
    --------
    >>> import astrolabe
    >>> # Make classification more strict (top 5% only)
    >>> astrolabe.config(redundant=0.95, model_driven=0.95)
    >>> # Make it more lenient (top 20%)
    >>> astrolabe.config(redundant=0.8, model_driven=0.8)
    >>> # Reset to defaults
    >>> astrolabe.config(reset=True)
    """
    global _EDGE_CONFIG

    if reset:
        _EDGE_CONFIG = EdgeConfig()
        return _EDGE_CONFIG

    kwargs = {}
    if redundant is not None:
        kwargs["redundant"] = redundant
    if model_driven is not None:
        kwargs["model_driven"] = model_driven
    if nonlinear_mi is not None:
        kwargs["nonlinear_mi"] = nonlinear_mi
    if nonlinear_corr is not None:
        kwargs["nonlinear_corr"] = nonlinear_corr

    if kwargs:
        _EDGE_CONFIG.update(**kwargs)

    return _EDGE_CONFIG


@dataclass
class Astrolabe:
    """Core analysis object for Astrolabe.

    This class encapsulates:
    - Model fitting (XGBoost)
    - SHAP values and SHAP interactions
    - Optimized network construction (nodes + links)
    - Linear vs ML comparison via linear coefficients
    - Ablation study: testing feature importance with redundant features removed
    """

    df_X: pd.DataFrame
    df_y: Union[pd.Series, np.ndarray]
    task: str = "regression"  # "regression" or "classification"
    random_state: int = 42
    # Ratio-based thresholds (0.0 to 1.0) for edge inclusion
    # Edges are included if: correlation >= max_corr * min_corr_ratio OR interaction >= max_interaction * min_interaction_ratio
    min_corr_ratio: float = 0.0  # 0.0 = include all (filter by interaction only)
    min_interaction_ratio: float = 0.05  # 0.05 = 5% of max interaction score
    xgb_kwargs: Dict[str, Any] = field(default_factory=dict)
    _custom_model: Optional[Any] = field(default=None)
    ablation: bool = False  # Enable ablation study
    ablation_top_n: int = 5  # Number of top features to test in ablation
    # Column names to ablate in addition to (or instead of) the automatic redundant-edge top-N.
    # Order is preserved; unknown names are skipped with a warning.
    ablation_extra_features: list[str] = field(default_factory=list)

    # Will be populated during initialization
    model: Any = field(init=False)
    shap_values: np.ndarray = field(init=False)
    shap_interaction_values: np.ndarray = field(init=False)
    feature_names: list[str] = field(init=False)
    feature_importance: np.ndarray = field(init=False)
    feature_importance_normalized: np.ndarray = field(init=False)
    linear_coef_dict: Dict[str, float] = field(init=False)
    network_data: Dict[str, Any] = field(init=False)
    feature_metadata: Dict[str, Dict[str, Any]] = field(init=False)
    ablation_results_: Dict[str, Dict[str, Any]] = field(
        init=False, default_factory=dict
    )
    ablation_candidates_: list[str] = field(init=False, default_factory=list)
    multicollinearity_warnings_: list[Dict[str, Any]] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        if not isinstance(self.df_X, pd.DataFrame):
            raise TypeError("df_X must be a pandas DataFrame")

        # Coerce y to 1D numpy array
        if isinstance(self.df_y, (pd.Series, pd.DataFrame)):
            self.df_y = np.asarray(self.df_y).ravel()
        else:
            self.df_y = np.asarray(self.df_y).ravel()

        # Basic NA / inf filtering (match calculate_data.ipynb behavior)
        # Drop any rows where y is NaN/inf or X has NaN/inf
        y_mask = np.isfinite(self.df_y)
        X_values = self.df_X.to_numpy()
        X_mask = np.all(np.isfinite(X_values), axis=1)
        mask = y_mask & X_mask

        if not np.all(mask):
            # Apply mask to both X and y
            self.df_X = self.df_X.loc[mask].reset_index(drop=True)
            self.df_y = self.df_y[mask]

        if self.df_X.shape[0] == 0:
            raise ValueError("No valid rows left after dropping NaN/inf in X or y.")

        self.task = self.task.lower()
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")

        self.feature_names = list(self.df_X.columns)

        # Run full pipeline
        if self._custom_model is not None:
            # User provided a custom model, skip training
            self._load_custom_model()
        else:
            # Train a new XGBoost model
            self._fit_model()
        self._compute_shap()
        self._compute_linear_coefficients()
        self._compute_feature_metadata()
        self._compute_network()

        # Run ablation study if enabled
        if self.ablation:
            self._run_ablation_study()

    def set_xgb_params(self, **kwargs: Any) -> None:
        """Update XGBoost hyperparameters to be used for model creation.

        Examples
        --------
        >>> a = Astrolabe(df_X, df_y)
        >>> a.set_xgb_params(n_estimators=200, max_depth=4)
        >>> # Alternatively, pass xgb_kwargs at construction time
        >>> a2 = Astrolabe(df_X, df_y, xgb_kwargs={"n_estimators":200, "max_depth":4})

        The provided keyword arguments are merged into the existing `xgb_kwargs`
        dictionary and will override XGBoost defaults when the model is created or
        when ablations re-fit models.
        """
        if not kwargs:
            return
        # Basic validation: keys must be str
        for k in kwargs.keys():
            if not isinstance(k, str):
                raise TypeError("xgb parameter names must be strings")
        # Merge overrides
        self.xgb_kwargs.update(kwargs)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def _make_xgb_model(self) -> Any:
        """Create an XGBoost model with library defaults, merged with ``xgb_kwargs``."""
        kw = _merge_xgb_kwargs(self.task, self.random_state, self.xgb_kwargs)
        if self.task == "regression":
            return XGBRegressor(**kw)
        return XGBClassifier(**kw)

    def _fit_model(self) -> None:
        print(f"Fitting XGBoost {self.task} model...")
        print(f"Data shape: {self.df_X.shape}")
        self.model = self._make_xgb_model()
        # XGBoost 3.x sklearn estimators record pandas metadata (e.g. feature_names_in_)
        # when fit with a DataFrame. SHAP then clones/patches the estimator and can hit
        # AttributeError on read-only fitted attributes. Fitting on a NumPy array avoids
        # that path while keeping column order aligned with self.feature_names.
        X_fit = np.asarray(self.df_X)
        self.model.fit(X_fit, self.df_y)
        print("XGBoost model training completed.")
        # Feature importance will be computed from SHAP values after _compute_shap()
        # Store a placeholder for now
        self.feature_importance = None
        self.feature_importance_normalized = None

    def _load_custom_model(self) -> None:
        """Load a pre-trained model provided by the user."""
        print(f"Loading custom {self.task} model...")
        print(f"Data shape: {self.df_X.shape}")
        self.model = self._custom_model
        print("Model loaded.")
        # Feature importance will be computed from SHAP values after _compute_shap()
        # Store a placeholder for now
        self.feature_importance = None
        self.feature_importance_normalized = None

    def _compute_shap(self) -> None:
        """Compute SHAP main effects and interaction values.

        Normalizes shapes so that:
        - self.shap_values: (n_samples, n_features)
        - self.shap_interaction_values: (n_samples, n_features, n_features)
        """
        print("Computing SHAP values and interactions...")
        print(f"Data shape: {self.df_X.shape}")

        # Choose appropriate SHAP explainer based on model type
        explainer = self._get_shap_explainer()

        # Main effects
        print("Computing SHAP main effects...")
        raw_values = explainer.shap_values(self.df_X)
        values = np.asarray(raw_values)
        if isinstance(raw_values, list):
            # (n_classes, n_samples, n_features) -> sum over classes
            values = np.stack(raw_values, axis=0).sum(axis=0)
        if values.ndim == 3:
            # (n_classes, n_samples, n_features)
            values = values.sum(axis=0)
        if values.ndim != 2:
            raise RuntimeError(f"Unexpected SHAP values shape: {values.shape}")
        self.shap_values = values
        print(f"SHAP values shape: {self.shap_values.shape}")

        # Compute feature importance from SHAP values (mean absolute SHAP)
        self.feature_importance = np.mean(np.abs(self.shap_values), axis=0)
        max_imp = (
            float(self.feature_importance.max())
            if self.feature_importance.size
            else 0.0
        )
        if max_imp > 0:
            self.feature_importance_normalized = self.feature_importance / max_imp
        else:
            self.feature_importance_normalized = self.feature_importance.copy()
        print(f"SHAP-based feature importance shape: {self.feature_importance.shape}")
        print(
            f"SHAP-based feature importance range: [{self.feature_importance.min():.4f}, {self.feature_importance.max():.4f}]"
        )

        # Interactions
        print("Computing SHAP interaction values...")
        try:
            raw_int = explainer.shap_interaction_values(self.df_X)
            inter = np.asarray(raw_int)
            if isinstance(raw_int, list):
                inter = np.stack(raw_int, axis=0).sum(axis=0)
            if inter.ndim == 4:
                # (n_classes, n_samples, n_features, n_features)
                inter = inter.sum(axis=0)
            if inter.ndim != 3:
                raise RuntimeError(f"Unexpected SHAP interaction shape: {inter.shape}")
            self.shap_interaction_values = inter
            print(f"SHAP interactions shape: {self.shap_interaction_values.shape}")
        except (AttributeError, NotImplementedError) as e:
            # Some explainers don't support interaction values (e.g., KernelExplainer)
            print(
                f"Warning: SHAP interaction values not available for this model type."
            )
            print(f"Creating zero interaction matrix as fallback.")
            n_samples, n_features = self.shap_values.shape
            self.shap_interaction_values = np.zeros((n_samples, n_features, n_features))

    def _get_shap_explainer(self):
        """Get the appropriate SHAP explainer for the model type."""
        model_type = type(self.model).__name__
        print(f"Model type: {model_type}")

        # Check if model is tree-based
        tree_models = (
            "XGBRegressor",
            "XGBClassifier",
            "RandomForestRegressor",
            "RandomForestClassifier",
            "GradientBoostingRegressor",
            "GradientBoostingClassifier",
            "LGBMRegressor",
            "LGBMClassifier",
            "ExtraTreesRegressor",
            "ExtraTreesClassifier",
        )

        # Try TreeExplainer for tree-based models
        if model_type in tree_models or hasattr(self.model, "tree_"):
            try:
                return shap.TreeExplainer(self.model)
            except Exception as e:
                pass  # Fall through to KernelExplainer

        # For neural networks and other models, use KernelExplainer
        # Use a subset of data as background for efficiency
        background_size = min(100, len(self.df_X))
        background = shap.sample(self.df_X, background_size)
        return shap.KernelExplainer(self.model.predict, background)

    def _compute_linear_coefficients(self) -> None:
        """Fit a simple linear model on scaled features.

        Used to compare a linear baseline vs the ML model.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df_X)

        if self.task == "regression":
            lin_model: Any = LinearRegression()
        else:
            lin_model = LogisticRegression(max_iter=1000, n_jobs=None)

        lin_model.fit(X_scaled, self.df_y)

        coefs = np.asarray(lin_model.coef_)
        if coefs.ndim == 2:
            # multi-class: average magnitude across classes
            coefs = coefs.mean(axis=0)
        self.linear_coef_dict = {
            self.feature_names[i]: float(coefs[i])
            for i in range(len(self.feature_names))
        }

    def _compute_feature_metadata(self) -> None:
        """Classify each feature as categorical or continuous.

        Rules match the existing notebook logic (<=10 unique values => categorical).
        """
        meta: Dict[str, Dict[str, Any]] = {}
        for name in self.feature_names:
            series = self.df_X[name].dropna()
            unique_vals = series.unique()
            n_unique = len(unique_vals)
            if n_unique <= 10:
                meta[name] = {
                    "type": "categorical",
                    "values": sorted(
                        [
                            (
                                float(v)
                                if isinstance(v, (int, float, np.number))
                                else str(v)
                            )
                            for v in unique_vals
                        ]
                    ),
                }
            else:
                meta[name] = {
                    "type": "continuous",
                    "min": float(series.min()) if len(series) else 0.0,
                    "max": float(series.max()) if len(series) else 0.0,
                }
        self.feature_metadata = meta

    def _compute_network(self) -> None:
        """Build nodes and links (correlations, SHAP interactions, MI, empirical percentiles)."""
        print("Computing network...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            net_data, multicollinearity_warnings = build_network(
                self.df_X,
                self.feature_names,
                self.shap_values,
                self.shap_interaction_values,
                self.linear_coef_dict,
                random_state=self.random_state,
                min_corr_ratio=self.min_corr_ratio,
                min_interaction_ratio=self.min_interaction_ratio,
                collect_multicollinearity=True,
                use_tqdm=True,
            )
        self.network_data = net_data
        self.multicollinearity_warnings_ = multicollinearity_warnings

        if multicollinearity_warnings:
            print(
                f"⚠ {len(multicollinearity_warnings)} high-correlation pairs detected"
            )

    def _calculate_percentile(self, value: float, all_values: list) -> float:
        return percentile_rank(value, all_values)

    def _run_ablation_study(self) -> None:
        """Perform ablation study on features with redundant edges.

        For each selected feature:
        1. Identify features with redundant edges
        2. Select top N by SHAP importance
        3. Append any ``ablation_extra_features`` (manual targets, e.g. a specific lab value)
        4. Re-run analysis without each feature
        5. Store results for comparison
        """
        print("\nAblation study:")

        col_set = set(self.feature_names)
        extras_ordered: list[str] = []
        for name in self.ablation_extra_features or []:
            if name in col_set:
                extras_ordered.append(name)
            else:
                pass  # Skip silently if not found

        # Step 1: Identify features with redundant edges (based on Pearson correlation >= 0.7)
        features_with_redundant = set()

        for link in self.network_data["links"]:
            # Use Pearson correlation for multicollinearity detection
            pearson_r = abs(link.get("corr_pearson", 0))
            if pearson_r >= 0.7:
                features_with_redundant.add(link["source"])
                features_with_redundant.add(link["target"])

        if not features_with_redundant and not extras_ordered:
            self.ablation_candidates_ = []
            self.ablation_results_ = {}
            return

        ranked: list[tuple[str, float]] = []
        if features_with_redundant:

            # Step 2: Select top N by SHAP importance among redundant-edge participants
            importance_dict = {
                node["id"]: node["importance"]
                for node in self.network_data["nodes"]
                if node["id"] in features_with_redundant
            }

            ranked = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )[: max(0, self.ablation_top_n)]

        auto_candidates = [fname for fname, _ in ranked]
        seen: set[str] = set(auto_candidates)
        merged = list(auto_candidates)
        for fname in extras_ordered:
            if fname not in seen:
                merged.append(fname)
                seen.add(fname)

        self.ablation_candidates_ = merged

        if not self.ablation_candidates_:
            self.ablation_results_ = {}
            return

        # Step 3: Run analysis for each candidate

        for feature_to_remove in tqdm(
            self.ablation_candidates_, desc="Ablation experiments"
        ):
            # Create dataset without this feature
            remaining_features = [
                f for f in self.feature_names if f != feature_to_remove
            ]
            df_X_ablated = self.df_X[remaining_features].copy()

            # Re-run core analysis pipeline
            try:
                kw = _merge_xgb_kwargs(self.task, self.random_state, self.xgb_kwargs)
                model_ablated = (
                    XGBRegressor(**kw)
                    if self.task == "regression"
                    else XGBClassifier(**kw)
                )

                model_ablated.fit(df_X_ablated.values, self.df_y)

                explainer = shap.TreeExplainer(model_ablated)
                shap_values_ablated = explainer.shap_values(df_X_ablated.values)

                if self.task == "classification" and isinstance(
                    shap_values_ablated, list
                ):
                    shap_values_ablated = shap_values_ablated[1]

                shap_interaction_values_ablated = explainer.shap_interaction_values(
                    df_X_ablated.values
                )

                if self.task == "regression":
                    linear_model = LinearRegression()
                else:
                    linear_model = LogisticRegression(
                        max_iter=1000, random_state=self.random_state
                    )

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_X_ablated.values)
                linear_model.fit(X_scaled, self.df_y)

                coefs = np.asarray(linear_model.coef_)
                if coefs.ndim == 2:
                    coefs = coefs.mean(axis=0)
                linear_coef_dict_ablated = {
                    remaining_features[i]: float(coefs[i])
                    for i in range(len(remaining_features))
                }

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    net_ablated, _ = build_network(
                        df_X_ablated,
                        remaining_features,
                        shap_values_ablated,
                        shap_interaction_values_ablated,
                        linear_coef_dict_ablated,
                        random_state=self.random_state,
                        min_corr_ratio=self.min_corr_ratio,
                        min_interaction_ratio=self.min_interaction_ratio,
                        collect_multicollinearity=False,
                        use_tqdm=False,
                    )

                self.ablation_results_[feature_to_remove] = {
                    "nodes": net_ablated["nodes"],
                    "links": net_ablated["links"],
                    "shap_values": shap_values_ablated,
                    "shap_interaction_values": shap_interaction_values_ablated,
                    "feature_names": remaining_features,
                }

            except Exception as e:
                print(f"\nWarning: Ablation failed for {feature_to_remove}: {str(e)}")
                self.ablation_results_[feature_to_remove] = None

        print(
            f"\nAblation study complete: {len(self.ablation_results_)} experiments run"
        )
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Status and summary
    # ------------------------------------------------------------------
    def status(self) -> None:
        """Display comprehensive statistics about the Astrolabe object.

        Shows:
        - Feature importance range and distribution
        - Interaction score range and distribution
        - Edge relationship counts (model-driven, redundant, non-linear)
        - Network connectivity statistics
        """
        print("=" * 70)
        print("ASTROLABE STATUS REPORT")
        print("=" * 70)

        # Model information
        print(f"\n{'MODEL INFORMATION':-^70}")
        print(f"  Task type:              {self.task}")
        print(f"  Model type:             {type(self.model).__name__}")
        print(f"  Number of features:     {len(self.feature_names)}")
        print(f"  Number of samples:      {len(self.df_X)}")

        # Feature importance statistics
        print(f"\n{'FEATURE IMPORTANCE (SHAP-based)':-^70}")
        importance_values = [node["importance"] for node in self.network_data["nodes"]]
        if importance_values:
            print(
                f"  Range:                  [{min(importance_values):.4f}, {max(importance_values):.4f}]"
            )
            print(f"  Mean:                   {np.mean(importance_values):.4f}")
            print(f"  Median:                 {np.median(importance_values):.4f}")
            print(f"  Std Dev:                {np.std(importance_values):.4f}")

            # Top 5 features
            sorted_nodes = sorted(
                self.network_data["nodes"], key=lambda x: x["importance"], reverse=True
            )
            print(f"\n  Top 5 features:")
            for i, node in enumerate(sorted_nodes[:5], 1):
                print(f"    {i}. {node['id']:<20} importance: {node['importance']:.4f}")

        # Interaction score statistics
        print(f"\n{'INTERACTION SCORES':-^70}")
        links = self.network_data["links"]
        if links:
            interaction_scores = [link["importance"] for link in links]
            print(f"  Total edges:            {len(links)}")
            print(
                f"  Range:                  [{min(interaction_scores):.4f}, {max(interaction_scores):.4f}]"
            )
            print(f"  Mean:                   {np.mean(interaction_scores):.4f}")
            print(f"  Median:                 {np.median(interaction_scores):.4f}")
            print(f"  Std Dev:                {np.std(interaction_scores):.4f}")
        else:
            print(f"  No edges in network")

        # Edge relationship analysis
        if links:
            print(f"\n{'EDGE RELATIONSHIP ANALYSIS (Percentile-based)':-^70}")

            # Use global config for thresholds
            cfg = _EDGE_CONFIG
            model_driven = []
            redundant = []
            nonlinear = []

            for link in links:
                corr_pct = link.get("corr_percentile", 0)
                pearson_r = abs(link.get("corr_pearson", 0))
                mi_pct = link.get("mi_percentile", 0)
                interaction_pct = link.get("interaction_percentile", 0)

                # Redundant: Pearson correlation >= 0.7 (multicollinearity threshold)
                if pearson_r >= 0.7:
                    redundant.append(link)

                # Model-driven: High interaction based on config threshold
                if interaction_pct >= cfg.model_driven:
                    model_driven.append(link)

                # Non-linear: High MI but low correlation based on config thresholds
                if mi_pct >= cfg.nonlinear_mi and corr_pct < cfg.nonlinear_corr:
                    nonlinear.append(link)

            print(
                f"  Model-driven interactions:  {len(model_driven):>4} edges (top {(1-cfg.model_driven)*100:.0f}%)"
            )
            if model_driven:
                # Show top 3
                sorted_md = sorted(
                    model_driven, key=lambda x: x["importance"], reverse=True
                )[:3]
                for link in sorted_md:
                    print(
                        f"    • {link['source']} ↔ {link['target']} (score: {link['importance']:.4f})"
                    )

            print(
                f"\n  Redundant relationships:    {len(redundant):>4} edges (|Pearson r| >= 0.7)"
            )
            if redundant:
                # Show top 3
                sorted_red = sorted(
                    redundant, key=lambda x: abs(x["corr_pearson"]), reverse=True
                )[:3]
                for link in sorted_red:
                    print(
                        f"    • {link['source']} ↔ {link['target']} (Pearson: {link['corr_pearson']:.4f}, Spearman: {link['corr_spearman']:.4f})"
                    )

            print(
                f"\n  Non-linear relationships:   {len(nonlinear):>4} edges (top {(1-cfg.nonlinear_mi)*100:.0f}% MI, <{cfg.nonlinear_corr*100:.0f}% corr)"
            )
            if nonlinear:
                # Show top 3
                sorted_nl = sorted(
                    nonlinear, key=lambda x: x["mutual_information"], reverse=True
                )[:3]
                for link in sorted_nl:
                    print(
                        f"    • {link['source']} ↔ {link['target']} (MI: {link['mutual_information']:.4f}, MI_pct: {link['mi_percentile']:.2f}, Spearman: {link['corr_spearman']:.4f})"
                    )

        # Network connectivity
        print(f"\n{'NETWORK CONNECTIVITY':-^70}")
        if links:
            # Calculate degree distribution
            degree_count = {}
            for link in links:
                for node_id in [link["source"], link["target"]]:
                    degree_count[node_id] = degree_count.get(node_id, 0) + 1

            degrees = list(degree_count.values())
            isolated_nodes = len(self.feature_names) - len(degree_count)

            print(f"  Connected features:     {len(degree_count)}")
            print(f"  Isolated features:      {isolated_nodes}")
            print(f"  Average degree:         {np.mean(degrees):.2f}")
            print(f"  Max degree:             {max(degrees)}")

            # Most connected features
            sorted_by_degree = sorted(
                degree_count.items(), key=lambda x: x[1], reverse=True
            )[:5]
            print(f"\n  Most connected features:")
            for i, (feature, degree) in enumerate(sorted_by_degree, 1):
                print(f"    {i}. {feature:<20} connections: {degree}")
        else:
            print(f"  No connections in network")

        print("\n" + "=" * 70)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: Union[str, "pathlib.Path"]) -> None:
        """Serialize this Astrolabe object to disk using joblib."""
        joblib.dump(self, path)

    # ------------------------------------------------------------------
    # Widget integration
    # ------------------------------------------------------------------
    def explore(self):
        """Return an interactive widget layout for exploration.

        This constructs the AnyWidget-based network view with integrated
        vega-lite charts and returns it. In a notebook, you can simply call::

            astro.explore()

        and Jupyter will display the returned widget once.
        """
        from .widget import AstrolabeWidget

        widget = AstrolabeWidget(self)
        return widget

    def plot(
        self,
        mode: str,
        x: str,
        y: Optional[str] = None,
        color: Optional[str] = None,
        ablation: Optional[str] = None,
        smooth: bool = False,
        smooth_frac: float = 0.3,
        fixed_range: Optional[list[list[float]]] = None,
    ):
        """Return a plot-only widget helper.

        This is a convenience wrapper around ``AstrolabeWidget.prepare_explicit_plot``.
        For full analysis navigation, prefer ``explore()``.

        Parameters
        ----------
        mode : str
            ``SHAPdependence`` or ``FeatureScatter``.
        x : str
            Primary feature for the x-axis.
        y : str, optional
            Secondary feature for feature-scatter mode.
        color : str, optional
            Optional color feature for SHAP dependence mode.
        ablation : str, optional
            Name of an ablated feature state to visualize (e.g., ``"HE_glu"``).
            If omitted, the original (non-ablated) state is used.
        smooth : bool, default False
            If True, overlay a LOWESS trend line on continuous scatter plots.
        smooth_frac : float, default 0.3
            Fraction of points used in each LOWESS local window.
        fixed_range : list[list[float]], optional
            Fixed axis ranges as ``[[x_min, x_max], [y_min, y_max]]``.
            When set, zoom/pan via scroll is disabled and the chart uses
            the provided axis domains.
        """
        from .widget import AstrolabeWidget

        widget = AstrolabeWidget(self, plot_only=True)
        widget.prepare_explicit_plot(
            mode=mode,
            x=x,
            y=y,
            color=color,
            ablation=ablation,
            smooth=smooth,
            smooth_frac=smooth_frac,
            fixed_range=fixed_range,
        )
        return widget

    def show_multicollinearity(self, threshold: float = 0.7) -> pd.DataFrame:
        """Display feature pairs with potential multicollinearity issues.

        Parameters
        ----------
        threshold : float, default 0.7
            Absolute Pearson correlation threshold for warning.

        Returns
        -------
        pd.DataFrame
            Table of feature pairs with |Pearson r| >= threshold
        """
        if not hasattr(self, "multicollinearity_warnings_"):
            print("No multicollinearity data available. Run network computation first.")
            return pd.DataFrame()

        warnings = [
            w
            for w in self.multicollinearity_warnings_
            if abs(w["pearson_r"]) >= threshold
        ]

        if not warnings:
            print(f"No multicollinearity issues found (|Pearson r| >= {threshold})")
            return pd.DataFrame()

        df = pd.DataFrame(warnings)
        df = df.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)

        print(f"WARNING: Found {len(df)} feature pairs with |Pearson r| >= {threshold}")
        print("Recommendation: Consider removing one feature from each pair or using")
        print("regularization techniques to handle multicollinearity.\n")

        return df


def load(path: Union[str, "pathlib.Path"]) -> Astrolabe:
    """Load a previously-saved Astrolabe object from disk."""
    obj = joblib.load(path)
    if not isinstance(obj, Astrolabe):
        raise TypeError("Loaded object is not an Astrolabe instance")
    return obj


def load_model(
    df_X: pd.DataFrame,
    df_y: Union[pd.Series, np.ndarray],
    task: str = "regression",
    model: Any = None,
    **kwargs: Any,
) -> Astrolabe:
    """Create an Astrolabe instance with a pre-trained model.

    Parameters
    ----------
    df_X : pd.DataFrame
        Feature matrix.
    df_y : array-like
        Target values.
    task : {"regression", "classification"}, default "regression"
        Type of prediction task.
    model : Any
        Pre-trained model (e.g., XGBoost, sklearn model with predict method).
        The model should be compatible with SHAP TreeExplainer or other SHAP explainers.
    **kwargs : dict
        Additional keyword arguments forwarded to :class:`Astrolabe`.

    Returns
    -------
    Astrolabe
        An Astrolabe instance using the provided model.

    Examples
    --------
    >>> import joblib
    >>> from xgboost import XGBRegressor
    >>> # Train and save a model
    >>> model = XGBRegressor()
    >>> model.fit(X_train, y_train)
    >>> joblib.dump(model, 'model/my_model.astrolabe')
    >>> # Load the model and create Astrolabe
    >>> loaded_model = joblib.load('model/my_model.astrolabe')
    >>> astro = astrolabe.load_model(X_test, y_test, task="regression", model=loaded_model)
    """
    if model is None:
        raise ValueError("model parameter is required for load_model()")

    return Astrolabe(df_X=df_X, df_y=df_y, task=task, _custom_model=model, **kwargs)
