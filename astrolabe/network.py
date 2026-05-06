"""Shared feature-network construction (nodes, links, percentiles)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm


def percentile_rank(value: float, all_values: list[float]) -> float:
    if not all_values:
        return 0.0
    sorted_values = sorted(all_values)
    rank = sum(1 for v in sorted_values if v <= value)
    return rank / len(sorted_values)


def build_network(
    df_X: pd.DataFrame,
    feature_names: List[str],
    shap_values: np.ndarray,
    shap_interaction_values: np.ndarray,
    linear_coef_dict: Dict[str, float],
    *,
    random_state: int,
    min_corr_ratio: float,
    min_interaction_ratio: float,
    collect_multicollinearity: bool,
    use_tqdm: bool,
    tqdm_desc_interactions: str = "  • interactions",
    tqdm_desc_links: str = "  • links",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Returns ``({"nodes": ..., "links": ...}, multicollinearity_warnings)``."""
    n_features = len(feature_names)
    corr_matrix_spearman = df_X.corr(method="spearman").to_numpy()
    corr_matrix_pearson = df_X.corr(method="pearson").to_numpy()

    nodes: List[Dict[str, Any]] = []
    for i, fname in enumerate(feature_names):
        feature_values = df_X.iloc[:, i].values
        shap_values_for_feature = shap_values[:, i]
        importance = float(np.mean(np.abs(shap_values_for_feature)))
        feature_series = pd.Series(feature_values, index=df_X.index)
        shap_series = pd.Series(shap_values_for_feature, index=df_X.index)
        direction = feature_series.corr(shap_series)
        if pd.isna(direction):
            direction = 0.0
        linear_coefficient = float(linear_coef_dict.get(fname, 0.0))
        nodes.append(
            {
                "id": fname,
                "importance": float(importance),
                "direction": float(direction),
                "linear_coefficient": linear_coefficient,
            }
        )

    mi_fn = mutual_info_regression
    all_interactions: List[float] = []
    all_correlations: List[float] = []
    interaction_cache: Dict[Tuple[int, int], float] = {}
    interaction_values_cache: Dict[Tuple[int, int], np.ndarray] = {}

    irange = tqdm(range(n_features), desc=tqdm_desc_interactions, leave=False)
    if not use_tqdm:
        irange = range(n_features)

    for i in irange:
        for j in range(i):
            correlation_spearman = (
                float(corr_matrix_spearman[i, j])
                if not pd.isna(corr_matrix_spearman[i, j])
                else 0.0
            )
            interaction_values = shap_interaction_values[:, i, j]
            interaction_importance = float(np.mean(np.abs(interaction_values)))
            main_effect_i = float(np.mean(np.abs(shap_values[:, i])))
            main_effect_j = float(np.mean(np.abs(shap_values[:, j])))
            # Python float only: np.sqrt yields np.float64 which breaks ipywidgets JSON sync.
            gm = math.sqrt(main_effect_i * main_effect_j)
            interaction_importance_normalized = (
                (interaction_importance / gm) if gm > 1e-10 else 0.0
            )
            all_interactions.append(interaction_importance_normalized)
            all_correlations.append(abs(correlation_spearman))
            interaction_cache[(i, j)] = interaction_importance_normalized
            interaction_values_cache[(i, j)] = interaction_values

    if all_interactions:
        max_interaction = max(all_interactions)
        max_corr = max(all_correlations)
        interaction_threshold = max_interaction * min_interaction_ratio
        corr_threshold = (
            max_corr * min_corr_ratio if min_corr_ratio > 0 else 0.0
        )
    else:
        interaction_threshold = 0.0
        corr_threshold = 0.0

    links: List[Dict[str, Any]] = []
    multicollinearity_warnings: List[Dict[str, Any]] = []

    jrange = tqdm(range(n_features), desc=tqdm_desc_links, leave=False)
    if not use_tqdm:
        jrange = range(n_features)

    for i in jrange:
        for j in range(i):
            correlation_spearman = (
                float(corr_matrix_spearman[i, j])
                if not pd.isna(corr_matrix_spearman[i, j])
                else 0.0
            )
            correlation_pearson = (
                float(corr_matrix_pearson[i, j])
                if not pd.isna(corr_matrix_pearson[i, j])
                else 0.0
            )

            if collect_multicollinearity and abs(correlation_pearson) >= 0.7:
                multicollinearity_warnings.append(
                    {
                        "feature_i": feature_names[i],
                        "feature_j": feature_names[j],
                        "pearson_r": correlation_pearson,
                    }
                )

            interaction_importance_normalized = interaction_cache[(i, j)]

            if (
                abs(correlation_spearman) < corr_threshold
                and interaction_importance_normalized < interaction_threshold
            ):
                continue

            mi = mi_fn(
                df_X.iloc[:, i : i + 1].values,
                df_X.iloc[:, j].values,
                random_state=random_state,
                n_neighbors=3,
            )[0]
            mutual_information = float(mi)

            interaction_vals = interaction_values_cache[(i, j)]
            sum_interaction = float(np.sum(interaction_vals))
            sum_abs_interaction = float(np.sum(np.abs(interaction_vals)))
            interaction_direction = (
                sum_interaction / sum_abs_interaction
                if sum_abs_interaction != 0
                else 0.0
            )

            links.append(
                {
                    "source": feature_names[i],
                    "target": feature_names[j],
                    "importance": float(interaction_importance_normalized),
                    "direction": float(interaction_direction),
                    "corr_spearman": float(correlation_spearman),
                    "corr_pearson": float(correlation_pearson),
                    "mutual_information": float(mutual_information),
                    "model_coefficient": float(interaction_importance_normalized),
                }
            )

    if links:
        corr_values = [abs(link["corr_spearman"]) for link in links]
        corr_pearson_values = [abs(link["corr_pearson"]) for link in links]
        mi_values = [link["mutual_information"] for link in links]
        interaction_vals_list = [link["importance"] for link in links]

        for link in links:
            link["corr_percentile"] = float(
                percentile_rank(abs(link["corr_spearman"]), corr_values)
            )
            link["corr_pearson_percentile"] = float(
                percentile_rank(abs(link["corr_pearson"]), corr_pearson_values)
            )
            link["mi_percentile"] = float(
                percentile_rank(link["mutual_information"], mi_values)
            )
            link["interaction_percentile"] = float(
                percentile_rank(link["importance"], interaction_vals_list)
            )

    return {"nodes": nodes, "links": links}, multicollinearity_warnings
