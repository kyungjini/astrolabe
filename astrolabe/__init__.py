from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Union

from .core import Astrolabe, load as _load, load_model as _load_model, config as _config

try:
    __version__ = version("astrolabe")
except PackageNotFoundError:
    __version__ = "0.1.0"


def model(df_X, df_y, task: str = "regression", **kwargs: Any) -> Astrolabe:
    """Factory function returning an :class:`Astrolabe` instance.

    Parameters
    ----------
    df_X : pandas.DataFrame
        Feature matrix.
    df_y : array-like
        Target values.
    task : {"regression", "classification"}, default "regression"
        Type of prediction task.
    **kwargs : dict
        Additional keyword arguments forwarded to :class:`Astrolabe`.
    """

    return Astrolabe(df_X=df_X, df_y=df_y, task=task, **kwargs)


def load(path: Union[str, "pathlib.Path"]):  # type: ignore[name-defined]
    """Load a saved :class:`Astrolabe` instance from disk."""

    return _load(path)


def load_model(
    df_X, df_y, task: str = "regression", model=None, **kwargs: Any
) -> Astrolabe:
    """Create an Astrolabe instance with a pre-trained model.

    Parameters
    ----------
    df_X : pandas.DataFrame
        Feature matrix.
    df_y : array-like
        Target values.
    task : {"regression", "classification"}, default "regression"
        Type of prediction task.
    model : Any
        Pre-trained model (e.g., XGBoost, sklearn model with predict method).
    **kwargs : dict
        Additional keyword arguments forwarded to :class:`Astrolabe`.

    Returns
    -------
    Astrolabe
        An Astrolabe instance using the provided model.
    """

    return _load_model(df_X=df_X, df_y=df_y, task=task, model=model, **kwargs)


def config(
    redundant: Union[float, None] = None,
    model_driven: Union[float, None] = None,
    nonlinear_mi: Union[float, None] = None,
    nonlinear_corr: Union[float, None] = None,
    reset: bool = False,
):
    """Configure edge classification thresholds globally.

    All thresholds are percentile-based (0.0 to 1.0).
    Higher values mean more strict criteria (fewer edges classified).

    Parameters
    ----------
    redundant : float, optional
        Percentile threshold for redundant edges (high correlation).
        Default: 0.9 (top 10% = strict). Lower values = more lenient.
    model_driven : float, optional
        Percentile threshold for model-driven edges (high interaction).
        Default: 0.97 (stricter). Lower values = more lenient.
    nonlinear_mi : float, optional
        Percentile threshold for MI in non-linear edge detection.
        Default: 0.75 (top 25% = moderate). Lower values = more lenient.
    nonlinear_corr : float, optional
        Maximum correlation percentile for non-linear edges.
        Default: 0.4 (bottom 60% = moderate). Higher values = more lenient.
    reset : bool, optional
        Reset all thresholds to defaults. Default: False

    Returns
    -------
    EdgeConfig
        Current configuration object

    Examples
    --------
    >>> import astrolabe
    >>> # More strict: only top 5% classified (fewer badges)
    >>> astrolabe.config(redundant=0.95, model_driven=0.95)
    >>>
    >>> # More lenient: top 20% classified (more badges)
    >>> astrolabe.config(redundant=0.8, model_driven=0.8)
    >>>
    >>> # Reset to defaults
    >>> astrolabe.config(reset=True)
    """
    return _config(
        redundant=redundant,
        model_driven=model_driven,
        nonlinear_mi=nonlinear_mi,
        nonlinear_corr=nonlinear_corr,
        reset=reset,
    )


__all__ = [
    "Astrolabe",
    "__version__",
    "model",
    "load",
    "load_model",
    "config",
]
