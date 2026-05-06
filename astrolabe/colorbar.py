"""Standalone viridis colorbar PNG (saturation bands + rounded bar; no matplotlib colorbar chrome)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def export_colorbar(
    path: PathLike,
    vmin: float,
    vmax: float,
    *,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    width: int = 48,
    height: int = 320,
    corner_radius: int = 6,
    background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    border_rgb: Tuple[float, float, float] = (0.82, 0.84, 0.88),
    border_alpha: float = 0.35,
    border_width: float = 1.0,
    **ignored: Any,
) -> Path:
    """Save a **viridis** color strip as PNG (no ticks, labels, or Matplotlib ``colorbar`` UI).

    The strip always spans the **full range** ``[vmin, vmax]`` along its long axis:

    - **Vertical:** bottom = ``vmin``, top = ``vmax``.
    - **Horizontal:** left = ``vmin``, right = ``vmax``.

    **Saturation (widget semantics):**

    - Values ``<= lower_threshold`` are drawn at the **minimum** viridis color (bottom of ramp).
    - Values ``>= upper_threshold`` at the **maximum** viridis color (top).
    - Between thresholds, hues follow Matplotlib/Vega-Lite ``viridis`` (same mapping as plots).

    If thresholds are omitted, they default to ``vmin`` / ``vmax``, so the whole bar is one smooth
    gradient without solid end caps.

    Parameters
    ----------
    path :
        Output ``.png`` path.
    vmin, vmax :
        Full value range mapped end-to-end on the bar.
    lower_threshold, upper_threshold :
        Inner domain; outside it, color stays at the min / max viridis endpoint.
    orientation :
        ``\"vertical\"`` or ``\"horizontal\"``.
    width, height :
        Pixel size of the output image (for vertical: ``width`` = bar thickness, ``height`` = length).
    corner_radius :
        Rounded corners in pixels.
    background :
        RGB in ``[0, 1]`` for pixels outside the rounded bar.
    border_rgb, border_alpha, border_width :
        Light anti-aliased outline on the color patch (set ``border_width=0`` to disable).

    Returns
    -------
    pathlib.Path
        Resolved path written to disk.

    Raises
    ------
    ImportError
        If matplotlib is unavailable (viridis LUT).
    ValueError
        On invalid ranges.

    Notes
    -----
    For compatibility, unknown legacy kwargs ``label``, ``dpi``, and ``figsize`` are ignored.
    """
    if ignored.keys() - {"label", "dpi", "figsize"}:
        bad = sorted(ignored.keys() - {"label", "dpi", "figsize"})
        raise TypeError(f"unexpected keyword arguments: {bad!r}")

    cmap = _viridis_lut()

    if not (vmin < vmax):
        raise ValueError(f"vmin must be < vmax, got vmin={vmin}, vmax={vmax}")

    lo = float(lower_threshold if lower_threshold is not None else vmin)
    hi = float(upper_threshold if upper_threshold is not None else vmax)

    if not (lo < hi):
        raise ValueError(
            f"lower_threshold must be < upper_threshold, got {lo} and {hi}"
        )
    if not (vmin <= lo < hi <= vmax):
        raise ValueError(
            f"thresholds must satisfy vmin <= lower < upper <= vmax "
            f"(vmin={vmin}, vmax={vmax}, lower={lo}, upper={hi})"
        )

    if orientation == "vertical":
        w_px, h_px = int(width), int(height)
    else:
        w_px, h_px = int(height), int(width)

    if w_px < 4 or h_px < 4:
        raise ValueError("width and height must be at least 4 pixels")

    r = max(0, min(int(corner_radius), w_px // 2, h_px // 2))

    if orientation == "vertical":
        # Image row 0 = top of screen → vmax at top; last row = bottom → vmin
        denom_v = float(h_px - 1) if h_px > 1 else 1.0
        row_idx = np.arange(h_px, dtype=np.float64)
        vals_v = vmin + ((h_px - 1 - row_idx) / denom_v) * (vmax - vmin)
        inner_rgb = _row_colors(vals_v, lo, hi, cmap)
        inner = np.broadcast_to(inner_rgb[:, None, :], (h_px, w_px, 3))
    else:
        # Column 0 = left → vmin; last column → vmax
        denom_h = float(w_px - 1) if w_px > 1 else 1.0
        col_idx = np.arange(w_px, dtype=np.float64)
        vals_h = vmin + (col_idx / denom_h) * (vmax - vmin)
        inner_rgb = _row_colors(vals_h, lo, hi, cmap)
        inner = np.broadcast_to(inner_rgb[None, :, :], (h_px, w_px, 3))

    rgba = np.zeros((inner.shape[0], inner.shape[1], 4), dtype=np.float32)
    rgba[:, :, :3] = inner
    rgba[:, :, 3] = 1.0

    mask = _rounded_rect_mask(inner.shape[0], inner.shape[1], r)
    bg = np.array(background[:3], dtype=np.float64)[None, None, :]
    rgba[:, :, :3] = rgba[:, :, :3] * mask[..., None] + bg * (1.0 - mask[..., None])
    rgba[:, :, 3] = mask.astype(np.float32)

    if border_width > 0 and border_alpha > 0:
        a = float(min(border_alpha * border_width, 1.0))
        _blend_edge(rgba, mask, border_rgb, a)

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_png_rgba(out, rgba)
    return out


def _viridis_lut():
    try:
        import matplotlib.pyplot as plt

        try:
            return plt.colormaps["viridis"]
        except (AttributeError, KeyError):
            return plt.cm.get_cmap("viridis")
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "export_colorbar needs matplotlib for the viridis colormap. "
            "Install with: pip install matplotlib"
        ) from e


def _row_colors(
    vals: np.ndarray,
    lo: float,
    hi: float,
    cmap,
) -> np.ndarray:
    """One RGB row per value in ``vals`` (shape (h, 3))."""
    rgb_lo = np.asarray(cmap(0.0)[:3], dtype=np.float64)
    rgb_hi = np.asarray(cmap(1.0)[:3], dtype=np.float64)
    out = np.empty((len(vals), 3), dtype=np.float64)
    low = vals <= lo
    high = vals >= hi
    mid = ~low & ~high
    out[low] = rgb_lo
    out[high] = rgb_hi
    if np.any(mid):
        t = (vals[mid] - lo) / (hi - lo)
        out[mid] = cmap(t)[:, :3]
    return out.astype(np.float32)


def _rounded_rect_mask(h: int, w: int, r: int) -> np.ndarray:
    """Values in ``[0, 1]``, 1 inside rounded rectangle."""
    if r <= 0:
        return np.ones((h, w), dtype=np.float64)
    try:
        from PIL import Image, ImageDraw

        im = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(im)
        draw.rounded_rectangle((0, 0, w, h), radius=r, fill=255)
        return np.asarray(im, dtype=np.float64) / 255.0
    except ImportError:
        return _rounded_rect_mask_numpy(h, w, r)


def _rounded_rect_mask_numpy(h: int, w: int, r: int) -> np.ndarray:
    """Squared corners if Pillow unavailable."""
    return np.ones((h, w), dtype=np.float64)


def _blend_edge(
    rgba: np.ndarray,
    mask: np.ndarray,
    border_rgb: Tuple[float, float, float],
    border_alpha: float,
) -> None:
    """Light stroke along the silhouette of ``mask`` (high |∇mask|)."""
    m = mask.astype(np.float64)
    gy, gx = np.gradient(m)
    edge = np.sqrt(gx * gx + gy * gy)
    mx = edge.max()
    if mx < 1e-12:
        return
    edge /= mx
    edge *= border_alpha
    br, bg, bb = border_rgb
    for i, bv in enumerate((br, bg, bb)):
        c = rgba[:, :, i].astype(np.float64)
        rgba[:, :, i] = c * (1.0 - edge) + bv * edge
    rgba[:, :, :] = np.clip(rgba[:, :, :], 0.0, 1.0)


def _save_png_rgba(path: Path, rgba: np.ndarray) -> None:
    arr = np.clip(rgba, 0.0, 1.0)
    arr_u8 = np.round(arr * 255.0).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(arr_u8, mode="RGBA").save(path, format="PNG")
    except ImportError:
        try:
            import matplotlib.image as mpimg

            mpimg.imsave(path, arr_u8, format="png")
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Install pillow (pip install pillow) or use matplotlib.image for PNG export"
            ) from e
