"""Convolution-based filters (Gaussian, median, Sobel, Prewitt, Scharr) from skimage.filters."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


FILTERS = ("gaussian", "median", "sobel", "prewitt", "scharr")
EDGE_FILTERS = {"sobel": filters.sobel, "prewitt": filters.prewitt, "scharr": filters.scharr}
# skimage implements the edge operators with fixed 3x3 kernels.
EDGE_KERNEL_SIZE = 3


def _parse_filter(raw: str) -> str:
    """Match ``raw`` against ``FILTERS`` by exact name or unambiguous prefix."""
    cleaned = raw.strip().lower()
    if not cleaned:
        return "gaussian"
    if cleaned in FILTERS:
        return cleaned
    matches = [name for name in FILTERS if name.startswith(cleaned)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unknown filter '{raw}'. Use one of: {', '.join(FILTERS)}.")


def _filter_channel(channel: np.ndarray, filter_name: str, size: int, sigma: float) -> np.ndarray:
    """Apply one filter to a 2D float channel with values in [0, 1]."""
    if filter_name == "gaussian":
        radius = (size - 1) // 2
        # scipy's kernel radius is int(truncate * sigma + 0.5), so this yields exactly `size` taps.
        return filters.gaussian(channel, sigma=sigma, truncate=radius / sigma, mode="reflect", preserve_range=True)
    if filter_name == "median":
        footprint = np.ones((size, size), dtype=bool)
        return filters.median(channel, footprint=footprint, mode="reflect")
    # Edge operators return the gradient magnitude.
    return EDGE_FILTERS[filter_name](channel, mode="reflect")


def convolve(
    image: Image.Image,
    filter_name: str = "gaussian",
    size: int = 3,
    sigma: float = 1.0,
) -> Image.Image:
    """Return a new image filtered with one of the skimage.filters operators.

    ``size`` is the (odd) kernel size for the Gaussian and median filters; the
    edge operators always use skimage's 3x3 kernels and return the gradient
    magnitude, rescaled so the strongest edge maps to 255.
    """
    if filter_name not in FILTERS:
        raise ValueError(f"Unknown filter '{filter_name}'. Use one of: {', '.join(FILTERS)}.")
    if size < 1 or size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")

    if image.mode not in ("L", "RGB", "RGBA"):
        image = image.convert("RGB")
    data = np.asarray(image, dtype=np.float64) / 255.0
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    # Filter only the colour channels: an opaque alpha has zero gradient, so
    # edge operators would otherwise make the whole result transparent.
    color_channels = 3 if image.mode == "RGBA" else data.shape[2]

    output = np.stack(
        [_filter_channel(data[:, :, c], filter_name, size, sigma) for c in range(color_channels)],
        axis=2,
    )
    if filter_name in EDGE_FILTERS:
        peak = output.max()
        # Ignore float round-off on flat images so it is not stretched to full white.
        output = output / peak if peak > 1e-6 else np.zeros_like(output)
    if color_channels < data.shape[2]:
        output = np.concatenate([output, data[:, :, color_channels:]], axis=2)

    result = np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8)
    if image.mode == "L":
        result = result[:, :, 0]
    return Image.fromarray(result)


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution"
    category = "Filtering"
    description = "Filter the image with a Gaussian, median, Sobel, Prewitt, or Scharr kernel."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        raw_filter = simpledialog.askstring(
            "Convolution",
            f"Filter ({', '.join(FILTERS)}):",
            parent=parent,
            initialvalue="gaussian",
        )
        if raw_filter is None:
            return None
        try:
            filter_name = _parse_filter(raw_filter)
        except ValueError as error:
            return ToolResult(
                message=f"Convolution cancelled: {error}",
                details={"Filter": raw_filter, "Error": str(error)},
            )

        if filter_name in EDGE_FILTERS:
            size = EDGE_KERNEL_SIZE
        else:
            size = simpledialog.askinteger(
                "Convolution",
                "Kernel size (odd integer):",
                parent=parent,
                initialvalue=3,
                minvalue=1,
                maxvalue=99,
            )
            if size is None:
                return None
            if size % 2 == 0:
                return ToolResult(
                    message="Convolution cancelled: kernel size must be odd.",
                    details={"Kernel size": str(size), "Error": "Kernel size must be odd."},
                )

        sigma = 1.0
        if filter_name == "gaussian":
            sigma = simpledialog.askfloat(
                "Convolution",
                "Sigma (standard deviation of the Gaussian):",
                parent=parent,
                initialvalue=1.0,
                minvalue=0.1,
                maxvalue=50.0,
            )
            if sigma is None:
                return None

        assert document.current is not None
        output = convolve(document.current, filter_name, size, sigma)
        details = {
            "Operation": "Convolution",
            "Filter": filter_name,
            "Kernel size": f"{size}x{size}",
            "Output mode": output.mode,
        }
        if filter_name == "gaussian":
            details["Sigma"] = f"{sigma:g}"
        message = f"Applied {filter_name} filter"
        message += f" (sigma={sigma:g})." if filter_name == "gaussian" else "."
        return ToolResult(image=output, message=message, details=details)
