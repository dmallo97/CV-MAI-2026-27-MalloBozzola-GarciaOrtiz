"""Gaussian convolution filter with a selectable direction."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


DIRECTIONS = ("normal", "horizontal", "vertical")


def _parse_direction(raw: str) -> str:
    cleaned = raw.strip().lower()
    if not cleaned:
        return "normal"
    for direction in DIRECTIONS:
        if direction.startswith(cleaned):
            return direction
    raise ValueError(f"Unknown direction '{raw}'. Use normal, horizontal, or vertical.")


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Normalised 1D Gaussian kernel with radius ceil(3 * sigma)."""
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    radius = max(1, math.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_kernel_2d(sigma: float) -> np.ndarray:
    """Full 2D Gaussian kernel (outer product of the 1D kernel)."""
    kernel = gaussian_kernel_1d(sigma)
    return np.outer(kernel, kernel)


def _convolve_axis(data: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve a (H, W, C) float array along one spatial axis, reflecting at the borders."""
    radius = len(kernel) // 2
    pad = [(0, 0)] * data.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(data, pad, mode="reflect")
    length = data.shape[axis]
    output = np.zeros_like(data)
    # The Gaussian is symmetric, so correlation and convolution give the same result.
    for offset, weight in enumerate(kernel):
        output += weight * np.take(padded, np.arange(offset, offset + length), axis=axis)
    return output


def gaussian_convolve(image: Image.Image, sigma: float = 1.0, direction: str = "normal") -> Image.Image:
    """Return a new image convolved with a Gaussian kernel.

    ``horizontal`` convolves each row with a 1D kernel, ``vertical`` each column,
    and ``normal`` applies the full 2D kernel (computed separably: rows, then columns).
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction '{direction}'. Use normal, horizontal, or vertical.")
    kernel = gaussian_kernel_1d(sigma)

    if image.mode not in ("L", "RGB", "RGBA"):
        image = image.convert("RGB")
    data = np.asarray(image, dtype=np.float64)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    if direction in ("normal", "horizontal"):
        data = _convolve_axis(data, kernel, axis=1)
    if direction in ("normal", "vertical"):
        data = _convolve_axis(data, kernel, axis=0)

    result = np.clip(np.rint(data), 0, 255).astype(np.uint8)
    if image.mode == "L":
        result = result[:, :, 0]
    return Image.fromarray(result)


class GaussianConvolutionTool(ForensicsTool):
    tool_id = "gaussian_convolution"
    title = "Gaussian convolution"
    category = "Filtering"
    description = "Convolve the image with a Gaussian kernel (normal, horizontal, or vertical)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        raw_direction = simpledialog.askstring(
            "Gaussian convolution",
            "Direction (normal, horizontal, vertical):",
            parent=parent,
            initialvalue="normal",
        )
        if raw_direction is None:
            return None
        try:
            direction = _parse_direction(raw_direction)
        except ValueError as error:
            return ToolResult(
                message=f"Gaussian convolution cancelled: {error}",
                details={"Input": raw_direction, "Error": str(error)},
            )

        sigma = simpledialog.askfloat(
            "Gaussian convolution",
            "Sigma (standard deviation of the kernel):",
            parent=parent,
            initialvalue=1.0,
            minvalue=0.1,
            maxvalue=50.0,
        )
        if sigma is None:
            return None

        assert document.current is not None
        output = gaussian_convolve(document.current, sigma, direction)
        size = len(gaussian_kernel_1d(sigma))
        kernel_shape = {
            "normal": f"{size}x{size}",
            "horizontal": f"1x{size}",
            "vertical": f"{size}x1",
        }[direction]
        return ToolResult(
            image=output,
            message=f"Applied {direction} Gaussian convolution (sigma={sigma:g}).",
            details={
                "Operation": "Gaussian convolution",
                "Direction": direction,
                "Sigma": f"{sigma:g}",
                "Kernel size": kernel_shape,
                "Output mode": output.mode,
            },
        )
