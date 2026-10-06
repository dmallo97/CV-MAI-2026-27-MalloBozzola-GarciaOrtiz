"""Gaussian convolution filter with a selectable direction."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


DIRECTIONS = ("normal", "horizontal", "vertical")
DIRECTION_LABELS = {
    "normal": "Full convolution",
    "horizontal": "Horizontal convolution",
    "vertical": "Vertical convolution",
}
SIGMA_RANGE = (0.1, 50.0)


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


class ConvolutionDialog(simpledialog.Dialog):
    """Modal dialog with direction radio buttons and a sigma spinbox."""

    def __init__(self, parent: tk.Misc) -> None:
        self.options: tuple[str, float] | None = None
        super().__init__(parent, "Gaussian convolution")

    def body(self, master: tk.Frame) -> tk.Widget:
        self.direction = tk.StringVar(master, value="normal")
        self.sigma = tk.StringVar(master, value="1.0")

        frame = ttk.LabelFrame(master, text="Direction", padding=8)
        frame.pack(fill="x", padx=8, pady=(8, 4))
        buttons = [
            ttk.Radiobutton(frame, text=DIRECTION_LABELS[name], value=name, variable=self.direction)
            for name in DIRECTIONS
        ]
        for button in buttons:
            button.pack(anchor="w")

        row = ttk.Frame(master, padding=(8, 4))
        row.pack(fill="x")
        ttk.Label(row, text="Sigma:").pack(side="left")
        ttk.Spinbox(
            row,
            from_=SIGMA_RANGE[0],
            to=SIGMA_RANGE[1],
            increment=0.5,
            textvariable=self.sigma,
            width=8,
        ).pack(side="left", padx=(6, 0))
        return buttons[0]

    def validate(self) -> bool:
        try:
            sigma = float(self.sigma.get())
        except ValueError:
            sigma = None
        low, high = SIGMA_RANGE
        if sigma is None or not low <= sigma <= high:
            messagebox.showwarning(
                "Gaussian convolution",
                f"Sigma must be a number between {low:g} and {high:g}.",
                parent=self,
            )
            return False
        self.options = (self.direction.get(), sigma)
        return True


def _ask_options(parent: tk.Misc) -> tuple[str, float] | None:
    return ConvolutionDialog(parent).options


class GaussianConvolutionTool(ForensicsTool):
    tool_id = "gaussian_convolution"
    title = "Gaussian convolution"
    category = "Filtering"
    description = "Convolve the image with a Gaussian kernel (full, horizontal, or vertical)."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_options(parent)
        if options is None:
            return None
        direction, sigma = options

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
            message=f"Applied Gaussian {DIRECTION_LABELS[direction].lower()} (sigma={sigma:g}).",
            details={
                "Operation": "Gaussian convolution",
                "Direction": DIRECTION_LABELS[direction],
                "Sigma": f"{sigma:g}",
                "Kernel size": kernel_shape,
                "Output mode": output.mode,
            },
        )
