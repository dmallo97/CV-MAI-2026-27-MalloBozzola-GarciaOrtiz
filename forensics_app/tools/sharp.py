"""Manual image sharpening (unsharp masking) built on the Gaussian convolution tool."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .convolution import convolve


def gaussian_size(sigma: float) -> int:
    """Odd kernel size covering +/- 3 sigma of the Gaussian."""
    return 2 * math.ceil(3 * sigma) + 1


def sharpen(image: Image.Image, sigma: float = 1.0, amount: float = 1.0) -> Image.Image:
    """Return ``original + amount * (original - gaussian_blur(original))``.

    The difference between the image and its Gaussian blur is a mask of the
    borders (the high frequencies); adding it back to the original enhances them.
    """
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    blurred = convolve(image, "gaussian", gaussian_size(sigma), sigma)

    original = np.asarray(image, dtype=np.float64)
    mask = original - np.asarray(blurred, dtype=np.float64)
    sharpened = original + amount * mask

    result = np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode=image.mode)


class SharpeningTool(ForensicsTool):
    tool_id = "sharpening"
    title = "Image sharpening"
    category = "Enhancement"
    description = "Sharpen the image by adding back its difference from a Gaussian-blurred copy."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        sigma = simpledialog.askfloat(
            "Image sharpening",
            "Sigma of the Gaussian filter used to find the borders:",
            parent=parent,
            initialvalue=1.0,
            minvalue=0.1,
            maxvalue=50.0,
        )
        if sigma is None:
            return None

        assert document.current is not None
        output = sharpen(document.current, sigma)
        size = gaussian_size(sigma)
        return ToolResult(
            image=output,
            message=f"Sharpened the image (sigma={sigma:g}).",
            details={
                "Operation": "Sharpening (original + border mask)",
                "Sigma": f"{sigma:g}",
                "Kernel size": f"{size}x{size}",
                "Output mode": output.mode,
            },
        )
