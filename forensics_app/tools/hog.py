"""Histogram of Oriented Gradients (HOG) extraction and visualization."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image
from skimage import exposure, feature, img_as_float, img_as_ubyte

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

DEFAULT_ORIENTATIONS = 9
DEFAULT_PIXELS_PER_CELL = 6
DEFAULT_CELLS_PER_BLOCK = 3


def compute_hog(
    image: Image.Image,
    orientations: int = DEFAULT_ORIENTATIONS,
    pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
    cells_per_block: int = DEFAULT_CELLS_PER_BLOCK,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the HOG feature vector and its visualization (same size as the image)."""
    if image.mode == "L":
        pixels, channel_axis = img_as_float(np.asarray(image)), None
    else:
        pixels, channel_axis = img_as_float(np.asarray(image.convert("RGB"))), -1
    return feature.hog(
        pixels,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        visualize=True,
        channel_axis=channel_axis,
    )


def hog_to_image(hog_image: np.ndarray) -> Image.Image:
    rescaled = exposure.rescale_intensity(hog_image, out_range=(0.0, 1.0))
    return Image.fromarray(img_as_ubyte(rescaled), mode="L")


class HogDialog(simpledialog.Dialog):
    """Modal dialog asking for the three HOG parameters."""

    FIELDS = (
        ("Orientations:", DEFAULT_ORIENTATIONS),
        ("Pixels per cell:", DEFAULT_PIXELS_PER_CELL),
        ("Cells per block:", DEFAULT_CELLS_PER_BLOCK),
    )

    def __init__(self, parent: tk.Misc) -> None:
        self.options: tuple[int, int, int] | None = None
        super().__init__(parent, "HOG extraction")

    def body(self, master: tk.Frame) -> tk.Widget:
        frame = ttk.LabelFrame(master, text="Parameters", padding=8)
        frame.pack(fill="x", padx=8, pady=8)
        self.values = []
        spinboxes = []
        for row, (label, default) in enumerate(self.FIELDS):
            variable = tk.StringVar(master, value=str(default))
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            spinbox = ttk.Spinbox(frame, from_=1, to=64, increment=1, textvariable=variable, width=8)
            spinbox.grid(row=row, column=1, padx=(6, 0), pady=2)
            self.values.append(variable)
            spinboxes.append(spinbox)
        return spinboxes[0]

    def validate(self) -> bool:
        try:
            options = tuple(int(value.get()) for value in self.values)
            valid = all(option >= 1 for option in options)
        except ValueError:
            valid = False
        if not valid:
            messagebox.showwarning("HOG extraction", "All parameters must be integers >= 1.", parent=self)
            return False
        self.options = options
        return True


def _ask_options(parent: tk.Misc) -> tuple[int, int, int] | None:
    return HogDialog(parent).options


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG extraction"
    category = "Feature descriptors"
    description = "Compute the Histogram of Oriented Gradients and show its visualization."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_options(parent)
        if options is None:
            return None
        orientations, pixels_per_cell, cells_per_block = options

        assert document.current is not None
        features, hog_image = compute_hog(document.current, orientations, pixels_per_cell, cells_per_block)
        cells_y, cells_x = (side // pixels_per_cell for side in hog_image.shape)
        return ToolResult(
            image=hog_to_image(hog_image),
            message=f"Extracted {features.size:,} HOG features with {orientations} orientations.",
            details={
                "Operation": "HOG extraction",
                "Orientations": str(orientations),
                "Pixels per cell": f"{pixels_per_cell} x {pixels_per_cell}",
                "Cells per block": f"{cells_per_block} x {cells_per_block}",
                "Cells (w x h)": f"{cells_x} x {cells_y}",
                "Blocks (w x h)": f"{cells_x - cells_per_block + 1} x {cells_y - cells_per_block + 1}",
                "Feature vector length": f"{features.size:,}",
            },
        )
