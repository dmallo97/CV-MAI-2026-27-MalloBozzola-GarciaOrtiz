"""Stretch image channel ranges between two percentiles and display the resulting histogram."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

DEFAULT_LOW_PERCENTILE = 10.0
DEFAULT_HIGH_PERCENTILE = 90.0
CHANNEL_NAMES = ("Red", "Green", "Blue")


def stretch_contrast(
    image: Image.Image,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> tuple[Image.Image, dict[str, str]]:
    """Linearly map each RGB channel's [low, high] percentile range onto 0–255.

    Returns the stretched image and the input range used for each channel.
    """
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100.")

    source = np.asarray(image.convert("RGB"))
    stretched = source.copy()
    channel_ranges: dict[str, str] = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        values = source[:, :, channel]
        minimum, maximum = np.percentile(values, (low_percentile, high_percentile))
        channel_ranges[f"{name} input range"] = f"{minimum:.0f}–{maximum:.0f}"
        if maximum > minimum:
            scaled = (values.astype(np.float32) - minimum) * (255.0 / (maximum - minimum))
            stretched[:, :, channel] = np.rint(np.clip(scaled, 0, 255)).astype(np.uint8)
    return Image.fromarray(stretched, mode="RGB"), channel_ranges


class ContrastStretchingDialog(simpledialog.Dialog):
    """Modal dialog asking for the low and high percentiles."""

    def __init__(self, parent: tk.Misc) -> None:
        self.options: tuple[float, float] | None = None
        super().__init__(parent, "Contrast stretching")

    def body(self, master: tk.Frame) -> tk.Widget:
        self.low = tk.StringVar(master, value=f"{DEFAULT_LOW_PERCENTILE:g}")
        self.high = tk.StringVar(master, value=f"{DEFAULT_HIGH_PERCENTILE:g}")

        frame = ttk.LabelFrame(master, text="Percentiles", padding=8)
        frame.pack(fill="x", padx=8, pady=8)
        spinboxes = []
        for row, (label, variable) in enumerate((("Low:", self.low), ("High:", self.high))):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            spinbox = ttk.Spinbox(frame, from_=0, to=100, increment=1, textvariable=variable, width=8)
            spinbox.grid(row=row, column=1, padx=(6, 0), pady=2)
            spinboxes.append(spinbox)
        return spinboxes[0]

    def validate(self) -> bool:
        try:
            low, high = float(self.low.get()), float(self.high.get())
            valid = 0 <= low < high <= 100
        except ValueError:
            valid = False
        if not valid:
            messagebox.showwarning(
                "Contrast stretching",
                "Percentiles must be numbers with 0 <= low < high <= 100.",
                parent=self,
            )
            return False
        self.options = (low, high)
        return True


def _ask_options(parent: tk.Misc) -> tuple[float, float] | None:
    return ContrastStretchingDialog(parent).options


class ContrastStretchingTool(ForensicsTool):
    tool_id = "contrast_stretching"
    title = "Contrast stretching"
    category = "Enhancement"
    description = "Stretch RGB channel ranges between two percentiles and show the resulting histogram."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_options(parent)
        if options is None:
            return None
        low_percentile, high_percentile = options

        assert document.current is not None
        output, channel_ranges = stretch_contrast(document.current, low_percentile, high_percentile)

        stretched = np.asarray(output)
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for channel, name in enumerate(CHANNEL_NAMES):
            counts, _ = np.histogram(stretched[:, :, channel], bins=256, range=(0, 256))
            axis.plot(range(256), counts, color=name.lower(), label=name)

        axis.set_title("Histogram after contrast stretching")
        axis.set_xlabel("Pixel intensity")
        axis.set_ylabel("Pixel count")
        axis.set_xlim(0, 255)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        plt.show(block=False)

        return ToolResult(
            image=output,
            message=(
                f"Applied contrast stretching (percentiles {low_percentile:g}–{high_percentile:g}); "
                "displayed the resulting RGB histogram."
            ),
            details={
                "Operation": "Contrast stretching",
                "Low percentile": f"{low_percentile:g}",
                "High percentile": f"{high_percentile:g}",
                **channel_ranges,
            },
        )
