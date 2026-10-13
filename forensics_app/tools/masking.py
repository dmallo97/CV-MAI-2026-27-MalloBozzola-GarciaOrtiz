from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def build_mask(image: Image.Image, threshold: int) -> np.ndarray:
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be between 0 and 255.")
    return np.asarray(image.convert("L")) > threshold


def apply_mask(image: Image.Image, mask: np.ndarray, invert: bool = False) -> Image.Image:
    mode = "L" if image.mode == "L" else "RGB"
    pixels = np.asarray(image if image.mode == mode else image.convert(mode))
    if mask.shape != pixels.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {pixels.shape[:2]}.")

    keep = ~mask if invert else mask
    if mode == "RGB":
        keep = keep[:, :, np.newaxis]
    return Image.fromarray((pixels * keep).astype(np.uint8), mode=mode)


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Threshold mask"
    category = "Segmentation"
    description = "Keep pixels brighter or darker than an intensity threshold."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        threshold = simpledialog.askinteger(
            "Masking",
            "Intensity threshold (0–255):",
            parent=parent,
            initialvalue=135,
            minvalue=0,
            maxvalue=255,
        )
        if threshold is None:
            return None

        keep_bright = messagebox.askyesnocancel(
            "Masking",
            "Keep pixels brighter than the threshold?\n"
            "Yes = bright (masked image 1), No = dark (masked image 2)",
            parent=parent,
        )
        if keep_bright is None:
            return None

        assert document.current is not None
        source = document.current
        mask = build_mask(source, threshold)
        masked_bright = apply_mask(source, mask)
        masked_dark = apply_mask(source, mask, invert=True)
        output = masked_bright if keep_bright else masked_dark

        figure, axes = plt.subplots(1, 4, figsize=(12, 4))
        panels = (
            (source, "Original image"),
            (mask, "Binary mask"),
            (masked_bright, "Masked image 1"),
            (masked_dark, "Masked image 2"),
        )
        for axis, (data, title) in zip(axes, panels):
            axis.imshow(data, cmap="gray")
            axis.set_title(title)
            axis.axis("off")
        figure.tight_layout()
        plt.show(block=False)

        kept = int(mask.sum()) if keep_bright else int((~mask).sum())
        total = mask.size
        return ToolResult(
            image=output,
            message=f"Applied threshold mask at {threshold}, keeping {'bright' if keep_bright else 'dark'} pixels.",
            details={
                "Threshold": threshold,
                "Kept": "bright (> threshold)" if keep_bright else "dark (≤ threshold)",
                "Pixels kept": f"{kept} of {total} ({100 * kept / total:.1f}%)",
                "Output mode": output.mode,
            },
        )
