"""Stretch image channel ranges and display the resulting histogram."""

from __future__ import annotations

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ContrastStretchingTool(ForensicsTool):
	tool_id = "contrast_stretching"
	title = "Contrast stretching"
	category = "Enhancement"
	description = "Stretch RGB channel ranges and show the resulting histogram."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
		assert document.current is not None
		source = np.asarray(document.current.convert("RGB"))
		stretched = source.copy()
		channel_ranges: dict[str, str] = {}

		for channel, name in enumerate(("Red", "Green", "Blue")):
			values = source[:, :, channel]
			minimum = int(values.min())
			maximum = int(values.max())
			channel_ranges[f"{name} input range"] = f"{minimum}–{maximum}"
			if maximum > minimum:
				scaled = (values.astype(np.float32) - minimum) * (255.0 / (maximum - minimum))
				stretched[:, :, channel] = np.rint(scaled).astype(np.uint8)

		output = Image.fromarray(stretched, mode="RGB")
		figure, axis = plt.subplots(figsize=(8, 4.5))
		for channel, color, name in ((0, "red", "Red"), (1, "green", "Green"), (2, "blue", "Blue")):
			counts, _ = np.histogram(stretched[:, :, channel], bins=256, range=(0, 256))
			axis.plot(range(256), counts, color=color, label=name)

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
			message="Applied contrast stretching; displayed the resulting RGB histogram.",
			details={"Operation": "Contrast stretching", **channel_ranges},
		)
