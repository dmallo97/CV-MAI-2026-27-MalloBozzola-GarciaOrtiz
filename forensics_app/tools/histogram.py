"""Display per-channel intensity histograms for the working image."""

from __future__ import annotations

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class HistogramVisualizationTool(ForensicsTool):
	tool_id = "histogram_visualization"
	title = "Histogram visualization"
	category = "Analysis"
	description = "Visualize the red, green, and blue intensity distributions."

	def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
		assert document.current is not None
		image = np.asarray(document.current.convert("RGB"))

		figure, axis = plt.subplots(figsize=(8, 4.5))
		for channel, color, name in ((0, "red", "Red"), (1, "green", "Green"), (2, "blue", "Blue")):
			counts, _ = np.histogram(image[:, :, channel], bins=256, range=(0, 256))
			axis.plot(range(256), counts, color=color, label=name)

		axis.set_title("RGB intensity histogram")
		axis.set_xlabel("Pixel intensity")
		axis.set_ylabel("Pixel count")
		axis.set_xlim(0, 255)
		axis.grid(True, alpha=0.25)
		axis.legend()
		figure.tight_layout()
		plt.show(block=False)

		return ToolResult(
			message="Displayed the RGB histogram in a separate window.",
			details={
				"Operation": "RGB histogram",
				"Image size": f"{document.current.width} × {document.current.height}",
				"Pixels": f"{document.current.width * document.current.height:,}",
			},
		)
