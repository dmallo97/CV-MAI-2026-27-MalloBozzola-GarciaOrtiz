"""Canny edge detection and visualization of the detected edges."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import color, feature, img_as_float, img_as_ubyte

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


ANALYSIS_KEY = "canny"
NO_EDGES_MESSAGE = "No edges detected yet. Run Canny edge detection first."


@dataclass(frozen=True)
class CannyResult:
    """Edges found by the Canny tool and the image they were detected on."""

    source: Image.Image
    edges: np.ndarray
    sigma: float


def canny_edges(image: Image.Image, sigma: float = 1.0) -> np.ndarray:
    """Return the boolean Canny edge map of the image's grayscale version."""
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    gray = img_as_float(np.asarray(image.convert("L")))
    return feature.canny(gray, sigma=sigma)


def edge_map(edges: np.ndarray) -> Image.Image:
    """Draw the edges in white on a black (all-zeros) image."""
    output = np.zeros(edges.shape, dtype=np.uint8)
    output[edges] = 255
    return Image.fromarray(output, mode="L")


def overlay_edges(image: Image.Image, edges: np.ndarray) -> Image.Image:
    """Superimpose the edges in red on the grayscale version of the image."""
    gray = img_as_float(np.asarray(image.convert("L")))
    rgb = color.gray2rgb(gray)
    red_edges = rgb.copy()
    red_edges[edges] = [1, 0, 0]
    return Image.fromarray(img_as_ubyte(red_edges), mode="RGB")


def _edge_details(edges: np.ndarray) -> dict[str, str]:
    edge_pixels = int(edges.sum())
    return {
        "Edge pixels": f"{edge_pixels:,}",
        "Edge density": f"{edge_pixels / edges.size:.2%}",
    }


class CannyEdgeTool(ForensicsTool):
    tool_id = "canny_edges"
    title = "Canny edge detection"
    category = "Edge detection"
    description = "Detect edges with the Canny algorithm and show them in white on a black background."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        sigma = simpledialog.askfloat(
            "Canny edge detection",
            "Sigma (standard deviation of the Gaussian smoothing):",
            parent=parent,
            initialvalue=1.0,
            minvalue=0.1,
            maxvalue=50.0,
        )
        if sigma is None:
            return None

        assert document.current is not None
        edges = canny_edges(document.current, sigma)
        document.analyses[ANALYSIS_KEY] = CannyResult(document.current.copy(), edges, sigma)
        return ToolResult(
            image=edge_map(edges),
            message=f"Detected Canny edges (sigma={sigma:g}).",
            details={
                "Operation": "Canny edge detection",
                "Sigma": f"{sigma:g}",
                **_edge_details(edges),
            },
        )


class EdgeVisualizationTool(ForensicsTool):
    tool_id = "edge_visualization"
    title = "Edge visualization"
    category = "Edge detection"
    description = "Draw the last detected Canny edges in red over the grayscale image they came from."
    unavailable_message = NO_EDGES_MESSAGE

    def is_available(self, document: ImageDocument) -> bool:
        return ANALYSIS_KEY in document.analyses

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        result: CannyResult | None = document.analyses.get(ANALYSIS_KEY)
        if result is None:
            return ToolResult(message=NO_EDGES_MESSAGE)

        output = overlay_edges(result.source, result.edges)
        return ToolResult(
            image=output,
            message=f"Showing Canny edges (sigma={result.sigma:g}) in red over the grayscale image.",
            details={
                "Operation": "Edge visualization",
                "Sigma": f"{result.sigma:g}",
                **_edge_details(result.edges),
            },
        )
