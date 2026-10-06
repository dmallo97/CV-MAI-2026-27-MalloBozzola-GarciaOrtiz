"""Canny edge detection shown as an edge map or as red edges over the grayscale image."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import color, feature, img_as_float, img_as_ubyte

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


VISUALIZATIONS = ("edges", "overlay")


def _parse_visualization(raw: str) -> str:
    cleaned = raw.strip().lower()
    if not cleaned:
        return "edges"
    for visualization in VISUALIZATIONS:
        if visualization.startswith(cleaned):
            return visualization
    raise ValueError(f"Unknown visualization '{raw}'. Use edges or overlay.")


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


class CannyEdgeTool(ForensicsTool):
    tool_id = "canny_edges"
    title = "Canny edge detection"
    category = "Analysis"
    description = "Detect edges with the Canny algorithm as an edge map or as red edges over the grayscale image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        raw_visualization = simpledialog.askstring(
            "Canny edge detection",
            "Visualization (edges = edge map, overlay = red edges on grayscale):",
            parent=parent,
            initialvalue="edges",
        )
        if raw_visualization is None:
            return None
        try:
            visualization = _parse_visualization(raw_visualization)
        except ValueError as error:
            return ToolResult(
                message=f"Canny edge detection cancelled: {error}",
                details={"Input": raw_visualization, "Error": str(error)},
            )

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
        if visualization == "overlay":
            output = overlay_edges(document.current, edges)
            shown_as = "edges shown in red over the grayscale image"
        else:
            output = edge_map(edges)
            shown_as = "edges shown in white on black"
        edge_pixels = int(edges.sum())
        return ToolResult(
            image=output,
            message=f"Detected Canny edges (sigma={sigma:g}); {shown_as}.",
            details={
                "Operation": "Canny edge detection",
                "Visualization": visualization,
                "Sigma": f"{sigma:g}",
                "Edge pixels": f"{edge_pixels:,}",
                "Edge density": f"{edge_pixels / edges.size:.2%}",
                "Output mode": output.mode,
            },
        )
