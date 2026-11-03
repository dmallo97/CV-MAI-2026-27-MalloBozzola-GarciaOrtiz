"""Histogram of visual words (bag of words) using the saved codebook."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .codebook import load_codebook
from .feature_matching import IMAGE_FILE_TYPES, describe_params, extract_features, load_gray


# ---------- Pure functions ----------

def assign_words(descriptors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the closest centre (Euclidean distance) for every descriptor: (N, D) -> (N,)."""
    distances = np.linalg.norm(descriptors[:, None, :] - centers[None, :, :], axis=2)  # (N, K)
    return distances.argmin(axis=1)


def build_histogram(words: np.ndarray, k: int) -> np.ndarray:
    """k-dimensional histogram: fraction of descriptors assigned to each visual word."""
    counts = np.bincount(words, minlength=k)
    return counts / counts.sum()


def plot_histogram(axis, histogram: np.ndarray) -> None:
    axis.bar(np.arange(len(histogram)), histogram, color="lightgray", edgecolor="black")
    axis.set_title("Histogram of visual words")
    axis.set_xlabel("Visual word index")
    axis.set_ylabel("Normalized frequency")


def plot_image_histogram(gray, keypoints, words, histogram, name="New image") -> plt.Figure:
    """Image with its keypoints (left), histogram (right) and an arrow from each keypoint to its word."""
    figure, (image_axis, histogram_axis) = plt.subplots(1, 2, figsize=(12, 5))
    image_axis.imshow(gray, cmap="gray")
    image_axis.scatter(keypoints[:, 1], keypoints[:, 0], s=12, c=words, cmap="tab20")  # colour = word
    image_axis.set_title(name)
    image_axis.axis("off")
    plot_histogram(histogram_axis, histogram)

    # Arrow from the keypoint (col, row) in the image to the top of the bar of its word
    for (row, col), word in zip(keypoints, words):
        arrow = ConnectionPatch(
            xyA=(col, row), coordsA=image_axis.transData,
            xyB=(word, histogram[word]), coordsB=histogram_axis.transData,
            arrowstyle="->", color="gray", linewidth=0.5, alpha=0.5,
        )
        figure.add_artist(arrow)
    figure.tight_layout()
    return figure


# ---------- Tools ----------

class NewImageHistogramTool(ForensicsTool):
    tool_id = "new_image_histogram"
    title = "Compute histogram of visual words for new image"
    category = "Bag of Words"
    description = (
        "Ask for a new image, assign each of its descriptors to the closest visual word "
        "of the saved codebook and plot the normalized histogram with arrows from the keypoints."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        centers, _, params = load_codebook()
        filename = filedialog.askopenfilename(title="Open new image", filetypes=IMAGE_FILE_TYPES, parent=parent)
        if not filename:  # cancelled
            return None
        name = Path(filename).name
        gray = load_gray(filename)
        keypoints, descriptors = extract_features(gray, params)  # same descriptor as the codebook
        if len(descriptors) == 0:
            raise ValueError("No keypoints found in the image.")

        words = assign_words(descriptors.astype(float), centers)
        histogram = build_histogram(words, len(centers))
        plot_image_histogram(gray, keypoints, words, histogram, name)
        plt.show(block=False)

        return ToolResult(
            message=f"Assigned {len(words)} descriptors of {name} to {len(centers)} visual words.",
            details={
                "Operation": "Histogram of visual words (new image)",
                "Image": name,
                **describe_params(params),
                "Descriptors": str(len(words)),
                "Visual words (K)": str(len(centers)),
                "Words used": str(np.count_nonzero(histogram)),
                "Most frequent word": f"vw {histogram.argmax()} ({histogram.max():.3f})",
            },
        )


class CodebookHistogramTool(ForensicsTool):
    tool_id = "codebook_histogram"
    title = "Show codebook histogram"
    category = "Bag of Words"
    description = "Plot the normalized histogram of visual words of all the descriptors used to build the codebook."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        centers, words, params = load_codebook()
        histogram = build_histogram(words, len(centers))
        figure, axis = plt.subplots(figsize=(8, 4))
        plot_histogram(axis, histogram)
        axis.set_title("Histogram of visual words (codebook)")
        figure.tight_layout()
        plt.show(block=False)

        return ToolResult(
            message=f"Codebook histogram of {len(words)} descriptors over {len(centers)} visual words.",
            details={
                "Operation": "Histogram of visual words (codebook)",
                **describe_params(params),
                "Descriptors": str(len(words)),
                "Visual words (K)": str(len(centers)),
            },
        )
