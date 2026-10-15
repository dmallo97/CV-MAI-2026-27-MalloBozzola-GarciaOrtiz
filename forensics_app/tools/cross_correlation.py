"""Normalized cross-correlation (template matching) and its thresholded map."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image
from skimage import feature, img_as_float

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


ANALYSIS_KEY = "ncc"
NO_NCC_MESSAGE = "No correlation detected. Run Canny edge detection first."  # TODO


@dataclass(frozen=True)
class NCCResult:
    """NCC map computed by the NCC tool and what it was computed from."""
    # TODO: which fields does the threshold tool need
    source: Image.Image
    map: np.ndarray
    threshold: int




def ncc_map(image: Image.Image, template: Image.Image) -> np.ndarray:
    """Return the NCC map (values in [-1, 1]) of the
    # TODO: validate (template larger than image? -> ValueError)
    # TODO: both to grayscale float
    # TODO: feature.match_template(...)  -> decide pad_input
    ...


def ncc_to_image(ncc: np.ndarray) -> Image.Image:
    """Turn an NCC map in [-1, 1] into a displayable
    # TODO: how do you map [-1, 1] -> [0, 255]?
    ...


def threshold_ncc(ncc: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean mask of positions whose NCC i
    # TODO: validate threshold range -> ValueError
    ...


def _ncc_details(ncc: np.ndarray) -> dict[str, str]:
    # TODO: useful measurements (max value, its posit
    ...


# ---------- tools ----------

class NormalizedCrossCorrelationTool(ForensicsTool):
    tool_id = "..."      # TODO
    title = "..."
    category = "..."
    description = "..."

    def run(self, parent: tk.Misc, document: ImageDoc
        # TODO: ask the user for the template file; cancel -> return None
        # TODO: open it (what if it is not a valid im
        # TODO: compute map, store NCCResult in document.analyses
        # TODO: return ToolResult with the map image
        ...


class NCCThresholdTool(ForensicsTool):
    tool_id = "..."      # TODO
    title = "..."
    category = "..."
    description = "..."
    unavailable_message = NO_NCC_MESSAGE

    def is_available(self, document: ImageDocument) -> bool:
        ...  # TODO

    def run(self, parent: tk.Misc, document: ImageDoc
        # TODO: get NCCResult from analyses
        # TODO: ask threshold; cancel -> return None
        # TODO: mask -> image, return ToolResult
        ...