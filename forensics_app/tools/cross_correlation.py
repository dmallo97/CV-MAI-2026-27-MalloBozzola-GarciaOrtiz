from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageTk
from skimage import color, feature, img_as_float, img_as_ubyte, measure, transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .template_dialogs import(
    MAX_PREVIEW_SIZE,
    SOURCE_FILE,
    Box,
    ask_angle,
    ask_region, 
    ask_template_source,
    ask_threshold,
)

ANALYSIS_KEY = "cross_correlation"
NO_NCC_MESSAGE = "No correlation map yet. Run normalized cross-correlation first."
# Brightness kept for non-matching pixels in the overlay
DIM_FACTOR = 0.4
# Tools must not import from the ui package, so the file filter is defined here.
TEMPLATE_FILE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
# Small templates are enlarged (nearest neighbour) so they stay visible in the preview window.
TEMPLATE_PREVIEW_MIN_SIDE = 160
DEFAULT_THRESHOLD = 0.9
THRESHOLD_MARGIN = 0.05
# The single preview window shared by the template matching tools (None while closed).
_template_window: tk.Toplevel | None = None


@dataclass(frozen=True)
class CrossCorrelationResult:
    """Correlation map and everything needed to visualize it later"""

    source: Image.Image
    correlation: np.ndarray  # float, shape (H, W), values in [-1, 1]
    template_source: str  # file name of the template
    template_shape: tuple[int, int]  # (height, width) before rotation
    angle: float
    template: np.ndarray  # rotated template actually used, float in [0, 1]


# ---------- Pure functions ----------
def to_gray_float(image: Image.Image) -> np.ndarray:
    """Convert any PIL image to a 2-D float array in [0, 1]."""
    return img_as_float(np.asarray(image.convert("L")))


def load_template(path: str | Path) -> np.ndarray:
    """Read an image file as a 2-D float template in [0, 1]."""
    with Image.open(path) as image:
        return to_gray_float(image)
    
    
def crop_template(image:np.ndarray, box: Box) -> np.ndarray:
    """Cut the (left, tip, right, bottom) region out of a 2-D image."""
    left, top, right, bottom = box
    height, width = image.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Invalid region {box} for an image of {width}x{height} pixels.")
    return image[top:bottom, left:right].copy()


def rotate_template(template: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the template by `angle` degrees."""
    if angle == 0:
        return template
    return transform.rotate(template, angle)


def normalized_cross_correlation(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Return the NCC map with the same shape as `image`, values in [-1, 1].
    With ``pad_input=True`` the peak is located at the *centre* of the matched region"""
    if image.ndim != 2 or template.ndim != 2:
        raise ValueError("Image and template must be 2-D (grayscale) arrays.")
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise ValueError("The template must not be larger than the image.")
    return feature.match_template(image, template, pad_input=True)


def correlation_to_image(correlation: np.ndarray) -> Image.Image:
    """Stretch the correlation map min-max to [0, 255], like matplotlib's imshow(cmap='gray')."""
    low, high = float(correlation.min()), float(correlation.max())
    if high - low < 1e-12:
        scaled = np.zeros_like(correlation)
    else:
        scaled = (correlation - low) / (high - low)
    return Image.fromarray(img_as_ubyte(scaled), mode="L")


def threshold_matches(correlation: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of the pixels whose correlation is >= threshold."""
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [-1, 1].")
    return correlation >= threshold


def default_threshold(correlation: np.ndarray) -> float:
    """Start at DEFAULT_THRESHOLD, ot just below the peak when the peak is lower."""
    peak = float(correlation.max())
    return round(max(-1.0, min(DEFAULT_THRESHOLD, peak - THRESHOLD_MARGIN)), 2)

def count_regions(matches: np.ndarray) -> int:
    """Number of connected blobs (8-connectivity) in a boolean mask."""
    return int(measure.label(matches, connectivity=2).max())

def preview_factor(shape: tuple[int, int], max_size: tuple[int, int]) -> int:
    """Integer downscale factor so an array of `shape`(H, W) fits in max_size (W, H)"""
    height, width = shape
    max_height, max_width = max_size
    return max(1, math.ceil(max(width / max_width, height / max_height)))

  
def downscale_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Shrink a 2D float image by averaging factor x factor blocks."""
    if factor == 1:
        return image
    return measure.block_reduce(image, (factor, factor), np.mean)
    
    
def downscale_any(mask: np.ndarray, factor: int) -> np.ndarray:
    """Shrink a boolean mask keeping a block True if ANY of its pixels is True
    
    Averaging would make isolated matches (a single bright NCC peak) dissapear."""
    if factor == 1:
        return mask
    return measure.block_reduce(mask, (factor, factor), np.max).astype(bool)
    
def overlay_matches(image: Image.Image, matches: np.ndarray) -> Image.Image:
    """Dimmed grayscale image with matching pixels marked in red."""
    return render_overlay(to_gray_float(image), matches)
    
    
def render_overlay(gray:np.ndarray, matches: np.ndarray) -> Image.Image:
    """overlay_matches from a 2D float array in [0, 1]."""
    if gray.shape != matches.shape:
        raise ValueError("The match mask must have the same shape as the image.")
    dimmed = color.gray2rgb(gray) * DIM_FACTOR
    dimmed[matches] = [1.0, 0.0, 0.0]
    return Image.fromarray(img_as_ubyte(dimmed), mode="RGB")

def template_preview(template: np.ndarray) -> Image.Image:
    """Grayscale PIL image of the template, enlarged if it is very small."""
    preview = Image.fromarray(img_as_ubyte(np.clip(template, 0.0, 1.0)), mode="L")
    scale = max(1, TEMPLATE_PREVIEW_MIN_SIDE // max(preview.size))
    if scale > 1:
        preview = preview.resize(
            (preview.width * scale, preview.height * scale), Image.Resampling.NEAREST
        )
    return preview


def show_template_window(parent: tk.Misc, template: np.ndarray, caption: str) -> None:
    """Show the (rotated) template next to the main window, reusing the window if open."""
    global _template_window
    if _template_window is None or not _template_window.winfo_exists():
        _template_window = tk.Toplevel(parent)
        _template_window.title("Template")
        _template_window.resizable(False, False)
        _template_window.transient(parent.winfo_toplevel())
    for child in _template_window.winfo_children():
        child.destroy()

    photo = ImageTk.PhotoImage(template_preview(template), master=_template_window)
    label = tk.Label(_template_window, image=photo, background="#20242b", padx=8, pady=8)
    label.image = photo  # keep a reference so Tk does not drop the image
    label.pack()
    tk.Label(_template_window, text=caption, padx=8, pady=4).pack()
    _template_window.lift()


def _peak_details(correlation: np.ndarray) -> dict[str, str]:
    row, col = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    return {
        "Max correlation": f"{correlation.max():.4f}",
        "Peak (x, y)": f"({col}, {row})",
    }


# ---------- Tools ----------

class CrossCorrelationTool(ForensicsTool):
    tool_id = "cross_correlation"
    title = "Normalized cross-correlation"
    category = "Template matching"
    description = (
        "Load a template from a file or select it on the image, rotate it with a dial, "
        "and show the normalized cross-correlation map (bright = good match)."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        gray = to_gray_float(document.current)
        
        source = ask_template_source(parent)
        if source is None:
            return None
        if source == SOURCE_FILE:
            picked = self._template_from_file(parent)
        else:
            picked = self._template_from_selection(parent, document.current, gray)
        if picked is None:
            return None
        template, template_source = picked
        
        angle = ask_angle(parent, lambda value: template_preview(rotate_template(template, value)))
        if angle is None:
            return None

        rotated = rotate_template(template, angle)
        correlation = normalized_cross_correlation(gray, rotated)
        height, width = template.shape
        document.analyses[ANALYSIS_KEY] = CrossCorrelationResult(
            document.current.copy(), correlation, template_source, (height, width), angle, rotated
        )
        show_template_window(parent, rotated, f"{template_source} rotated {angle:g} deg")

        return ToolResult(
            image=correlation_to_image(correlation),
            message=f"Computed NCC with a {width}x{height} template rotated {angle:g} degrees.",
            details={
                "Operation": "Normalized cross-correlation",
                "Template": template_source,
                "Template size (w x h)": f"{width} x {height}",
                "Angle": f"{angle:g} deg",
                **_peak_details(correlation),
            },
        )
        
    @staticmethod
    def _template_from_file(parent: tk.Misc) -> tuple[np.ndarray, str] | None:
        filename = filedialog.askopenfilename(
            title="Open template image",
            filetypes=TEMPLATE_FILE_TYPES, parent=parent
        )
        if not filename:
            return None
        return load_template(filename), Path(filename).name
    
    @staticmethod
    def _template_from_selection(parent: tk.Misc, image: Image.Image, gray: np.ndarray) -> tuple[np.ndarray, str] | None:
        box = ask_region(parent, image)
        if box is None:
            return None
        left, top, right, bottom = box
        name = f"region ({left}, {top}) {right - left}x{bottom - top}"
        return crop_template(gray, box), name


class TemplateMatchThresholdTool(ForensicsTool):
    tool_id = "template_match_threshold"
    title = "Thresholded matches"
    category = "Template matching"
    description = ("Move a slider to mark in red, live over the dimmed image, the pixels whose "
        "correlation reaches the threshold.")
    unavailable_message = NO_NCC_MESSAGE

    def is_available(self, document: ImageDocument) -> bool:
        return ANALYSIS_KEY in document.analyses

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        result: CrossCorrelationResult | None = document.analyses.get(ANALYSIS_KEY)
        if result is None:
            return ToolResult(message=NO_NCC_MESSAGE)

        show_template_window(
            parent, result.template, f"{result.template_source} rotated {result.angle:g} deg"
        )
        gray = to_gray_float(result.source)
        factor = preview_factor(gray.shape, MAX_PREVIEW_SIZE)
        preview_gray = downscale_mean(gray, factor) # computed once, reuse on every move
        
        def render_preview(value: float) -> tuple[Image.Image, str]:
            # Threshold at full resolution, then shrink only the result for display.
            full_matches = threshold_matches(result.correlation, value)
            preview = render_overlay(preview_gray, downscale_any(full_matches, factor))
            return preview, f"{int(full_matches.sum()):,} matching pixels"
        
        threshold = ask_threshold(parent, render_preview, default_threshold(result.correlation))
        if threshold is None:
            return None
        matches = threshold_matches(result.correlation, threshold)
        match_count = int(matches.sum())
        
        return ToolResult(
            image=overlay_matches(result.source, matches),
            message=f"{match_count:,} pixels with correlation >= {threshold:g}.",
            details={
                "Operation": "Thresholded template matches",
                "Template": result.template_source,
                "Angle": f"{result.angle:g} deg",
                "Threshold": f"{threshold:g}",
                "Matching pixels": f"{match_count:,}",
                **_peak_details(result.correlation),
                "Match regions": str(count_regions(matches)),
            },
        )
    