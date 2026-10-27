"""Match local features (ORB / SIFT) between the current image and a second image."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage import img_as_float
from skimage.feature import ORB, SIFT, match_descriptors, plot_matched_features

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .corners import draw_corners
from .geometric_verification import (
    DEFAULT_MIN_INLIERS,
    DEFAULT_MODEL,
    DEFAULT_RESIDUAL_THRESHOLD,
    MIN_SAMPLES,
    MODELS,
    RansacParams,
)

DIALOG_TITLE = "Feature Matching"
METHODS = ("orb", "sift") # ORB -> bit strings; Hamming / SIFT -> hisograms; Euclidean
METRICS = {"orb": "hamming", "sift": "euclidean"} 
DESCRIPTOR_LENGTHS = {"orb": 256, "sift": 128}
DEFAULT_ORB_KEYPOINTS = 200
DEFAULT_SIFT_OCTAVES = 4
DEFAULT_SIFT_SCALES = 3
DEFAULT_SIFT_SIGMA = 1.6
DEFAULT_MAX_RATIO = 0.8
IMAGE_FILE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
FIGURE_SIZE = (10.0, 5.0)
FIGURE_DPI = 100

INLIER_COLOR = "lime"
OUTLIER_COLOR = "red"

@dataclass(frozen = True)
class FeatureParams:
    """ Descriptor configurations (all set to default; methods is the only field used)"""
    method: str = "orb"
    n_keypoints: int = DEFAULT_ORB_KEYPOINTS # ORB
    n_octaves: int = DEFAULT_SIFT_OCTAVES # SIFT
    n_scales: int = DEFAULT_SIFT_SCALES # SIFT
    sigma_min: float = DEFAULT_SIFT_SIGMA # SIFT

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown descriptor '{self.method}'. Use one of: {', '.join(METHODS)}.")
        if self.n_keypoints < 1 or self.n_octaves < 1 or self.n_scales < 1:
            raise ValueError("Keypoints, octaves and scales must be positive integers.")
        if self.sigma_min <= 0:
            raise ValueError("Sigma must be positive.")

    @property
    def metric(self) -> str:
        return METRICS[self.method]

# ---------- Pure functions ----------
def to_gray_float(image: Image.Image) -> np.ndarray:
    return img_as_float(np.asarray(image.convert("L")))


def load_gray(path:str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return to_gray_float(image)


def _build_extraction(params: FeatureParams) -> ORB | SIFT:
    if params.method == "orb":
        return ORB(n_keypoints=params.n_keypoints)
    return SIFT(n_octaves=params.n_octaves, n_scales=params.n_scales, sigma_min=params.sigma_min)


def extract_features(gray: np.ndarray, params: FeatureParams) -> tuple[np.ndarray, np.ndarray]:
    """Detect keypoints in ONE image and compute their descriptors.
    
    Returns a tuple of arrays with the keypoinys and descriptors:
    - keypoints have shape (N, 2) with (row, cols) coordinates
    - descriptors have shape (N, 256) in bool for ORB and (N, 128) in uint8 for SIFT"""

    if gray.ndim != 2:
        raise ValueError("Feature extraction needs to be a 2D (grayscale) array.")

    extractor = _build_extraction(params)
    try:
        extractor.detect_and_extract(gray)
    except RuntimeError: # skimage raises when it finds no features at all
        return np.empty((0, 2)), np.empty((0, DESCRIPTOR_LENGTHS[params.method]))
    return extractor.keypoints, extractor.descriptors

def match_features(
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        metric: str,
        max_ratio: float = DEFAULT_MAX_RATIO,
        cross_check: bool = True,
) -> np.ndarray:
    """Nearest-neighbor matching filter. It returns an int array (K, 2):
    row [i, j] pairs keypoint i of image 1 with keypoint j of image 2."""

    if not 0.0 < max_ratio <= 1.0:
        raise ValueError("Max ratio must be in (0, 1].")
    if descriptors1.ndim != 2 or descriptors2.ndim != 2 or descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Both descriptor sets must be 2-D and have the same length.")
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return np.empty((0, 2), dtype=np.intp)
    return match_descriptors(
        descriptors1, descriptors2, metric=metric, max_ratio=max_ratio, cross_check=cross_check
    )

def draw_matches(
        gray1: np.ndarray,
        gray2: np.ndarray,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        matches: np.ndarray,
        title: str = "",
        inliers: np.ndarray | None = None,
) -> Image.Image:
    """Render both images side by side with keypoints and match lines as an RGB image.
    
    If `inliers` (bool mask, one flag per match) is given, inlier matches are drawn in green and outliers in red;
    otherwise every match gets a random color.
    """
    figure = Figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot()
    if inliers is None:
        plot_matched_features(
            gray1, gray2,
            keypoints0=keypoints1, keypoints1=keypoints2, matches=matches,
            ax=axis, keypoints_color="r",
        )
    else: 
        plot_matched_features(
            gray1, gray2,
            keypoints0=keypoints1, keypoints1 = keypoints2, matches=matches[~inliers],
            ax=axis, keypoints_color="r", matches_color=OUTLIER_COLOR,
        )
        # Second pass only adds inlier lines, so they end up on top of the outliers.
        plot_matched_features(
            gray1, gray2,
            keypoints0=keypoints1, keypoints1 = keypoints2, matches=matches[inliers],
            ax=axis, matches_color=INLIER_COLOR, only_matches=True,
        )
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())  # (H, W, 4) uint8
    return Image.fromarray(np.ascontiguousarray(rgba[..., :3]))    

def describe_params(params: FeatureParams) -> dict[str, str]:
    """Human-readable parameters for the details panel."""
    if params.method == "orb":
        return {"Descriptor": "ORB", "Keypoints requested": str(params.n_keypoints)}
    return {
        "Descriptor": "SIFT",
        "Octaves": str(params.n_octaves),
        "Scales per octave": str(params.n_scales),
        "Sigma min": f"{params.sigma_min:g}",
    }

# ---------- Dialogs ----------
def ask_feature_params(parent: tk.Misc) -> FeatureParams | None:
    """Ask for the descriptor and its parameters; ``None`` if the user cancels."""
    method = simpledialog.askstring(
        DIALOG_TITLE, "Descriptor (orb / sift):", parent=parent, initialvalue="orb"
    )
    if method is None:
        return None
    method = method.strip().lower()

    if method == "orb":
        n_keypoints = simpledialog.askinteger(
            DIALOG_TITLE, "Number of ORB keypoints:", parent=parent,
            initialvalue=DEFAULT_ORB_KEYPOINTS, minvalue=1,
        )
        if n_keypoints is None:
            return None
        return FeatureParams("orb", n_keypoints=n_keypoints)
    
    if method == "sift":
        n_octaves = simpledialog.askinteger(
            DIALOG_TITLE, "SIFT octaves:", parent=parent,
            initialvalue=DEFAULT_SIFT_OCTAVES, minvalue=1,
        )

        if n_octaves is None:
            return None
        n_scales = simpledialog.askinteger(
            DIALOG_TITLE, "SIFT scales per octave:", parent=parent,
            initialvalue=DEFAULT_SIFT_SCALES, minvalue=1,
        )
        if n_scales is None:
            return None
        sigma_min = simpledialog.askfloat(
            DIALOG_TITLE, "SIFT sigma min:", parent=parent,
            initialvalue=DEFAULT_SIFT_SIGMA, minvalue=0.01,
        )
        if sigma_min is None:
            return None
        return FeatureParams("sift", n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min)

    raise ValueError(f"Unknown descriptor '{method}'. Use one of: {', '.join(METHODS)}.")

def ask_matching_params(parent: tk.Misc) -> tuple[float, bool] | None:
    """Ask for (max_ratio, cross_check); ``None`` if the user cancels."""
    max_ratio = simpledialog.askfloat(
        DIALOG_TITLE, "Max ratio for Lowe's test (0-1, 1 = disabled):", parent=parent,
        initialvalue=DEFAULT_MAX_RATIO, minvalue=0.01, maxvalue=1.0,
    )
    if max_ratio is None:
        return None
    cross_check = messagebox.askyesno(
        DIALOG_TITLE, "Keep only mutual best matches (cross check)?", parent=parent
    )
    return max_ratio, bool(cross_check)

def ask_ransac_params(parent: tk.Misc) -> RansacParams | None:
    """Ask for the RANSAC model, residual threshold and min inliers; ``None`` if cancelled."""   
    model = simpledialog.askstring(DIALOG_TITLE, "Geometric model (homography / fundamental):", parent=parent, initialvalue=DEFAULT_MODEL)
    if model is None:
        return None
    model = model.strip().lower()
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Use one of: {', '.join(MODELS)}.")
    residual_threshold = simpledialog.askfloat(DIALOG_TITLE, "Residual threshold in pixels (max reprojection error of an inlier):", parent=parent, initialvalue=DEFAULT_RESIDUAL_THRESHOLD, minvalue=0.1)
    if residual_threshold is None:
        return None
    min_inliers = simpledialog.askinteger(DIALOG_TITLE, "Min inliers to trust the model:", parent=parent, initialvalue=max(DEFAULT_MIN_INLIERS, MIN_SAMPLES[model]), minvalue=MIN_SAMPLES[model])
    if min_inliers is None:
        return None
    return RansacParams(model, residual_threshold=residual_threshold, min_inliers=min_inliers)


# ---------- Tools ----------
class KeypointsTool(ForensicsTool):
    tool_id = "keypoints"
    title = "Keypoints (ORB / SIFT)"
    category = "Features"
    description = "Detect ORB or SIFT keypoints in the current image, compute their descriptors and draw them."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        params = ask_feature_params(parent)
        if params is None:
            return None

        assert document.current is not None
        keypoints, descriptors = extract_features(to_gray_float(document.current), params)
        return ToolResult(
            image=draw_corners(document.current, keypoints),
            message=f"Extracted {len(keypoints)} {params.method.upper()} keypoints.",
            details={"Operation": "Keypoint extraction",
                     **describe_params(params),
                     "Keypoints": str(len(keypoints)),
                     "Descriptor length": str(descriptors.shape[1]),
                     },
        )


class FeatureMatchingTool(ForensicsTool):
    tool_id = "feature_matching"
    title = "Match with second image"
    category = "Features"
    description = (
        "Detect ORB or SIFT keypoints in the current image and in a second image file, "
        "match their descriptors and draw matches side by side"
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        filename = filedialog.askopenfilename(
            title="Open second image", filetypes=IMAGE_FILE_TYPES, parent=parent
        )
        if not filename:  # "" (or ()) when cancelled
            return None
        params = ask_feature_params(parent)
        if params is None:
            return None
        matching = ask_matching_params(parent)
        if matching is None:
            return None
        max_ratio, cross_check = matching

        gray1 = to_gray_float(document.current)
        gray2 = load_gray(filename)
        keypoints1, descriptors1 = extract_features(gray1, params)
        keypoints2, descriptors2 = extract_features(gray2, params)
        matches = match_features(descriptors1, descriptors2, params.metric, max_ratio, cross_check)

        second_name = Path(filename).name
        title = f"Current image <-> {second_name} (matches={len(matches)})"
        return ToolResult(
            image=draw_matches(gray1, gray2, keypoints1, keypoints2, matches, title),
            message= (
                f"{len(matches)} matches between {len(keypoints1)} and {len(keypoints2)} "
                f"{params.method.upper()} keypoints."
            ),
            details={"Operation": "Feature matching",
                     "Second image": second_name,
                     **describe_params(params),
                     "Metric": params.metric,
                     "Max ratio":f"{max_ratio:g}",
                     "Cross check": "yes" if cross_check else "no",
                     "Keypoints (Current image)": str(len(keypoints1)),
                     f"Keypoints {second_name}": str(len(keypoints2)),
                     "Matches": str(len(matches)),

                     },
        )