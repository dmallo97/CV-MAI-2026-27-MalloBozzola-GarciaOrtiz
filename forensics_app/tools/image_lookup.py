"""Find inside a folder, the image that best matches the current one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import UnidentifiedImageError
from skimage import transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .feature_matching import(
    DIALOG_TITLE,
    FeatureParams,
    ask_feature_params,
    ask_matching_params,
    ask_ransac_params,
    describe_params,
    draw_matches,
    extract_features,
    load_gray,
    match_features,
    to_gray_float,
)
from .geometric_verification import RansacParams, VerificationResult, verify_matches

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
# Images are shrunk so their longest side is at most this many pixels before
# extracting features: faster and every image is compared at a similar scale
DEFAULT_MAX_SIDE = 800
TOP_N_DETAILS = 5 # how many ranked candidates are listed in the details panel

@dataclass(frozen=True)
class LookupSettings:
    """Everything that controls how two images are compared."""
    
    features: FeatureParams
    max_ratio: float
    cross_check: bool
    ransac: RansacParams | None # None -> baseline ranking by number of matches
    max_side: int = DEFAULT_MAX_SIDE
    
    def __post_init__(self) -> None:
        if self.max_side < 32:
            raise ValueError("Max side must be at least 32 pixels.")
        
@dataclass(frozen=True)
class ReferenceFeatures:
    """Keypoints of the reference image, computed only once per search."""
    
    gray: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    
@dataclass(frozen=True)
class Candidate:
    """Comparison of the reference against one image of the folder.
    
    The pixels are NOT stored: with hundreds of images that would waste a lot of memory.
    Only the best candidate is reloaded later to draw it. 
    """
    
    path: Path
    keypoints: np.ndarray # (N, 2) (row, col) in the resized candidate image
    matches: np.ndarray # (K, 2) indices [reference keypoint, candidate keypoint]
    verification: VerificationResult | None # None when RANSAC is disabled
    
    @property
    def n_matches(self) -> int:
        return len(self.matches)
    
    @property
    def n_inliers(self) -> int:
        return self.verification.n_inliers if self.verification is not None else 0
    
    @property
    def score(self) -> int:
        """Number used to rank the candidates (higher = more similar)."""
        if self.verification is None:
            return self.n_matches # baseline from the assignment
        # Too few inliers can be product of pure chance, RANSAC always explain 4 random points
        return self.n_inliers if self.verification.reliable else 0
    
# ---------- Pure functions ----------

def list_image_files(folder:str | Path, exclude: str | Path | None = None) -> list[Path]:
    """Image files directly inside `folder`(not recursive), sorted by name.
    
    `exclude` (usually the reference file itself) is left out of the list.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"'{folder}' is not a folder.")
    excluded = Path(exclude).resolve() if exclude is not None else None
    return sorted(
        (
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.resolve() != excluded
        ),
        key=lambda path: path.name.lower())
    
    
def limit_size(gray: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale the 2D omgae so its longest side is at most max_side (never enlarges)."""
    scale = max_side / max(gray.shape)
    if scale >= 1.0:
        return gray
    return transform.rescale(gray, scale, anti_aliasing=True)

   
def prepare_reference(gray: np.ndarray, settings: LookupSettings) -> ReferenceFeatures:
    """Resize the reference image and extract its features once."""
    small = limit_size(gray, settings.max_side)
    keypoints, descriptors = extract_features(small, settings.features)
    return ReferenceFeatures(small, keypoints, descriptors)


def compare_with_reference(gray: np.ndarray, path: Path, reference: ReferenceFeatures, settings: LookupSettings) -> Candidate:
    """Extract, match and (optionally) verify one candidate image."""
    small = limit_size(gray, settings.max_side)
    keypoints, descriptors = extract_features(small, settings.features)
    matches = match_features(reference.descriptors, descriptors, settings.features.metric, settings.max_ratio, settings.cross_check)
    verification = None
    if settings.ransac is not None:
        verification = verify_matches(reference.keypoints, keypoints, matches, settings.ransac)
    return Candidate(path, keypoints, matches, verification)
    
    
def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Best first by score, then by raw matches, then by name (stable and deterministic)."""
    return sorted(candidates, key=lambda c: (-c.score, -c.n_matches, c.path.name.lower()))
    
    
def search_folder(paths: list[Path], reference: ReferenceFeatures, settings: LookupSettings) -> tuple[list[Candidate], list[Path]]:
    """Compare the reference with every path. Returns (ranked candidates, skipped paths)."""
    candidates: list[Candidate] =  []
    skipped: list[Path] = []
    for path in paths:
        try:
            gray = load_gray(path)
        except (OSError, UnidentifiedImageError): # corrupt or not really an image
            skipped.append(path)
            continue
        candidates.append(compare_with_reference(gray, path, reference, settings))
    return rank_candidates(candidates), skipped

def describe_candidate(candidate: Candidate) -> str:
    """Short text for one row of the ranking of candidates."""
    if candidate.verification is None:
        return f"{candidate.n_matches} matches"
    text = f"{candidate.n_inliers} inliers / {candidate.n_matches} matches"
    return text if candidate.verification.reliable else f"{text} (unreliable)"


def describe_ranking(settings: LookupSettings) -> str:
    if settings.ransac is None:
        return "Number of matches"
    return f"RANSAC inliers ({settings.ransac.model})"

# ---------- Tool ----------
class ImageLookupTool(ForensicsTool):
    tool_id = "image_lookup"
    title = "Lookup closest image in folder"
    category = "Features"
    description = (
        "Compare the current image with every image of a folder using ORB or SIFT matches (optionally verified with RANSAC) and show the most similar one."
    )
    
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        
        folder = filedialog.askdirectory(
            title="Folder with the images to search",
            parent=parent, mustexist = True
        )
        if not folder: 
            return None
        paths = list_image_files(folder, exclude=document.path)
        if not paths:
            return ToolResult(message=f"No images found in '{Path(folder).name}'.")
        
        settings = self._ask_settings(parent)
        if settings is None:
            return None
        
        # The search blocks the window for a few seconds: show a busy cursor meanwhile
        window = parent.winfo_toplevel()
        window.configure(cursor="watch")
        window.update_idletasks()
        try:
            reference = prepare_reference(to_gray_float(document.current), settings)
            ranking, skipped = search_folder(paths, reference, settings)
        finally:
            window.configure(cursor="")
            
        details: dict[str, str] = {
            "Operation": "Closest image lookup",
            "Folder": Path(folder).name,
            "Images compared": str(len(ranking)),
            "Skipped (unreadable)": str(len(skipped)),
            **describe_params(settings.features),
            "Ranking by": describe_ranking(settings),
            "Reference keypoints": str(len(reference.keypoints)),
        }
        for position, candidate in enumerate(ranking[:TOP_N_DETAILS], start=1):
            details[f"#{position} {candidate.path.name}"] = describe_candidate(candidate)
            
        if not ranking or ranking[0].score == 0:
            return ToolResult(
                message = "No image in the folder is similar enough to the current one.",
                details=details,
            )
        
        best = ranking[0]
        best_gray = limit_size(load_gray(best.path), settings.max_side)
        inliers = best.verification.inliers if best.verification is not None else None
        title = (
            f"Reference: {document.path.name if document.path else 'current image'}"
            f" <---> Best:{best.path.name} ({describe_candidate(best)})"
                 )
        return ToolResult(
            image=draw_matches(reference.gray, best_gray, reference.keypoints, best.keypoints, best.matches, title, inliers=inliers),
            message=f"Closest image: {best.path.name} ({describe_candidate(best)}).",
            details=details,        
        )
    
    @staticmethod
    def _ask_settings(parent: tk.Misc) -> LookupSettings | None:
        features = ask_feature_params(parent)
        if features is None:
            return None
        matching = ask_matching_params(parent)
        if matching is None:
            return None
        max_ratio, cross_check = matching
        use_ransac = messagebox.askyesnocancel(
            DIALOG_TITLE,
            "Verify the matches with RANSAC and rank by inliers?\n"
            "(No = Simplest ranking -> maximum number of matches)",
            parent=parent,
        )
        
        if use_ransac is None:
            return None
        ransac_params = None
        if use_ransac:
            ransac_params = ask_ransac_params(parent)
            if ransac_params is None:
                return None
        return LookupSettings(features, max_ratio, cross_check, ransac_params)