"""Geometric verification of features mathces using RANSAC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, ProjectiveTransform

MODELS = ("homography", "fundamental")
MIN_SAMPLES = {"homography": 4, "fundamental": 8}
MODEL_CLASSES = {"homography": ProjectiveTransform, "fundamental": FundamentalMatrixTransform}
DEFAULT_MODEL = "homography"
DEFAULT_RESIDUAL_THRESHOLD = 3.0
DEFAULT_MAX_TRIALS = 1000
DEFAULT_MIN_INLIERS = 10
# A homography that sacles an area by more than this factor (of flips it) would be rejected.
MAX_AREA_CHANGE = 25
RANSAC_SEED = 0 # fixed
 
Model = ProjectiveTransform | FundamentalMatrixTransform

@dataclass(frozen=True)
class RansacParams:
    """Configuration of the RANSAC geometric validation"""
    
    model: str = DEFAULT_MODEL
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD
    max_trials: int = DEFAULT_MAX_TRIALS
    min_inliers: int = DEFAULT_MIN_INLIERS
    
    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"Unknown model '{self.model}'. Use one of {','.join(MODELS)}.")
        if self.residual_threshold <= 0:
            raise ValueError("The residual threshold must be positive.")
        if self.max_trials < 1:
            raise ValueError("Max trials must be a positive integer.")
        if self.min_inliers < self.min_samples:
            raise ValueError(f"Min inliers must be at least {self.min_samples} for a {self.model}.")
        
    @property
    def min_samples(self) -> int:
        return MIN_SAMPLES[self.model]
    
@dataclass(frozen = True)
class VerificationResult:
    """Outcome of RANSAC over a set of matches."""
    inliers: np.ndarray # bool, shape (K,), one flag per match (match/no match)
    model: Model | None # None when no model could be fitted
    reliable: bool # True when there are at least `min_inliers` inliers
    
    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())
    
    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / len(self.inliers) if len(self.inliers) else 0.0
    
    
def matched_points(keypoints1: np.ndarray, keypoints2: np.ndarray, matches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of the matched keypoints as (x, y) arrays of shape (K, 2).
    
    scikit-image keypoints are (row, col) = (y, x), but geometric transforms work in (x, y), so the columns are shaped with ```[:,::-1]``
    to invert the matrix."""
    
    src = keypoints1[matches[:, 0]][:, ::-1]
    dst = keypoints2[matches[:, 1]][:, ::-1]
    return src.astype(float), dst.astype(float)
        
        
def is_plausible_homography(model: ProjectiveTransform, *_samples: np.ndarray) -> bool:
    """Reject homographies that mirror the image or change its area absurdly.
    
    The determinant of the 2x2 linear part is the local area scale factor:
    negative means a mirror flip, ~0 means the image collapses onto a line.
    """
    det = float(np.linalg.det(model.params[:2, :2]))
    return 1.0 / MAX_AREA_CHANGE < det < MAX_AREA_CHANGE


def verify_matches(keypoints1: np.ndarray, keypoints2: np.ndarray, matches: np.ndarray, params: RansacParams) -> VerificationResult:
    """Split matches into inliers / outliers of the best geometric model found by RANSAC."""
    no_inliers = np.zeros(len(matches), dtype=bool)
    if len(matches) < params.min_samples: # not enough points to fit even one model
        return VerificationResult(no_inliers, None, reliable=False)
    
    src, dst = matched_points(keypoints1, keypoints2, matches)
    model, inliers = ransac(
        (src, dst), 
        MODEL_CLASSES[params.model], 
        min_samples = params.min_samples,
        residual_threshold=params.residual_threshold,
        max_trials = params.max_trials,
        is_model_valid = is_plausible_homography if params.model == "homography" else None,
        rng = RANSAC_SEED,
        )
    if model is None or inliers is None: # every trial failed
        return VerificationResult(no_inliers, None, reliable=False)
    inliers = np.asarray(inliers, dtype=bool)
    return VerificationResult(inliers, model, reliable=int(inliers.sum()) >= params.min_inliers)