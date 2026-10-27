from __future__ import annotations
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import numpy as np
from PIL import Image, ImageDraw
from skimage import exposure, feature, img_as_float
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cross_correlation import show_template_window

METHODS = ("fast", "harris")
DEFAULT_METHOD = "fast"
DEFAULT_MIN_DISTANCE = 5
DEFAULT_THRESHOLD_REL = 0.1
CORNER_COLOR = (255, 0, 0)
CORNER_RADIUS = 3


def to_gray_float(image: Image.Image) -> np.ndarray:
    return img_as_float(np.asarray(image.convert("L")))


def corner_response(image: Image.Image, method: str = DEFAULT_METHOD) -> np.ndarray:
    gray = to_gray_float(image)
    if method == "fast":
        return feature.corner_fast(gray)
    if method == "harris":
        return feature.corner_harris(gray)
    raise ValueError(f"Unknown corner detector '{method}'. Use one of: {', '.join(METHODS)}.")


def find_corners(
    response: np.ndarray,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    threshold_rel: float = DEFAULT_THRESHOLD_REL,
) -> np.ndarray:
    if min_distance < 1:
        raise ValueError("The minimum distance must be >= 1.")
    if not 0.0 <= threshold_rel <= 1.0:
        raise ValueError("The peak threshold must be in [0, 1].")
    return feature.corner_peaks(response, min_distance=min_distance, threshold_rel=threshold_rel)


def draw_corners(image: Image.Image, corners: np.ndarray, radius: int = CORNER_RADIUS) -> Image.Image:
    output = image.convert("L").convert("RGB")
    draw = ImageDraw.Draw(output)
    for row, col in corners:
        box = (col - radius, row - radius, col + radius, row + radius)
        draw.ellipse(box, outline=CORNER_COLOR)
    return output


class CornerDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc) -> None:
        self.options: tuple[str, int, float] | None = None
        super().__init__(parent, "Corner detection")

    def body(self, master: tk.Frame) -> tk.Widget:
        frame = ttk.LabelFrame(master, text="Parameters", padding=8)
        frame.pack(fill="x", padx=8, pady=8)
        self.method = tk.StringVar(master, value=DEFAULT_METHOD)
        self.min_distance = tk.StringVar(master, value=str(DEFAULT_MIN_DISTANCE))
        self.threshold_rel = tk.StringVar(master, value=str(DEFAULT_THRESHOLD_REL))

        ttk.Label(frame, text="Detector:").grid(row=0, column=0, sticky="w", pady=2)
        combobox = ttk.Combobox(frame, values=METHODS, textvariable=self.method, state="readonly", width=8)
        combobox.grid(row=0, column=1, padx=(6, 0), pady=2)
        ttk.Label(frame, text="Min distance (px):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(frame, from_=1, to=100, increment=1, textvariable=self.min_distance, width=8).grid(
            row=1, column=1, padx=(6, 0), pady=2
        )
        ttk.Label(frame, text="Peak threshold (relative, 0-1):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.threshold_rel, width=8).grid(
            row=2, column=1, padx=(6, 0), pady=2
        )
        return combobox

    def validate(self) -> bool:
        try:
            min_distance = int(self.min_distance.get())
            threshold_rel = float(self.threshold_rel.get())
            valid = min_distance >= 1 and 0.0 <= threshold_rel <= 1.0
        except ValueError:
            valid = False
        if not valid:
            messagebox.showwarning(
                self.title(),
                "Min distance must be an integer >= 1 and the threshold a number in [0, 1].",
                parent=self,
            )
            return False
        self.options = (self.method.get(), min_distance, threshold_rel)
        return True


def _ask_options(parent: tk.Misc) -> tuple[str, int, float] | None:
    return CornerDialog(parent).options


def _show_response(parent: tk.Misc, response: np.ndarray, caption: str) -> None:
    rescaled = exposure.rescale_intensity(response, out_range=(0.0, 1.0))
    show_template_window(parent, rescaled, caption)


class CornerDetectionTool(ForensicsTool):
    tool_id = "corner_detection"
    title = "Corner detection"
    category = "Features"
    description = (
        "Detect corners with FAST or Harris and keep the response peaks above a "
        "configurable threshold, drawn as red circles."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_options(parent)
        if options is None:
            return None
        method, min_distance, threshold_rel = options

        assert document.current is not None
        response = corner_response(document.current, method)
        corners = find_corners(response, min_distance, threshold_rel)
        _show_response(parent, response, f"{method.upper()} response")
        return ToolResult(
            image=draw_corners(document.current, corners),
            message=f"Detected {len(corners):,} {method.upper()} corners (threshold={threshold_rel:g}).",
            details={
                "Operation": "Corner detection",
                "Detector": method.upper(),
                "Min distance": f"{min_distance} px",
                "Peak threshold (relative)": f"{threshold_rel:g}",
                "Max response": f"{response.max():.4f}",
                "Corners": f"{len(corners):,}",
            },
        )
