from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import numpy as np
from PIL import Image, ImageDraw
from skimage import exposure, feature, img_as_float, img_as_ubyte
from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cross_correlation import TEMPLATE_FILE_TYPES, show_template_window

DEFAULT_ORIENTATIONS = 9
DEFAULT_PIXELS_PER_CELL = 6
DEFAULT_CELLS_PER_BLOCK = 3

# Person detection defaults, as in the Set 4.4 instructions
DETECTION_ORIENTATIONS = 16
DETECTION_PIXELS_PER_CELL = 8
DETECTION_CELLS_PER_BLOCK = 2
DETECTION_STEP = 4
EUCLIDEAN_COLOR = (255, 0, 0)
DOT_PRODUCT_COLOR = (0, 200, 255)
BOX_WIDTH = 2


def compute_hog(
    image: Image.Image,
    orientations: int = DEFAULT_ORIENTATIONS,
    pixels_per_cell: int = DEFAULT_PIXELS_PER_CELL,
    cells_per_block: int = DEFAULT_CELLS_PER_BLOCK,
) -> tuple[np.ndarray, np.ndarray]:
    if image.mode == "L":
        pixels, channel_axis = img_as_float(np.asarray(image)), None
    else:
        pixels, channel_axis = img_as_float(np.asarray(image.convert("RGB"))), -1
    return feature.hog(
        pixels,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        visualize=True,
        channel_axis=channel_axis,
    )


def hog_to_image(hog_image: np.ndarray) -> Image.Image:
    rescaled = exposure.rescale_intensity(hog_image, out_range=(0.0, 1.0))
    return Image.fromarray(img_as_ubyte(rescaled), mode="L")


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    score: float


def sliding_windows(
    image_size: tuple[int, int], window_size: tuple[int, int], step: int
) -> list[tuple[int, int]]:
    (width, height), (window_w, window_h) = image_size, window_size
    if window_w > width or window_h > height:
        raise ValueError("The person template must not be larger than the image.")
    if step < 1:
        raise ValueError("The step must be >= 1.")
    return [
        (x, y)
        for y in range(0, height - window_h + 1, step)
        for x in range(0, width - window_w + 1, step)
    ]


def detect_person(
    image: Image.Image,
    template: Image.Image,
    step: int = DETECTION_STEP,
    orientations: int = DETECTION_ORIENTATIONS,
    pixels_per_cell: int = DETECTION_PIXELS_PER_CELL,
    cells_per_block: int = DETECTION_CELLS_PER_BLOCK,
) -> tuple[Detection, Detection, int]:
    image, template = image.convert("RGB"), template.convert("RGB")
    corners = sliding_windows(image.size, template.size, step)
    template_fd, _ = compute_hog(template, orientations, pixels_per_cell, cells_per_block)
    euclidean, dot_product = [], []
    for x, y in corners:
        window = image.crop((x, y, x + template.width, y + template.height))
        window_fd, _ = compute_hog(window, orientations, pixels_per_cell, cells_per_block)
        euclidean.append(np.linalg.norm(template_fd - window_fd))
        dot_product.append(np.dot(window_fd, template_fd))

    best_euclidean, best_dot = int(np.argmin(euclidean)), int(np.argmax(dot_product))
    return (
        Detection(*corners[best_euclidean], float(euclidean[best_euclidean])),
        Detection(*corners[best_dot], float(dot_product[best_dot])),
        len(corners),
    )


def draw_detections(
    image: Image.Image, window_size: tuple[int, int], euclidean: Detection, dot_product: Detection
) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    width, height = window_size
    for detection, color in ((dot_product, DOT_PRODUCT_COLOR), (euclidean, EUCLIDEAN_COLOR)):
        box = (detection.x, detection.y, detection.x + width - 1, detection.y + height - 1)
        draw.rectangle(box, outline=color, width=BOX_WIDTH)
    return output


class HogDialog(simpledialog.Dialog):
    FIELDS = (
        ("Orientations:", DEFAULT_ORIENTATIONS),
        ("Pixels per cell:", DEFAULT_PIXELS_PER_CELL),
        ("Cells per block:", DEFAULT_CELLS_PER_BLOCK),
    )

    def __init__(
        self,
        parent: tk.Misc,
        title: str = "HOG extraction",
        fields: tuple[tuple[str, int], ...] = FIELDS,
    ) -> None:
        self.fields = fields
        self.options: tuple[int, ...] | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        frame = ttk.LabelFrame(master, text="Parameters", padding=8)
        frame.pack(fill="x", padx=8, pady=8)
        self.values = []
        spinboxes = []
        for row, (label, default) in enumerate(self.fields):
            variable = tk.StringVar(master, value=str(default))
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            spinbox = ttk.Spinbox(frame, from_=1, to=64, increment=1, textvariable=variable, width=8)
            spinbox.grid(row=row, column=1, padx=(6, 0), pady=2)
            self.values.append(variable)
            spinboxes.append(spinbox)
        return spinboxes[0]

    def validate(self) -> bool:
        try:
            options = tuple(int(value.get()) for value in self.values)
            valid = all(option >= 1 for option in options)
        except ValueError:
            valid = False
        if not valid:
            messagebox.showwarning(self.title(), "All parameters must be integers >= 1.", parent=self)
            return False
        self.options = options
        return True


def _ask_options(parent: tk.Misc) -> tuple[int, int, int] | None:
    return HogDialog(parent).options


DETECTION_FIELDS = (
    ("Window step (px):", DETECTION_STEP),
    ("Orientations:", DETECTION_ORIENTATIONS),
    ("Pixels per cell:", DETECTION_PIXELS_PER_CELL),
    ("Cells per block:", DETECTION_CELLS_PER_BLOCK),
)


def _ask_template_path(parent: tk.Misc) -> str:
    return filedialog.askopenfilename(
        title="Open person template", filetypes=TEMPLATE_FILE_TYPES, parent=parent
    )


def _ask_detection_options(parent: tk.Misc) -> tuple[int, int, int, int] | None:
    return HogDialog(parent, "Person detection", DETECTION_FIELDS).options


def _show_template_hog(parent: tk.Misc, hog_image: np.ndarray, caption: str) -> None:
    rescaled = exposure.rescale_intensity(hog_image, out_range=(0.0, 1.0))
    show_template_window(parent, rescaled, caption)


class HogTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG extraction"
    category = "Feature descriptors"
    description = "Compute the Histogram of Oriented Gradients and show its visualization."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_options(parent)
        if options is None:
            return None
        orientations, pixels_per_cell, cells_per_block = options

        assert document.current is not None
        features, hog_image = compute_hog(document.current, orientations, pixels_per_cell, cells_per_block)
        cells_y, cells_x = (side // pixels_per_cell for side in hog_image.shape)
        return ToolResult(
            image=hog_to_image(hog_image),
            message=f"Extracted {features.size:,} HOG features with {orientations} orientations.",
            details={
                "Operation": "HOG extraction",
                "Orientations": str(orientations),
                "Pixels per cell": f"{pixels_per_cell} x {pixels_per_cell}",
                "Cells per block": f"{cells_per_block} x {cells_per_block}",
                "Cells (w x h)": f"{cells_x} x {cells_y}",
                "Blocks (w x h)": f"{cells_x - cells_per_block + 1} x {cells_y - cells_per_block + 1}",
                "Feature vector length": f"{features.size:,}",
            },
        )


class HogPersonDetectionTool(ForensicsTool):
    tool_id = "hog_person_detection"
    title = "Person detection (HOG)"
    category = "Feature descriptors"
    description = (
        "Load a person template and find the window whose HOG descriptor is closest "
        "(Euclidean distance, red) and most similar (dot product, blue)."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        filename = _ask_template_path(parent)
        if not filename:  # the dialog returns "" (or ()) when cancelled
            return None
        options = _ask_detection_options(parent)
        if options is None:
            return None
        step, orientations, pixels_per_cell, cells_per_block = options

        assert document.current is not None
        with Image.open(filename) as opened:
            template = opened.convert("RGB")
        template_source = Path(filename).name
        try:
            euclidean, dot_product, windows = detect_person(
                document.current, template, step, orientations, pixels_per_cell, cells_per_block
            )
        except ValueError as error:
            return ToolResult(message=f"Person detection failed: {error}")

        template_fd, template_hog = compute_hog(template, orientations, pixels_per_cell, cells_per_block)
        _show_template_hog(parent, template_hog, f"HOG of {template_source}")
        width, height = template.size
        return ToolResult(
            image=draw_detections(document.current, template.size, euclidean, dot_product),
            message=(
                f"Compared {windows:,} windows: Euclidean best at ({euclidean.x}, {euclidean.y}), "
                f"dot product best at ({dot_product.x}, {dot_product.y})."
            ),
            details={
                "Operation": "HOG person detection",
                "Template": template_source,
                "Window size (w x h)": f"{width} x {height}",
                "Step": f"{step} px",
                "Orientations": str(orientations),
                "Pixels per cell": f"{pixels_per_cell} x {pixels_per_cell}",
                "Cells per block": f"{cells_per_block} x {cells_per_block}",
                "Feature vector length": f"{template_fd.size:,}",
                "Windows evaluated": f"{windows:,}",
                "Euclidean best (x, y)": f"({euclidean.x}, {euclidean.y})",
                "Euclidean distance": f"{euclidean.score:.4f}",
                "Dot product best (x, y)": f"({dot_product.x}, {dot_product.y})",
                "Dot product": f"{dot_product.score:.4f}",
            },
        )
