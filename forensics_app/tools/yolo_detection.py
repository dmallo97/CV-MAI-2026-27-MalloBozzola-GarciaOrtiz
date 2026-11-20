"""Object detection with a YOLO model pretrained by Ultralytics on COCO (80 classes)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_classification import figure_to_image
from .face_detection import FaceDetectionDialog, pil_to_bgr

if TYPE_CHECKING:  # only for the annotations, see the note above load_yolo
    from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = PROJECT_ROOT / "models" / "yolo11n.pt"  # nano model; Ultralytics downloads it the first time

# Ultralytics defaults: keep boxes with confidence >= 0.25, NMS drops boxes with IoU > 0.7 with a better one
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.7
COLORMAP = "tab20"  # one colour per class, as in the Set 9.1 slide
FIGURE_DPI = 100


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) top-left and bottom-right corners in pixels
    class_id: int  # index into the model names (COCO: 0 = person, 5 = bus, ...)
    label: str
    score: float  # confidence of the model


# ---------- Pure functions ----------

# Ultralytics is imported inside: it loads PyTorch, which takes a while and the rest of the app does not need it.
def load_yolo(path: Path = MODEL_FILE) -> YOLO:
    """Pretrained YOLO model; the official weights (yolo11n.pt, ...) are downloaded to `path` if missing."""
    from ultralytics import YOLO

    path.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(str(path))


def parse_detections(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, names: dict[int, str]) -> list[Detection]:
    """Turn the (N, 4) xyxy boxes, (N,) scores and (N,) class ids into Detection objects, best score first."""
    order = np.argsort(scores)[::-1]
    return [
        Detection(
            box=tuple(float(value) for value in boxes[index]),
            class_id=int(classes[index]),
            label=names[int(classes[index])],
            score=float(scores[index]),
        )
        for index in order
    ]


def detect_objects(model: YOLO, frame: np.ndarray, confidence: float, iou: float) -> list[Detection]:
    """Run the model on a BGR frame (the OpenCV order Ultralytics expects for numpy arrays)."""
    result = model.predict(frame, conf=confidence, iou=iou, verbose=False)[0]
    boxes = result.boxes  # tensors: xyxy (N, 4), conf (N,), cls (N,)
    return parse_detections(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), result.names)


# "Why are people blue?": results[0].plot() returns a BGR image (OpenCV order) and plt.imshow reads it as RGB,
# so red and blue are swapped. Here the boxes are drawn with matplotlib over the original RGB image instead.
def plot_detections(image: Image.Image, detections: list[Detection]) -> plt.Figure:
    """Image with a rectangle per detection and a '<label> <score>' tag, coloured by class."""
    width, height = image.size
    figure, axis = plt.subplots(figsize=(width / FIGURE_DPI, height / FIGURE_DPI + 0.5), dpi=FIGURE_DPI)
    axis.imshow(image.convert("RGB"))
    colors = matplotlib.colormaps[COLORMAP]
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = colors(detection.class_id % colors.N)
        axis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"))
        axis.text(
            x1, y1 - 10, f"{detection.label} {detection.score:.2f}",
            color="white", fontsize=12, backgroundcolor=color, clip_on=True,
        )
    axis.set_title("Detected Objects")
    axis.axis("off")
    figure.tight_layout()
    return figure


def describe_detections(detections: list[Detection]) -> dict[str, str]:
    """Count per class and the box and score of every detection."""
    details = {
        f"Count {label}": str(count) for label, count in sorted(Counter(d.label for d in detections).items())
    }
    for number, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.box
        details[f"Object {number}"] = f"{detection.label} {detection.score:.2f} (x1:{x1:.0f}, y1:{y1:.0f}, x2:{x2:.0f}, y2:{y2:.0f})"
    return details


# ---------- Dialog ----------

def _ask_options(parent: tk.Misc, title: str) -> tuple[float, float] | None:
    """(confidence, IoU) thresholds: the same two numbers as YuNet's score and NMS thresholds."""
    return FaceDetectionDialog(parent, title, DEFAULT_CONFIDENCE, DEFAULT_IOU).options


# ---------- Tools ----------

class YoloDetectionTool(ForensicsTool):
    tool_id = "yolo_detection"
    title = "Object detection (YOLO)"
    category = "Object detection"
    description = (
        "Detect the 80 COCO classes (person, car, bus, dog, ...) with a YOLO11n model pretrained by "
        "Ultralytics: one box per object with its label and confidence."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_options(parent, self.title)
        if options is None:
            return None
        confidence, iou = options

        assert document.current is not None
        # Loading PyTorch + the model (and the first download) block the window: show a busy cursor meanwhile
        window = parent.winfo_toplevel()
        window.configure(cursor="watch")
        window.update_idletasks()
        try:
            model = load_yolo(MODEL_FILE)
            detections = detect_objects(model, pil_to_bgr(document.current), confidence, iou)
        finally:
            window.configure(cursor="")

        figure = plot_detections(document.current, detections)
        image = figure_to_image(figure)
        plt.close(figure)
        width, height = document.current.size
        return ToolResult(
            image=image,
            message=f"Detected {len(detections)} object(s) with YOLO (confidence >= {confidence:g}).",
            details={
                "Operation": "Object detection (YOLO)",
                "Model": MODEL_FILE.name,
                "Image size (w x h)": f"{width} x {height}",
                "Confidence threshold": f"{confidence:g}",
                "IoU threshold (NMS)": f"{iou:g}",
                "Objects": str(len(detections)),
                **describe_detections(detections),
            },
        )
