"""Test the YOLO model trained on blood cells (Set 9.2) on an unseen image: ground truth and predicted boxes."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk

import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_classification import figure_to_image
from .face_detection import pil_to_bgr
from .yolo_detection import COLORMAP, FIGURE_DPI, Detection, _ask_options, detect_objects
from .yolo_training import MODEL_FILE, load_trained


@dataclass(frozen=True)
class GroundTruthBox:
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels, like Detection
    class_id: int
    label: str


# ---------- Pure functions ----------

def label_file_for(image_path: Path | None) -> Path | None:
    """<split>/images/<name>.jpg -> <split>/labels/<name>.txt, the YOLO layout written by the 9.2 conversion."""
    if image_path is None or image_path.parent.name != "images":
        return None
    label_file = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return label_file if label_file.is_file() else None


def yolo_to_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    """YOLO (cx, cy, w, h) normalised to [0, 1] -> (x1, y1, x2, y2) in pixels: the inverse of coco_to_yolo_box."""
    cx, cy, w, h = box
    return (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height


def read_ground_truth(label_file: Path, width: int, height: int, names: dict[int, str]) -> list[GroundTruthBox]:
    """One box per '<class> <cx> <cy> <w> <h>' line, scaled to an image of width x height pixels."""
    boxes = []
    for line in label_file.read_text().splitlines():
        values = line.split()
        if not values:
            continue
        class_id = int(values[0])
        box = yolo_to_xyxy(tuple(float(value) for value in values[1:5]), width, height)
        boxes.append(GroundTruthBox(box=box, class_id=class_id, label=names.get(class_id, str(class_id))))
    return boxes


def plot_ground_truth_and_predictions(
    image: Image.Image, ground_truth: list[GroundTruthBox], detections: list[Detection]
) -> plt.Figure:
    """Image with the ground truth boxes (solid) and the predicted ones (dashed), coloured by class."""
    width, height = image.size
    figure, axis = plt.subplots(figsize=(width / FIGURE_DPI + 1, height / FIGURE_DPI + 1), dpi=FIGURE_DPI)
    axis.imshow(image.convert("RGB"))
    colors = matplotlib.colormaps[COLORMAP]
    for truth in ground_truth:
        x1, y1, x2, y2 = truth.box
        color = colors(truth.class_id % colors.N)
        axis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"))
        axis.text(x1, y1, truth.label, color=color, fontsize=10, va="bottom", clip_on=True)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = colors(detection.class_id % colors.N)
        axis.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, linestyle="--", edgecolor=color, facecolor="none"
        ))
        axis.text(x2, y2, f"{detection.score:.2f}", color=color, fontsize=8, ha="right", va="bottom", clip_on=True)
    legend = [
        Line2D([], [], color="black", linewidth=2, label="Ground truth"),
        Line2D([], [], color="black", linewidth=1.5, linestyle="--", label="Prediction"),
    ]
    axis.legend(handles=legend, loc="upper right", fontsize=8)
    axis.set_title("Ground truth vs prediction")
    figure.tight_layout()
    return figure


def count_per_class(
    ground_truth: list[GroundTruthBox], detections: list[Detection], names: dict[int, str]
) -> dict[str, str]:
    """'Count <class>' -> 'GT n / Pred m' for every class of the model."""
    truth_counts = Counter(truth.label for truth in ground_truth)
    predicted_counts = Counter(detection.label for detection in detections)
    return {
        f"Count {name}": f"GT {truth_counts[name]} / Pred {predicted_counts[name]}"
        for _, name in sorted(names.items())
    }


# ---------- Tools ----------

class YoloTestingTool(ForensicsTool):
    tool_id = "yolo_testing"
    title = "Test YOLO (blood cells)"
    category = "Object detection"
    description = (
        "Run the YOLO model trained on blood cells (models/yolo_blood_cell.pt) on an unseen image and draw "
        "its predictions over the ground truth boxes. Open an image from datasets/blood_cell_yolo/test/images "
        "so its labels file is found."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_options(parent, self.title)
        if options is None:
            return None
        confidence, iou = options

        assert document.current is not None
        # Loading PyTorch + the model block the window: show a busy cursor meanwhile
        window = parent.winfo_toplevel()
        window.configure(cursor="watch")
        window.update_idletasks()
        try:
            model = load_trained(MODEL_FILE)
            detections = detect_objects(model, pil_to_bgr(document.current), confidence, iou)
        finally:
            window.configure(cursor="")

        width, height = document.current.size
        label_file = label_file_for(document.path)
        ground_truth = read_ground_truth(label_file, width, height, model.names) if label_file else []

        figure = plot_ground_truth_and_predictions(document.current, ground_truth, detections)
        image = figure_to_image(figure)
        plt.close(figure)
        if label_file:
            message = f"Predicted {len(detections)} cell(s), ground truth has {len(ground_truth)}."
        else:
            message = (
                f"Predicted {len(detections)} cell(s). No ground truth found: open the image from a "
                "<split>/images folder of the YOLO dataset to compare with its labels."
            )
        return ToolResult(
            image=image,
            message=message,
            details={
                "Operation": "Test YOLO (blood cells)",
                "Model": MODEL_FILE.name,
                "Image size (w x h)": f"{width} x {height}",
                "Ground truth file": str(label_file) if label_file else "not found",
                "Split": label_file.parent.parent.name if label_file else "unknown",  # 'test' = unseen in training
                "Confidence threshold": f"{confidence:g}",
                "IoU threshold (NMS)": f"{iou:g}",
                **count_per_class(ground_truth, detections, model.names),
            },
        )
