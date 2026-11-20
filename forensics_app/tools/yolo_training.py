"""Train a YOLO detector on the blood cell dataset (PC, RBC, WBC), converting it from COCO to YOLO format first."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import TYPE_CHECKING

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .yolo_detection import MODEL_FILE as PRETRAINED_FILE, PROJECT_ROOT, load_yolo

if TYPE_CHECKING:  # Ultralytics is imported lazily, as in yolo_detection
    from ultralytics import YOLO

DATASETS_DIR = PROJECT_ROOT / "datasets"
COCO_DIR = DATASETS_DIR / "blood_cell_coco"  # Roboflow COCO export: <split>/_annotations.coco.json + images
MODEL_FILE = PROJECT_ROOT / "models" / "yolo_blood_cell.pt"
RUNS_DIR = PROJECT_ROOT / "outputs" / "yolo"  # Ultralytics writes weights, curves and plots of every run here
COCO_ANNOTATIONS = "_annotations.coco.json"
SPLITS = ("train", "valid", "test")
DATA_FILE = "data.yaml"

DEFAULT_EPOCHS = 10
DEFAULT_IMAGE_SIZE = 640  # the Roboflow images are 640 x 640


# ---------- Pure functions ----------

# COCO: one JSON per split with every box as [x, y, w, h] in pixels (top-left corner) and a category id.
# YOLO: one .txt per image with a line "<class> <cx> <cy> <w> <h>" per box, centre and size divided by the image size.
def coco_to_yolo_box(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] in pixels -> YOLO (cx, cy, w, h) normalised to [0, 1]."""
    x, y, w, h = bbox
    return (x + w / 2) / width, (y + h / 2) / height, w / width, h / height


def coco_class_names(categories: list[dict]) -> dict[int, str]:
    """COCO category id -> name, without the Roboflow parent category (supercategory 'none', it has no boxes)."""
    return {category["id"]: category["name"] for category in categories if category["supercategory"] != "none"}


def convert_split(source: Path, output: Path, class_ids: dict[int, int]) -> tuple[int, int]:
    """Copy the images of a COCO split to output/images and write output/labels/<image>.txt.
    class_ids maps the COCO category id to the YOLO class (0, 1, ...). Returns the images and boxes written.
    """
    coco = json.loads((source / COCO_ANNOTATIONS).read_text())
    boxes_per_image = defaultdict(list)
    for annotation in coco["annotations"]:
        boxes_per_image[annotation["image_id"]].append(annotation)

    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for image in coco["images"]:
        shutil.copy2(source / image["file_name"], output / "images" / image["file_name"])
        lines = []
        for annotation in boxes_per_image[image["id"]]:
            box = coco_to_yolo_box(annotation["bbox"], image["width"], image["height"])
            lines.append(f"{class_ids[annotation['category_id']]} " + " ".join(f"{value:.6f}" for value in box))
        # An image without boxes keeps an empty label file: YOLO uses it as background
        (output / "labels" / f"{Path(image['file_name']).stem}.txt").write_text("\n".join(lines))
        n_boxes += len(lines)
    return len(coco["images"]), n_boxes


def write_data_yaml(output: Path, names: list[str]) -> Path:
    """data.yaml that Ultralytics reads: dataset root, split folders and class names."""
    lines = [
        f"path: {output.resolve()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(names)),
    ]
    path = output / DATA_FILE
    path.write_text("\n".join(lines) + "\n")
    return path


def convert_coco_to_yolo(source: str | Path, output: str | Path) -> tuple[Path, list[str], dict[str, tuple[int, int]]]:
    """Convert every split of a Roboflow COCO export to YOLO format.
    Returns the data.yaml path, the class names and (images, boxes) per split.
    """
    source, output = Path(source), Path(output)
    splits = [split for split in SPLITS if (source / split / COCO_ANNOTATIONS).is_file()]
    if "train" not in splits or "valid" not in splits:
        raise ValueError(f"'{source.name}' needs train/ and valid/ folders with a {COCO_ANNOTATIONS} file.")
    categories = json.loads((source / "train" / COCO_ANNOTATIONS).read_text())["categories"]
    coco_names = coco_class_names(categories)
    class_ids = {coco_id: index for index, coco_id in enumerate(sorted(coco_names))}  # PC=0, RBC=1, WBC=2
    names = [coco_names[coco_id] for coco_id in sorted(coco_names)]
    counts = {split: convert_split(source / split, output / split, class_ids) for split in splits}
    return write_data_yaml(output, names), names, counts


def yolo_dataset_dir(coco_folder: Path) -> Path:
    """datasets/blood_cell_coco -> datasets/blood_cell_yolo."""
    return DATASETS_DIR / f"{coco_folder.name.removesuffix('_coco')}_yolo"


def training_device() -> str:
    """Apple GPU (MPS) when available: Ultralytics picks CUDA or CPU by itself, never MPS."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


# Fine-tuning: start from the COCO weights of 9.1 (yolo11n.pt) and retrain them for the 3 blood cell classes.
def train_yolo(model: YOLO, data_file: Path, epochs: int, image_size: int, runs_dir: Path = RUNS_DIR) -> dict[str, float]:
    """Train the model and return the validation metrics of the best epoch (precision, recall, mAP)."""
    metrics = model.train(
        data=str(data_file), epochs=epochs, imgsz=image_size, device=training_device(),
        project=str(runs_dir), name="blood_cell", exist_ok=True, verbose=False,
    )
    return {key.removeprefix("metrics/").removesuffix("(B)"): float(value) for key, value in metrics.results_dict.items()}


def save_trained(model: YOLO, path: Path) -> Path:
    """Copy the best weights of the last training run (by validation fitness) to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model.trainer.best, path)
    return path


def load_trained(path: Path = MODEL_FILE) -> YOLO:
    from ultralytics import YOLO

    if not path.is_file():
        raise ValueError("No trained YOLO model found. Run 'Train YOLO (blood cells)' first.")
    return YOLO(str(path))


# ---------- Dialog ----------

class TrainYoloDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc) -> None:
        self.options: tuple[int, int] | None = None
        super().__init__(parent, "Train YOLO")

    def body(self, master: tk.Frame) -> tk.Widget:
        frame = ttk.LabelFrame(master, text="Parameters", padding=8)
        frame.pack(fill="x", padx=8, pady=8)
        self.epochs = tk.StringVar(master, value=str(DEFAULT_EPOCHS))
        self.image_size = tk.StringVar(master, value=str(DEFAULT_IMAGE_SIZE))

        ttk.Label(frame, text="Epochs:").grid(row=0, column=0, sticky="w", pady=2)
        first = ttk.Spinbox(frame, from_=1, to=300, increment=5, textvariable=self.epochs, width=8)
        first.grid(row=0, column=1, padx=(6, 0), pady=2)
        ttk.Label(frame, text="Image size (px, multiple of 32):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(frame, from_=320, to=1280, increment=32, textvariable=self.image_size, width=8).grid(
            row=1, column=1, padx=(6, 0), pady=2
        )
        return first

    def validate(self) -> bool:
        try:
            epochs, image_size = int(self.epochs.get()), int(self.image_size.get())
            valid = epochs >= 1 and image_size >= 32 and image_size % 32 == 0
        except ValueError:
            valid = False
        if not valid:
            messagebox.showwarning(
                self.title(), "Epochs must be >= 1 and the image size a multiple of 32.", parent=self
            )
            return False
        self.options = (epochs, image_size)
        return True


def _ask_options(parent: tk.Misc) -> tuple[int, int] | None:
    return TrainYoloDialog(parent).options


def _ask_coco_folder(parent: tk.Misc) -> Path | None:
    """Yes -> the blood cell dataset in datasets/blood_cell_coco, No -> any Roboflow COCO export."""
    use_blood_cells = messagebox.askyesnocancel(
        "Train YOLO",
        f"Use the blood cell dataset in '{COCO_DIR.relative_to(PROJECT_ROOT)}'? It is converted to YOLO "
        "format the first time.\n"
        "(No = choose your own COCO folder with train/, valid/ and test/)",
        parent=parent,
    )
    if use_blood_cells is None:
        return None
    if use_blood_cells:
        return COCO_DIR
    folder = filedialog.askdirectory(title="COCO folder (train, valid, test)", parent=parent, mustexist=True)
    return Path(folder) if folder else None


# ---------- Tools ----------

class YoloTrainingTool(ForensicsTool):
    tool_id = "yolo_training"
    title = "Train YOLO (blood cells)"
    category = "Object detection"
    description = (
        "Convert the blood cell dataset from COCO to YOLO format, fine-tune YOLO11n on PC, RBC and WBC "
        "and save the best weights to models/yolo_blood_cell.pt."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        coco_folder = _ask_coco_folder(parent)
        if coco_folder is None:
            return None
        options = _ask_options(parent)
        if options is None:
            return None
        epochs, image_size = options

        # Conversion + training block the window for minutes: show a busy cursor meanwhile
        window = parent.winfo_toplevel()
        window.configure(cursor="watch")
        window.update_idletasks()
        try:
            yolo_folder = yolo_dataset_dir(coco_folder)
            data_file = yolo_folder / DATA_FILE
            if not data_file.is_file():
                convert_coco_to_yolo(coco_folder, yolo_folder)
            model = load_yolo(PRETRAINED_FILE)
            metrics = train_yolo(model, data_file, epochs, image_size)
            model_file = save_trained(model, MODEL_FILE)
        finally:
            window.configure(cursor="")

        run_dir = Path(model.trainer.save_dir)
        curves = run_dir / "results.png"  # loss and metric curves per epoch, drawn by Ultralytics
        image = Image.open(curves).convert("RGB") if curves.is_file() else None
        return ToolResult(
            image=image,
            message=f"Trained YOLO for {epochs} epoch(s): mAP50 = {metrics['mAP50']:.3f} on the validation set. "
            f"Saved to {model_file.name}.",
            details={
                "Operation": "Train YOLO",
                "Dataset": str(yolo_folder),
                "Classes": ", ".join(model.names.values()),
                "Pretrained weights": PRETRAINED_FILE.name,
                "Epochs": str(epochs),
                "Image size": str(image_size),
                "Device": training_device(),
                "Precision": f"{metrics['precision']:.3f}",
                "Recall": f"{metrics['recall']:.3f}",
                "mAP50": f"{metrics['mAP50']:.3f}",
                "mAP50-95": f"{metrics['mAP50-95']:.3f}",
                "Run folder": str(run_dir),
                "Model file": str(model_file),
            },
        )
