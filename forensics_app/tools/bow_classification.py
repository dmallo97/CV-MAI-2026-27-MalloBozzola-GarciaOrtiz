"""Classify an image with a bag of visual words: histogram over the codebook words + SVM."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .codebook import CODEBOOK_FILE, PROJECT_ROOT, find_images, load_codebook, load_training_data
from .feature_matching import describe_params, extract_features, to_gray_float
from .visual_word_histogram import assign_words, build_histogram, plot_image_histogram

DATASETS_DIR = PROJECT_ROOT / "datasets"
DIALOG_TITLE = "Classify image"
TEST_SIZE = 0.25
RANDOM_STATE = 42
MIN_IMAGES_PER_CLASS = 2  # the stratified split and the calibration need at least 2 images per class
CALIBRATION_FOLDS = 5
PREVIEW_WIDTH = 900  # px of the histogram shown in the classify dialog
FIGURE_DPI = 100
MIN_FIGURE_WIDTH = 400  # px, so the title fits on small images


# ---------- Pure functions ----------

def export_split(source: str | Path, output: str | Path, test_size: float = TEST_SIZE) -> tuple[int, int]:
    """Copy source/<class>/* to output/train/<class>/ and output/test/<class>/ (stratified split).
    Returns the number of train and test images.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError(f"'{output}' already exists. Delete it to split again.")
    paths = find_images(source)
    labels = [path.parent.name for path in paths]  # class = parent folder, as in the codebook
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("The folder needs one subfolder per class (at least 2 classes).")
    if min(counts.values()) < MIN_IMAGES_PER_CLASS:
        raise ValueError(f"Every class needs at least {MIN_IMAGES_PER_CLASS} images.")

    # stratify keeps the same proportion of every class in train and test
    train, test = train_test_split(paths, test_size=test_size, stratify=labels, random_state=RANDOM_STATE)
    for split, split_paths in (("train", train), ("test", test)):
        for path in split_paths:
            target = output / split / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return len(train), len(test)


def image_histograms(words: np.ndarray, owners: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k-dimensional histogram of every codebook image, built from the words of its own descriptors.
    Images without descriptors have no histogram and are dropped. Returns (histograms (n, k), labels (n,)).
    """
    kept = np.unique(owners)  # images that own at least one descriptor
    histograms = np.array([build_histogram(words[owners == index], k) for index in kept])
    return histograms, labels[kept]


# The SVM learns the class of every training histogram (supervised); K-Means only built the words.
# The calibration fits a sigmoid on the SVM scores (Platt scaling) with cross-validation to get p(class);
# it replaces SVC(probability=True), deprecated in scikit-learn 1.9.
def train_classifier(histograms: np.ndarray, labels: np.ndarray) -> CalibratedClassifierCV:
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("The codebook images must belong to at least 2 classes to train a classifier.")
    if min(counts.values()) < MIN_IMAGES_PER_CLASS:
        raise ValueError(f"Every class needs at least {MIN_IMAGES_PER_CLASS} codebook images.")
    folds = min(CALIBRATION_FOLDS, min(counts.values()))  # every fold must contain every class
    classifier = CalibratedClassifierCV(SVC(kernel="rbf"), cv=folds, ensemble=False)
    return classifier.fit(histograms, labels)


def classify(classifier: CalibratedClassifierCV, histogram: np.ndarray) -> list[tuple[str, float]]:
    """(class, probability) pairs, most probable first.
    The ranking comes from predict_proba (not predict), so the top class is always the most probable one.
    """
    probabilities = classifier.predict_proba(histogram[None])[0]
    order = np.argsort(probabilities)[::-1]
    return [(str(classifier.classes_[index]), float(probabilities[index])) for index in order]


def figure_to_image(figure: Figure) -> Image.Image:
    """Render a matplotlib figure as an RGB PIL image."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())  # (H, W, 4) uint8
    return Image.fromarray(np.ascontiguousarray(rgba[..., :3]))


def render_prediction(image: Image.Image, label: str, probability: float) -> Image.Image:
    """The image with 'p(label) = 0.64' on top, as in the Set 7.3 slide."""
    width, height = image.size
    figure_width = max(width, MIN_FIGURE_WIDTH) / FIGURE_DPI
    figure = Figure(figsize=(figure_width, figure_width * height / width + 0.6), dpi=FIGURE_DPI)
    axis = figure.add_subplot()
    axis.imshow(image.convert("RGB"))
    axis.set_title(f"p({label}) = {probability:.2f}", size=14)
    axis.axis("off")
    figure.tight_layout()
    return figure_to_image(figure)


# ---------- Dialogs ----------

class ClassifyDialog(simpledialog.Dialog):
    """Show the histogram of visual words of the image; the OK button becomes 'Classify'."""

    def __init__(self, parent: tk.Misc, preview: Image.Image) -> None:
        self.preview = preview
        super().__init__(parent, DIALOG_TITLE)

    def body(self, master: tk.Frame) -> tk.Widget | None:
        self.photo = ImageTk.PhotoImage(self.preview, master=master)  # keep a reference or Tk drops it
        ttk.Label(master, image=self.photo).pack(padx=8, pady=8)
        return None

    def buttonbox(self) -> None:
        box = ttk.Frame(self)
        ttk.Button(box, text="Classify", width=10, command=self.ok, default="active").pack(side="left", padx=5, pady=5)
        ttk.Button(box, text="Cancel", width=10, command=self.cancel).pack(side="left", padx=5, pady=5)
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack()

    def apply(self) -> None:
        self.result = True


def _ask_classify(parent: tk.Misc, preview: Image.Image) -> bool:
    preview.thumbnail((PREVIEW_WIDTH, PREVIEW_WIDTH))
    return ClassifyDialog(parent, preview).result is not None


def _ask_source_folder(parent: tk.Misc) -> Path | None:
    folder = filedialog.askdirectory(title="Dataset folder (one subfolder per class)", parent=parent, mustexist=True)
    return Path(folder) if folder else None


# ---------- Tools ----------

class SplitDatasetTool(ForensicsTool):
    tool_id = "split_dataset"
    title = "Split dataset (train/test)"
    category = "Bag of Words"
    description = (
        "Copy a folder with one subfolder per class to datasets/<name>_split/train and test "
        f"(stratified, {TEST_SIZE:.0%} of every class for test)."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = _ask_source_folder(parent)
        if source is None:
            return None
        output = DATASETS_DIR / f"{source.name}_split"
        n_train, n_test = export_split(source, output)
        classes = sorted(path.name for path in (output / "train").iterdir() if path.is_dir())
        return ToolResult(
            message=f"Split {n_train + n_test} images into {n_train} train and {n_test} test images.",
            details={
                "Operation": "Split dataset (train/test)",
                "Source folder": str(source),
                "Output folder": str(output),
                "Classes": ", ".join(classes),
                "Train images": str(n_train),
                "Test images": str(n_test),
                "Test size": f"{TEST_SIZE:.0%}",
            },
        )


class ClassifyImageTool(ForensicsTool):
    tool_id = "classify_image"
    title = "Classify image (BoW + SVM)"
    category = "Bag of Words"
    description = (
        "Show the histogram of visual words of the current image and, after clicking 'Classify', "
        "train an RBF SVM on the codebook images and show the top probability."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        centers, words, params = load_codebook(CODEBOOK_FILE)
        _, owners, labels, paths = load_training_data(CODEBOOK_FILE)

        assert document.current is not None
        gray = to_gray_float(document.current)
        keypoints, descriptors = extract_features(gray, params)  # same descriptor as the codebook
        if len(descriptors) == 0:
            raise ValueError("No keypoints found in the image.")
        image_words = assign_words(descriptors.astype(float), centers)
        histogram = build_histogram(image_words, len(centers))

        name = document.path.name if document.path else "Current image"
        figure = plot_image_histogram(gray, keypoints, image_words, histogram, name)  # same plot as 7.2
        preview = figure_to_image(figure)
        plt.close(figure)
        if not _ask_classify(parent, preview):
            return None

        train_histograms, train_labels = image_histograms(words, owners, labels, len(centers))
        ranking = classify(train_classifier(train_histograms, train_labels), histogram)
        prediction, probability = ranking[0]

        # The true class is only known when the image sits in a class folder of the codebook (e.g. test/<class>/)
        true_label = document.path.parent.name if document.path else None
        if true_label not in labels:
            true_label = None
        if true_label is None:
            verdict, correct = "", "-"
        elif prediction == true_label:
            verdict, correct = " (correct)", "yes"
        else:
            verdict, correct = f" (wrong, it is {true_label})", "no"

        details = {
            "Operation": "Bag of words classification (SVM)",
            "Prediction": prediction,
            "Top probability": f"{probability:.2f}",
            **{f"p({label})": f"{p:.2f}" for label, p in ranking},
            "True class": true_label or "Unknown",
            "Correct": correct,
            **describe_params(params),
            "Descriptors": str(len(image_words)),
            "Visual words (K)": str(len(centers)),
            "Training images": ", ".join(f"{label} {count}" for label, count in sorted(Counter(train_labels).items())),
            "Classifier": "SVC (RBF kernel) + Platt calibration",
        }
        training_paths = {Path(path).resolve() for path in paths}
        if document.path is not None and document.path.resolve() in training_paths:
            details["Note"] = "This image was used to build the codebook and train the classifier."
        return ToolResult(
            image=render_prediction(document.current, prediction, probability),
            message=f"p({prediction}) = {probability:.2f}{verdict}.",
            details=details,
        )
