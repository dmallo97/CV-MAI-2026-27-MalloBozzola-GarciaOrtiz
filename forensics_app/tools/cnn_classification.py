"""Set 8: use a CNN trained in the notebook. A model is a folder with the state_dict (.pt), the passport
(<name>_passport.json) and, optionally, the feature vectors of the test set (test_features.npz).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .bow_classification import render_prediction
from .cnn_networks import ARCHITECTURES, build_network
from .codebook import PROJECT_ROOT

CATEGORY = "Classify using CNN"
PASSPORT_SUFFIX = "_passport.json"
FEATURES_FILE = "test_features.npz"
CIFAR_ROOT = PROJECT_ROOT / "datasets" / "cifar10"
REQUIRED_PASSPORT_KEYS = ("architecture", "num_classes", "class_names", "preprocessing", "weights_file")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
TOP_K = 3
TSNE_PERPLEXITY = 40
UMAP_NEIGHBOURS = 15
UMAP_MIN_DIST = 0.1
RANDOM_STATE = 0

# Passport keys shown in the 'Model' section; 'history' is plotted and 'description' goes in the header
MODEL_KEYS = ("model_name", "architecture", "num_classes", "weights_file", "weights_format", "torch_version", "saved_at")
SECTION_KEYS = (("Dataset", "dataset"), ("Preprocessing", "preprocessing"), ("Training", "training"),
                ("Test metrics", "test_metrics"))
NOT_LISTED_KEYS = {"history", "description", "class_names", *MODEL_KEYS, *(key for _, key in SECTION_KEYS)}


@dataclass(frozen=True)
class LoadedCnn:
    folder: Path
    passport: dict[str, Any]
    network: nn.Module

    @property
    def name(self) -> str:
        return str(self.passport.get("model_name", self.folder.name))

    @property
    def class_names(self) -> list[str]:
        return list(self.passport["class_names"])

    @property
    def features_file(self) -> Path:
        return self.folder / FEATURES_FILE


class CnnSession:
    """The model loaded by 'Load CNN model', shared by the other tools of the section."""

    def __init__(self) -> None:
        self.loaded: LoadedCnn | None = None


# ---------- Pure functions ----------

def find_passport(folder: str | Path) -> Path:
    folder = Path(folder)
    passports = sorted(folder.glob(f"*{PASSPORT_SUFFIX}"))
    if len(passports) != 1:
        raise ValueError(
            f"The model folder must contain exactly one '*{PASSPORT_SUFFIX}' file (found {len(passports)})."
        )
    return passports[0]


def load_passport(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        passport = json.load(file)
    missing = [key for key in REQUIRED_PASSPORT_KEYS if key not in passport]
    if missing:
        raise ValueError(f"The passport has no {', '.join(missing)}.")
    if passport["architecture"] not in ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{passport['architecture']}'. Available models: {', '.join(ARCHITECTURES)}."
        )
    return passport


def load_cnn(folder: str | Path, device: str = DEVICE) -> LoadedCnn:
    """Rebuild the architecture named in the passport and load its state_dict."""
    folder = Path(folder)
    passport = load_passport(find_passport(folder))
    weights = folder / passport["weights_file"]
    if not weights.is_file():
        raise ValueError(f"Weights file '{passport['weights_file']}' not found in the model folder.")
    network = build_network(passport["architecture"], int(passport["num_classes"]))
    # weights_only: only tensors are unpickled, never arbitrary objects
    network.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return LoadedCnn(folder, passport, network.to(device).eval())


def build_transform(preprocessing: dict[str, Any]) -> transforms.Compose:
    """Same transform as in training, rebuilt from the passport (resize, [0, 1] scaling, normalization)."""
    steps = [transforms.Resize(tuple(preprocessing["resize"])), transforms.ToTensor()]
    if preprocessing.get("normalize"):
        steps.append(transforms.Normalize(preprocessing["normalize"]["mean"], preprocessing["normalize"]["std"]))
    return transforms.Compose(steps)


def classify(loaded: LoadedCnn, image: Image.Image) -> list[tuple[str, float]]:
    """(class, probability) pairs of every class, most probable first."""
    device = next(loaded.network.parameters()).device
    x = build_transform(loaded.passport["preprocessing"])(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(loaded.network(x), dim=1)[0].cpu().numpy()
    order = np.argsort(probabilities)[::-1]
    return [(loaded.class_names[index], float(probabilities[index])) for index in order]


def extract_test_features(loaded: LoadedCnn, root: Path = CIFAR_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """forward_features of every CIFAR-10 test image (downloaded to root the first time) -> (features, labels)."""
    dataset_name = loaded.passport.get("dataset", {}).get("name")
    if dataset_name != "CIFAR-10":
        raise ValueError(f"Feature extraction is only available for CIFAR-10 models (this one: {dataset_name}).")
    test_set = datasets.CIFAR10(root=str(root), train=False, download=True,
                                transform=build_transform(loaded.passport["preprocessing"]))
    device = next(loaded.network.parameters()).device
    features, labels = [], []
    with torch.no_grad():
        for images, targets in DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False):
            features.append(loaded.network.forward_features(images.to(device)).cpu())
            labels.append(targets)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def save_features(path: Path, features: np.ndarray, labels: np.ndarray, class_names: list[str]) -> Path:
    np.savez_compressed(path, features=features, labels=labels, class_names=np.array(class_names))
    return path


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path) as data:
        return data["features"], data["labels"], data["class_names"].tolist()


def project_features(features: np.ndarray) -> dict[str, np.ndarray]:
    """2D t-SNE and UMAP embeddings (N, 2) of the standardized feature vectors (N, D)."""
    import umap  # slow import (numba), only needed here
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    standardized = StandardScaler().fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init="pca", random_state=RANDOM_STATE)
    umap_model = umap.UMAP(n_components=2, n_neighbors=UMAP_NEIGHBOURS, min_dist=UMAP_MIN_DIST,
                           random_state=RANDOM_STATE)
    return {"t-SNE": tsne.fit_transform(standardized), "UMAP": umap_model.fit_transform(standardized)}


def plot_projections(projections: dict[str, np.ndarray], labels: np.ndarray, class_names: list[str],
                     title: str) -> plt.Figure:
    figure, axes = plt.subplots(1, len(projections), figsize=(7 * len(projections), 6.5), squeeze=False)
    for axis, (method, embedding) in zip(axes[0], projections.items()):
        for index, name in enumerate(class_names):
            mask = labels == index
            axis.scatter(embedding[mask, 0], embedding[mask, 1], s=3, label=name)
        axis.set_title(method)
        axis.set_xlabel(f"{method} 1")
        axis.set_ylabel(f"{method} 2")
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0][-1].legend(markerscale=4, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_history(history: dict[str, Any]) -> Figure:
    """Loss/accuracy curves; history is a fit() dict or {phase: fit() dict} (transfer learning)."""
    phases = {"": history} if "loss" in history else history
    figure = Figure(figsize=(6, 6.5), dpi=100, layout="constrained")  # no overlap between titles and labels
    axes = figure.subplots(2, 1)
    offset = 0
    for phase, values in phases.items():
        epochs = np.arange(offset + 1, offset + len(values["loss"]) + 1)
        for axis, metric in zip(axes, ("loss", "accuracy")):
            axis.plot(epochs, values[metric], label=f"{phase} train".strip())
            axis.plot(epochs, values[f"val_{metric}"], "--", label=f"{phase} val".strip())
        offset += len(values["loss"])
    for axis, title in zip(axes, ("Cross entropy loss", "Classification accuracy")):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.legend(fontsize=8)
    return figure


def format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(item) for item in value)
    return str(value)


def _label(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _flatten(values: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    """Nested dicts become 'Parent / child' rows."""
    rows = []
    for key, value in values.items():
        label = f"{prefix}{_label(key)}"
        if isinstance(value, dict) and value:
            rows.extend(_flatten(value, f"{label} / "))
        else:
            rows.append((label, format_value(value)))
    return rows


def passport_sections(passport: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    """(section title, [(property, value)]) rows shown in the passport window."""
    sections = [("Model", [(_label(key), format_value(passport[key])) for key in MODEL_KEYS if key in passport])]
    for title, key in SECTION_KEYS:
        if isinstance(passport.get(key), dict):
            sections.append((title, _flatten(passport[key])))
    sections.append(("Classes", [(str(index), name) for index, name in enumerate(passport["class_names"])]))
    others = {key: value for key, value in passport.items() if key not in NOT_LISTED_KEYS}
    if others:
        sections.append(("Other", _flatten(others)))
    return sections


def metrics_summary(passport: dict[str, Any]) -> str:
    metrics = passport.get("test_metrics") or {}
    parts = []
    if "accuracy" in metrics:
        parts.append(f"Test accuracy: {metrics['accuracy']:.2%}")
    if "loss" in metrics:
        parts.append(f"Test loss: {metrics['loss']:.4f}")
    return "    ".join(parts) or "No test metrics in the passport"


# ---------- Dialogs ----------

class PassportWindow(tk.Toplevel):
    """Header (name, description, test metrics), passport sections and the training history plot."""

    def __init__(self, parent: tk.Misc, passport: dict[str, Any]) -> None:
        super().__init__(parent)
        name = passport.get("model_name", "CNN model")
        self.title(f"Model passport - {name}")
        self.geometry("1150x720")

        header = ttk.Frame(self, padding=(14, 12, 14, 6))
        header.pack(fill="x")
        ttk.Label(header, text=name, font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        if passport.get("description"):
            ttk.Label(header, text=passport["description"], wraplength=1100, foreground="#4a5361").pack(
                anchor="w", pady=(2, 6)
            )
        ttk.Label(header, text=metrics_summary(passport), font=("TkDefaultFont", 12, "bold"),
                  foreground="#1f6f43").pack(anchor="w")
        ttk.Separator(self).pack(fill="x", padx=14, pady=6)

        body = ttk.Frame(self, padding=(14, 0, 14, 8))
        body.pack(fill="both", expand=True)

        table = ttk.Frame(body)
        table.pack(side="left", fill="both", expand=True)
        tree = ttk.Treeview(table, columns=("value",), show="tree headings")
        tree.heading("#0", text="Property")
        tree.heading("value", text="Value")
        tree.column("#0", width=300, stretch=False)
        tree.column("value", width=330, stretch=True)
        tree.tag_configure("section", font=("TkDefaultFont", 10, "bold"), background="#eef1f5")
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        for title, rows in passport_sections(passport):
            section = tree.insert("", "end", text=title, open=True, tags=("section",))
            for label, value in rows:
                tree.insert(section, "end", text=label, values=(value,))

        if passport.get("history"):
            plot = ttk.LabelFrame(body, text="Training history", padding=6)
            plot.pack(side="right", fill="both", padx=(12, 0))
            canvas = FigureCanvasTkAgg(plot_history(passport["history"]), master=plot)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))


def _ask_model_folder(parent: tk.Misc) -> Path | None:
    folder = filedialog.askdirectory(
        title="Model folder (<name>.pt + <name>_passport.json)", parent=parent, mustexist=True
    )
    return Path(folder) if folder else None


def _ask_extract(parent: tk.Misc) -> bool:
    return messagebox.askyesno(
        "Feature projection",
        f"The model folder has no '{FEATURES_FILE}'.\n\n"
        "Extract the feature vectors of the CIFAR-10 test set (10000 images) now and save them in the "
        f"model folder? CIFAR-10 is downloaded to '{CIFAR_ROOT.relative_to(PROJECT_ROOT)}' the first time. "
        "This can take several minutes on CPU.",
        parent=parent,
    )


def _show_figure(figure: plt.Figure) -> None:
    plt.show(block=False)


@contextmanager
def _busy(parent: tk.Misc | None) -> Iterator[None]:
    """Watch cursor while a long computation blocks the window."""
    window = parent.winfo_toplevel() if parent is not None else None
    if window is not None:
        window.configure(cursor="watch")
        window.update_idletasks()
    try:
        yield
    finally:
        if window is not None:
            window.configure(cursor="")


# ---------- Tools ----------

class _CnnTool(ForensicsTool):
    category = CATEGORY
    unavailable_message = "Load a CNN model first ('Load CNN model')."

    def __init__(self, session: CnnSession) -> None:
        self.session = session

    def is_available(self, document: ImageDocument) -> bool:
        return self.session.loaded is not None

    @property
    def loaded(self) -> LoadedCnn:
        assert self.session.loaded is not None
        return self.session.loaded


class LoadCnnModelTool(_CnnTool):
    tool_id = "cnn_load_model"
    title = "Load CNN model"
    description = (
        "Load a model folder saved by the Set 8 notebook (state_dict .pt + passport .json). "
        f"Available models: {', '.join(ARCHITECTURES)}."
    )
    requires_image = False

    def is_available(self, document: ImageDocument) -> bool:
        return True

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        folder = _ask_model_folder(parent)
        if folder is None:
            return None
        with _busy(parent):
            loaded = load_cnn(folder)
        self.session.loaded = loaded
        passport = loaded.passport
        accuracy = (passport.get("test_metrics") or {}).get("accuracy")
        return ToolResult(
            message=f"Loaded {loaded.name} ({passport['architecture']}) on {DEVICE}.",
            details={
                "Operation": "Load CNN model",
                "Model": loaded.name,
                "Architecture": passport["architecture"],
                "Classes": str(passport["num_classes"]),
                "Input size": " x ".join(str(side) for side in passport["preprocessing"]["resize"]),
                "Test accuracy": f"{accuracy:.2%}" if accuracy is not None else "-",
                "Test features": "yes" if loaded.features_file.is_file() else "no",
                "Device": DEVICE,
                "Folder": str(folder),
            },
        )


class CnnPassportTool(_CnnTool):
    tool_id = "cnn_passport"
    title = "Display model's passport"
    description = "Show the passport of the loaded CNN: dataset, classes, preprocessing, training and history."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        PassportWindow(parent, self.loaded.passport)
        return ToolResult(
            message=f"Showing the passport of {self.loaded.name}.",
            details={"Operation": "Display model's passport", "Model": self.loaded.name,
                     "Test metrics": metrics_summary(self.loaded.passport)},
        )


class CnnClassifyTool(_CnnTool):
    tool_id = "cnn_classify"
    title = "Classify new image"
    description = "Classify the current image with the loaded CNN and show it with the predicted class."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        ranking = classify(self.loaded, document.current)
        prediction, probability = ranking[0]

        # The true class is only known when the image sits in a folder named as a class (e.g. test/cat/)
        true_label = document.path.parent.name if document.path else None
        if true_label not in self.loaded.class_names:
            true_label = None
        if true_label is None:
            verdict, correct = "", "-"
        elif prediction == true_label:
            verdict, correct = " (correct)", "yes"
        else:
            verdict, correct = f" (wrong, it is {true_label})", "no"

        width, height = document.current.size
        return ToolResult(
            image=render_prediction(document.current, prediction, probability),
            message=f"{self.loaded.name}: p({prediction}) = {probability:.2f}{verdict}.",
            details={
                "Operation": "Classify using CNN",
                "Model": self.loaded.name,
                "Prediction": prediction,
                **{f"Top {rank}": f"{label} ({p:.2f})" for rank, (label, p) in enumerate(ranking[:TOP_K], start=1)},
                "True class": true_label or "Unknown",
                "Correct": correct,
                "Input size (W x H)": f"{width} x {height}",
                "Model input": " x ".join(str(side) for side in self.loaded.passport["preprocessing"]["resize"]),
            },
        )


class CnnFeatureProjectionTool(_CnnTool):
    tool_id = "cnn_feature_projection"
    title = "Feature projection (t-SNE / UMAP)"
    description = (
        f"Load the test set feature vectors of the model ({FEATURES_FILE}, extracted from CIFAR-10 if missing) "
        "and project them to 2D with t-SNE and UMAP."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        loaded = self.loaded
        extracted = False
        if not loaded.features_file.is_file():
            if not _ask_extract(parent):
                return None
            with _busy(parent):
                features, labels = extract_test_features(loaded)
                save_features(loaded.features_file, features, labels, loaded.class_names)
            extracted = True

        features, labels, class_names = load_features(loaded.features_file)
        with _busy(parent):
            projections = project_features(features)
        _show_figure(plot_projections(projections, labels, class_names, f"Test set features - {loaded.name}"))
        return ToolResult(
            message=f"Projected {len(features)} feature vectors of {loaded.name} with t-SNE and UMAP.",
            details={
                "Operation": "Feature projection (t-SNE / UMAP)",
                "Model": loaded.name,
                "Feature vectors": f"{features.shape[0]} x {features.shape[1]}",
                "Features file": str(loaded.features_file),
                "Extracted now": "yes" if extracted else "no",
                "t-SNE perplexity": str(TSNE_PERPLEXITY),
                "UMAP neighbours": str(UMAP_NEIGHBOURS),
                "UMAP min dist": str(UMAP_MIN_DIST),
            },
        )
