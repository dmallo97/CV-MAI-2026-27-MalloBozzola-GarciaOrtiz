"""Build a codebook of visual words with K-Means over ORB / SIFT descriptors."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .feature_matching import IMAGE_FILE_TYPES, ask_feature_params, describe_params, extract_features, load_gray

DIALOG_TITLE = "Construct Codebook"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
WORDS_PER_CLASS = 10  # K = 10 * num_classes
PATCH_SIZE = 32  # side (px) of the patch shown for each visual word
N_COLS = 5
SHOWN_IMAGES = 6  # images displayed with their keypoints


# ---------- Pure functions ----------

def find_images(folder: str | Path) -> list[Path]:
    """All the images inside a folder (subfolders included)."""
    return sorted(path for path in Path(folder).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def count_classes(paths: list[Path]) -> int:
    """Each image belongs to the class given by its parent folder name."""
    return len({path.parent.name for path in paths})


def build_codebook(feature_descriptors: list[np.ndarray], n_clusters: int) -> tuple[MiniBatchKMeans, np.ndarray]:
    """Stack the descriptors of all the images and cluster them: each centre is a visual word."""
    descriptors = feature_descriptors[0] # descriptors of image 0 (N, D), N -> number of keypoints, D -> length of the descriptor
    for descriptor in feature_descriptors[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    descriptors = descriptors.astype(float)  # ORB descriptors are bool
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(descriptors)
    return kmeans, descriptors


def visual_word_patches(kmeans, descriptors, owners, keypoints, grays, size=PATCH_SIZE) -> list[np.ndarray]:
    """For each centre, crop the image patch around the keypoint whose descriptor is closest to it."""
    closest = kmeans.transform(descriptors).argmin(axis=0)  # (K,) one descriptor index per centre
    half = size // 2
    patches = []
    for index in closest:
        gray = np.pad(grays[owners[index]], half, mode="edge")  # padding so border patches fit
        row, col = keypoints[index].astype(int)
        patches.append(gray[row:row + size, col:col + size])
    return patches


def plot_keypoints(grays, keypoints_per_image, names) -> plt.Figure:
    """Grid with the first images of the collection and their keypoints in red."""
    grays = grays[:SHOWN_IMAGES]
    n_rows = int(np.ceil(len(grays) / N_COLS))
    figure, axes = plt.subplots(n_rows, N_COLS, figsize=(2.6 * N_COLS, 2.4 * n_rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, gray, keypoints, name in zip(axes.ravel(), grays, keypoints_per_image, names):
        axis.imshow(gray, cmap="gray")
        axis.scatter(keypoints[:, 1], keypoints[:, 0], s=8, c="red")
        axis.set_title(name, size=8)
    figure.tight_layout()
    return figure


def plot_patches(patches) -> plt.Figure:
    """Grid with one patch per visual word (vw 0, vw 1, ...)."""
    n_rows = int(np.ceil(len(patches) / N_COLS))
    figure, axes = plt.subplots(n_rows, N_COLS, figsize=(1.6 * N_COLS, 1.6 * n_rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for index, (axis, patch) in enumerate(zip(axes.ravel(), patches)):
        axis.imshow(patch, cmap="gray")
        axis.set_title(f"vw {index}", size=8)
    figure.tight_layout()
    return figure


# ---------- Dialog ----------

class ImageCollectionDialog(simpledialog.Dialog):
    """List of images that the user fills with 'Add folder' and 'Add file'."""

    def __init__(self, parent: tk.Misc) -> None:
        self.paths: list[Path] = []
        super().__init__(parent, DIALOG_TITLE)

    def body(self, master: tk.Frame) -> tk.Widget:
        buttons = ttk.Frame(master)
        buttons.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(buttons, text="Add folder", command=self.add_folder).pack(side="left")
        ttk.Button(buttons, text="Add file", command=self.add_files).pack(side="left", padx=6)
        self.count = ttk.Label(master, text="0 images")
        self.count.pack(anchor="w", padx=8)
        self.listbox = tk.Listbox(master, width=70, height=12)
        self.listbox.pack(fill="both", expand=True, padx=8, pady=8)
        return self.listbox

    def add_paths(self, paths: list[Path]) -> None:
        for path in paths:
            if path not in self.paths:
                self.paths.append(path)
                self.listbox.insert("end", str(path))
        self.count.configure(text=f"{len(self.paths)} images")

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Image folder", parent=self, mustexist=True)
        if folder:
            self.add_paths(find_images(folder))

    def add_files(self) -> None:
        filenames = filedialog.askopenfilenames(title="Add image", filetypes=IMAGE_FILE_TYPES, parent=self)
        self.add_paths([Path(name) for name in filenames])

    def validate(self) -> bool:
        if not self.paths:
            messagebox.showwarning(DIALOG_TITLE, "Add at least one image.", parent=self)
            return False
        return True

    def apply(self) -> None:
        self.result = self.paths


# ---------- Tool ----------

class CodebookTool(ForensicsTool):
    tool_id = "construct_codebook"
    title = "Construct Codebook"
    category = "Bag of Words"
    description = (
        "Extract ORB or SIFT descriptors from a collection of images, cluster them with "
        "K-Means (K = 10 * classes) and show the keypoints and the visual words."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        paths = ImageCollectionDialog(parent).result
        if not paths:
            return None
        params = ask_feature_params(parent)
        if params is None:
            return None
        n_classes = count_classes(paths)
        n_clusters = simpledialog.askinteger(
            DIALOG_TITLE, f"Number of visual words K ({n_classes} classes found):", parent=parent,
            initialvalue=WORDS_PER_CLASS * n_classes, minvalue=1,
        )
        if n_clusters is None:
            return None

        # Descriptors of every image; owners[i] = image of the i-th descriptor
        grays, keypoints_per_image, feature_descriptors, owners = [], [], [], []
        for index, path in enumerate(paths):
            gray = load_gray(path)
            keypoints, descriptors = extract_features(gray, params)
            grays.append(gray)
            keypoints_per_image.append(keypoints)
            feature_descriptors.append(descriptors)
            owners.extend([index] * len(keypoints))
        keypoints = np.vstack(keypoints_per_image)

        if len(keypoints) < n_clusters:
            raise ValueError(f"Only {len(keypoints)} descriptors found; K must not be larger.")
        kmeans, descriptors = build_codebook(feature_descriptors, n_clusters)

        plot_keypoints(grays, keypoints_per_image, [path.parent.name for path in paths])
        plot_patches(visual_word_patches(kmeans, descriptors, owners, keypoints, grays))
        plt.show(block=False)

        return ToolResult(
            message=f"Built a codebook of {n_clusters} visual words from {len(descriptors)} descriptors "
            f"of {len(paths)} images.",
            details={
                "Operation": "Construct Codebook",
                **describe_params(params),
                "Images": str(len(paths)),
                "Classes": str(n_classes),
                "Descriptors shape": str(descriptors.shape),
                "Visual words (K)": str(n_clusters),
                "Centres shape": str(kmeans.cluster_centers_.shape),
            },
        )
