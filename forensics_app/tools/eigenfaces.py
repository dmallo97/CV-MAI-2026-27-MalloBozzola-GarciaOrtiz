from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "lfw"
MODEL_FILE = PROJECT_ROOT / "models" / "eigenfaces.npz"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".pgm"})

DEFAULT_MIN_FACES_PER_PERSON = 70
DEFAULT_RESIZE = 0.4
TEST_SIZE = 0.25
RANDOM_STATE = 42

DEFAULT_COMPONENTS = 100
DEFAULT_SHOWN = 12
GALLERY_COLUMNS = 4

DEFAULT_NEIGHBOURS = 3
MAX_NEIGHBOURS = 10

@dataclass(frozen=True)
class FaceSet:
    images: np.ndarray  # (N, H, W) float in [0, 1]; N = number of images, H, W = height and width of each image; float values (for PCA)
    # images[i] would be a 2D matrix of size (H, W) for the i-th image
    labels: np.ndarray  # (N,) person name of every image 
    paths: list[Path]

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.images.shape[1], self.images.shape[2]
    
    
@dataclass(frozen=True)
class EigenfaceModel:
    mean: np.ndarray  # (D,) mean face, D = H * W (1D flattened image)
    eigenfaces: np.ndarray  # (K, D) one eigenface per row (U_K transposed); K directions of PCA
    coefficients: np.ndarray  # (N, K) wi of every training image (N training samples)
    labels: np.ndarray  # (N,) person name of every training image
    paths: np.ndarray  # (N,) file of every training image
    image_shape: tuple[int, int]  # (H, W)
    explained_variance_ratio: np.ndarray  # (K,)
    # x ~ mean + wi * ui; ui -> eigenfaces; wi -> coefficients

    @property
    def n_components(self) -> int:
        return len(self.eigenfaces)

    def save(self, path: Path = MODEL_FILE) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            eigenfaces=self.eigenfaces,
            coefficients=self.coefficients,
            labels=self.labels,
            paths=self.paths,
            image_shape=np.array(self.image_shape),
            explained_variance_ratio=self.explained_variance_ratio,
        )
        return path

    @classmethod # called over the class, not the instance (EigenfaceModel.load())
    def load(cls, path: Path = MODEL_FILE) -> EigenfaceModel:
        with np.load(path) as data:
            return cls(
                mean=data["mean"],
                eigenfaces=data["eigenfaces"],
                coefficients=data["coefficients"],
                labels=data["labels"],
                paths=data["paths"],
                image_shape=tuple(int(side) for side in data["image_shape"]),
                explained_variance_ratio=data["explained_variance_ratio"],
            )


@dataclass(frozen=True)
class Match:
    index: int  # row of the training face in the model
    label: str  # person name of the training face
    distance: float  # Euclidean distance between PCA coefficients (eigenface space, not pixels)


# ---------- Pure functions ----------

def to_uint8(face: np.ndarray) -> np.ndarray:
    """LFW faces may come in [0, 1] or [0, 255] depending on the sklearn version."""
    scale = 255.0 if face.max() <= 1.0 else 1.0
    return np.clip(face * scale, 0, 255).astype(np.uint8)

def face_label(path: Path) -> str:
    """Person name of a face image, given by its subfolder (George_W_Bush -> George W Bush)."""
    return path.parent.name.replace("_", " ")

def save_faces(images: np.ndarray, names: list[str], folder: Path) -> None:
    """Write folder/<Person_Name>/<index>.png: the subfolder keeps the image and the name pair."""
    for index, (face, name) in enumerate(zip(images, names)):
        person_dir = folder / name.replace(" ", "_") # from project folder, create a directory for the images of each person
        person_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_uint8(face), mode="L").save(person_dir / f"{index:04d}.png")


# We are using lfw (labeled faces wild) dataset from scikit learn  
def export_lfw(
    output_dir: Path = DATASET_DIR,
    min_faces_per_person: int = DEFAULT_MIN_FACES_PER_PERSON,
    resize: float = DEFAULT_RESIZE,
) -> tuple[int, int]:
    """Download LFW (cached by sklearn) and save it as train/ and test/ folders.
    Returns the length of the train and test images.
    """
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
    names = [str(lfw.target_names[target]) for target in lfw.target]
    train_images, test_images, train_names, test_names = train_test_split(
        lfw.images, names, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    save_faces(train_images, train_names, output_dir / "train")  #
    save_faces(test_images, test_names, output_dir / "test")
    return len(train_images), len(test_images)


def load_face_folder(folder: str | Path) -> FaceSet:
    """Read folder/<Person_Name>/*.png as grayscale faces labelled with the subfolder name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"'{folder}' is not a folder.")
    images, labels, paths = [], [], []
    for person_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
        for path in sorted(person_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("L"), dtype=np.float64) / 255.0) # turn image as grayscale floats and scale them
            labels.append(face_label(path))
            paths.append(path)
    if not images:
        raise ValueError(f"No face images found in the subfolders of '{folder.name}'.")
    if len({image.shape for image in images}) > 1:
        raise ValueError("All face images must have the same size.")
    return FaceSet(np.stack(images), np.array(labels), paths)


# Projection: the faces must be centered (row mean substraction). No retraining, model just computes the pca coefficients to represent faces into eigenface space
def project(faces: np.ndarray, mean: np.ndarray, eigenfaces: np.ndarray) -> np.ndarray:
    """PCA coefficients w = U_K^T (x - mu) for every row of faces (N, D) -> (N, K). (D=H*W)"""
    return (faces - mean) @ eigenfaces.T

# Fitting the faces into the pca model to build the eigenface directions that form the eigenface space. 
def build_eigenfaces(faces: FaceSet, n_components: int = DEFAULT_COMPONENTS) -> EigenfaceModel:
    data = faces.images.reshape(len(faces.images), -1)  # (N, H*W): one face per row
    max_components = min(data.shape)
    if not 1 <= n_components <= max_components:
        raise ValueError(f"The number of eigenfaces must be in [1, {max_components}].")
    
    # Define and fit the PCA model to the faces.
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_STATE).fit(data)
    # pca.mean_ (D,): vector of mean value of each pixel over all the faces
    # pca.components_ (K, D): eigenfaces (one per row) (basis of K eigenvectors over the D dimensional space)
    # pca.explained_variance_ratio_ (K,): fraction of variance of each direction
    
    return EigenfaceModel(
        mean=pca.mean_,
        eigenfaces=pca.components_,
        coefficients=project(data, pca.mean_, pca.components_),
        labels=faces.labels,
        paths=np.array([str(path) for path in faces.paths]),
        image_shape=faces.image_shape,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

# Plot the eigenfaces -> representations in the original space of the principal directions
def plot_eigenfaces(model: EigenfaceModel, k: int = DEFAULT_SHOWN, n_cols: int = GALLERY_COLUMNS) -> plt.Figure:
    """Gallery with the mean face followed by the first k eigenfaces."""
    k = min(k, model.n_components)
    n_rows = int(np.ceil((k + 1) / n_cols)) # keep the number of rows close to the number of columns
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 2.4 * n_rows))
    faces = [model.mean, *model.eigenfaces[:k]] # list of vectors (* unpacks the set of eigenfaces into individual vectors)
    titles = ["Mean face", *(f"Eigenface {i}" for i in range(k))]
    for axis in np.ravel(axes):
        axis.axis("off")
    for axis, face, title in zip(np.ravel(axes), faces, titles):
        axis.imshow(face.reshape(model.image_shape), cmap="gray")
        axis.set_title(title, size=10)
    figure.tight_layout()
    return figure


# Recognition: a new face is prepared exactly like the training ones (grayscale, same size, [0, 1]),
# projected onto the eigenfaces and compared with the coefficients of every training face.
def prepare_face(image: Image.Image, image_shape: tuple[int, int]) -> np.ndarray:
    """Flattened (D,) grayscale face in [0, 1], resized to the (H, W) of the model if needed."""
    height, width = image_shape
    gray = image.convert("L")
    if gray.size != (width, height):  # PIL sizes are (W, H)
        gray = gray.resize((width, height))
    return np.asarray(gray, dtype=np.float64).ravel() / 255.0


def find_closest(model: EigenfaceModel, coefficients: np.ndarray, k: int = DEFAULT_NEIGHBOURS) -> list[Match]:
    """The k training faces whose coefficients are closest (Euclidean) to the given (K,) ones."""
    distances = np.linalg.norm(model.coefficients - coefficients, axis=1)  # (N,) one per training face
    closest = np.argsort(distances)[:k]  # the slice already limits k to N
    return [Match(int(index), str(model.labels[index]), float(distances[index])) for index in closest]


def train_face(model: EigenfaceModel, index: int) -> np.ndarray:
    """Training face (H, W) in [0, 1]: read from its file, or rebuilt as mean + w U_K if the file moved."""
    path = Path(model.paths[index])
    if path.is_file():
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    return (model.mean + model.coefficients[index] @ model.eigenfaces).reshape(model.image_shape)


def plot_matches(face: np.ndarray, true_label: str | None, matches: list[Match], model: EigenfaceModel) -> plt.Figure:
    """The test face followed by its closest training faces, titled with their name and distance."""
    n_cols = len(matches) + 1
    figure, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.8), squeeze=False)
    faces = [face.reshape(model.image_shape), *(train_face(model, match.index) for match in matches)]
    titles = [f"Test\n{true_label or 'Unknown'}", *(f"{match.label}\nd = {match.distance:.2f}" for match in matches)]
    for axis, image, title in zip(axes.ravel(), faces, titles):
        axis.imshow(image, cmap="gray")
        axis.set_title(title, size=9)
        axis.axis("off")
    figure.tight_layout()
    return figure


# ---------- Dialog ----------

class EigenfaceDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc) -> None:
        self.options: tuple[int, int] | None = None
        super().__init__(parent, "Eigenface builder")

    def body(self, master: tk.Frame) -> tk.Widget:
        frame = ttk.LabelFrame(master, text="Parameters", padding=8)
        frame.pack(fill="x", padx=8, pady=8)
        self.components = tk.StringVar(master, value=str(DEFAULT_COMPONENTS))
        self.shown = tk.StringVar(master, value=str(DEFAULT_SHOWN))

        ttk.Label(frame, text="Eigenfaces to keep (K):").grid(row=0, column=0, sticky="w", pady=2)
        first = ttk.Spinbox(frame, from_=1, to=500, increment=10, textvariable=self.components, width=8)
        first.grid(row=0, column=1, padx=(6, 0), pady=2)
        ttk.Label(frame, text="Eigenfaces to show:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(frame, from_=1, to=40, increment=1, textvariable=self.shown, width=8).grid(
            row=1, column=1, padx=(6, 0), pady=2
        )
        return first

    def validate(self) -> bool:
        try:
            components, shown = int(self.components.get()), int(self.shown.get())
            valid = components >= 1 and shown >= 1
        except ValueError:
            valid = False
        if not valid:
            messagebox.showwarning(self.title(), "Both values must be integers >= 1.", parent=self)
            return False
        self.options = (components, shown)
        return True


def _ask_options(parent: tk.Misc) -> tuple[int, int] | None:
    return EigenfaceDialog(parent).options


def _ask_train_folder(parent: tk.Misc) -> Path | None:
    """Yes -> LFW exported to datasets/lfw, No -> any folder with one subfolder per person."""
    use_lfw = messagebox.askyesnocancel(
        "Eigenface builder",
        f"Use the LFW dataset? It is downloaded and saved to '{DATASET_DIR.relative_to(PROJECT_ROOT)}' "
        "(train/test) the first time.\n"
        "(No = choose your own train folder with one subfolder per person)",
        parent=parent,
    )
    if use_lfw is None:
        return None
    if use_lfw:
        return DATASET_DIR / "train"
    folder = filedialog.askdirectory(title="Train folder (one subfolder per person)", parent=parent, mustexist=True)
    return Path(folder) if folder else None


def _show_gallery(figure: plt.Figure) -> None:
    plt.show(block=False)


def _ask_neighbours(parent: tk.Misc) -> int | None:
    return simpledialog.askinteger(
        "Face recognition", f"Closest training faces to show (1-{MAX_NEIGHBOURS}):", parent=parent,
        initialvalue=DEFAULT_NEIGHBOURS, minvalue=1, maxvalue=MAX_NEIGHBOURS,
    )


def _show_matches(figure: plt.Figure) -> None:
    plt.show(block=False)

# ---------- Tools ----------

class EigenfaceBuilderTool(ForensicsTool):
    tool_id = "eigenface_builder"
    title = "Eigenface builder (PCA)"
    category = "Faces"
    description = (
        "Fit a PCA on a folder of training faces (LFW by default), save mean, eigenfaces and "
        "coefficients to models/eigenfaces.npz and plot the first eigenfaces."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        train_folder = _ask_train_folder(parent)
        if train_folder is None:
            return None
        options = _ask_options(parent)
        if options is None:
            return None
        n_components, shown = options

        # Download + PCA block the window for a while: show a busy cursor meanwhile
        window = parent.winfo_toplevel()
        window.configure(cursor="watch")
        window.update_idletasks()
        try:
            if train_folder == DATASET_DIR / "train" and not train_folder.is_dir():
                export_lfw()
            faces = load_face_folder(train_folder)
            model = build_eigenfaces(faces, n_components)
            model_file = model.save(MODEL_FILE) # save as .npz with 7 arrays: mean, eigenfaces, coefficients, labels, paths, image_shape y explained_variance_ratio
        finally:
            window.configure(cursor="")

        _show_gallery(plot_eigenfaces(model, shown))
        height, width = model.image_shape
        explained = model.explained_variance_ratio.sum()
        return ToolResult(
            message=f"Built {model.n_components} eigenfaces from {len(faces.labels)} faces "
            f"({explained:.1%} of the variance) and saved them to {model_file.name}.",
            details={
                "Operation": "Eigenface builder (PCA)",
                "Train folder": str(train_folder),
                "Training faces": str(len(faces.labels)),
                "People": str(len(np.unique(faces.labels))),
                "Face size (W x H)": f"{width} x {height}",
                "Eigenfaces (K)": str(model.n_components),
                "Explained variance": f"{explained:.1%}",
                "Mean shape": str(model.mean.shape),
                "Eigenfaces shape": str(model.eigenfaces.shape),
                "Coefficients shape": str(model.coefficients.shape),
                "Model file": str(model_file),
            },
        )


class FaceRecognitionTool(ForensicsTool):
    tool_id = "face_recognition"
    title = "Face recognition (eigenfaces)"
    category = "Faces"
    description = (
        "Project the current face onto the saved eigenfaces and show the closest "
        "training faces with their names."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if not MODEL_FILE.is_file():
            raise ValueError("No eigenface model found. Run 'Eigenface builder (PCA)' first.")
        k = _ask_neighbours(parent)
        if k is None:
            return None

        assert document.current is not None
        model = EigenfaceModel.load(MODEL_FILE)
        face = prepare_face(document.current, model.image_shape)
        coefficients = project(face[None], model.mean, model.eigenfaces)[0]  # (K,) w of the test face
        matches = find_closest(model, coefficients, k)

        # The true name is only known when the image sits in a person folder of the model (e.g. lfw/test/<Name>/)
        true_label = face_label(document.path) if document.path else None
        if true_label not in model.labels:
            true_label = None
        prediction = matches[0].label
        if true_label is None:
            verdict, correct = "", "-"
        elif prediction == true_label:
            verdict, correct = " (correct)", "yes"
        else:
            verdict, correct = f" (wrong, it is {true_label})", "no"

        _show_matches(plot_matches(face, true_label, matches, model))
        width, height = document.current.size
        model_height, model_width = model.image_shape
        details = {
            "Operation": "Face recognition (eigenfaces)",
            "Prediction": prediction,
            "True name": true_label or "Unknown",
            "Correct": correct,
            "Input size (W x H)": f"{width} x {height}",
            "Face size (W x H)": f"{model_width} x {model_height}",
            "Eigenfaces (K)": str(model.n_components),
            "Training faces": str(len(model.labels)),
            **{
                f"Match {number}": f"{match.label} (d = {match.distance:.2f})"
                for number, match in enumerate(matches, start=1)
            },
        }
        if matches[0].distance < 1e-6:
            details["Note"] = "The closest face is identical: this image belongs to the training set."
        return ToolResult(
            message=f"Closest face: {prediction} (d = {matches[0].distance:.2f}){verdict}.",
            details=details,
        )
