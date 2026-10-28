

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import urllib.request

import cv2
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult 

# 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MIN_MODEL_BYTES = 100_000  # the real model is ~230 KB; smaller means a broken download
# we set a threshold of minimum bytes

DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_TOP_K = 5000

LANDMARK_NAMES = ("Right eye", "Left eye", "Nose", "Right mouth corner", "Left mouth corner")
BOX_COLOR = (0, 0, 255)  # BGR, as everything drawn with OpenCV
LANDMARK_COLOR = (255, 0, 0)
LINE_WIDTH = 2 
LANDMARK_RADIUS = 4

WEBCAM_INDEX = 0
WEBCAM_WINDOW = "YuNet webcam (press q or Esc to quit)"
QUIT_KEYS = (ord("q"), 27)

@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]  # (x, y, width, height)
    # (x, y) origin coord at the left top corner of the box; width and height of the box in pixels
    landmarks: np.ndarray  # (5, 2) int, (x, y) in LANDMARK_NAMES order
    # contain the coords (default 2: (x, y)) of the landmarks (5 by default: eyes, mouth corners and nose) (row: landmark points, columns: coords)
    score: float
    # confidence of the model 
    
# ---------- Pure functions ----------

def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the pretrained YuNet model the first time it is needed."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
    if path.stat().st_size < MIN_MODEL_BYTES:
        path.unlink()
        raise RuntimeError("YuNet model download failed or the file is incomplete. Try again.")
    return path

# YuNet provides many boxes with given scores over different positions (candidates).
# Candidates are first filtered by score_threshold and then the top_k are selected.
# Non-maximum suppression (nms), as it name says, removes the overlapping boxes to the max score one (intersection over union).
# candidates whose IoU with the kept box is higher than nms_threshold are remove (overlapping too much to be considered a different box)
def create_detector(
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> cv2.FaceDetectorYN:
    """
    Builds and returns a cv2.FaceDetectorYN already configured. 
    score_threshold [0,1]: minimum confidence to accept a detection.
    nms_threshold [0,1]: threshold to suppress overlaping boxes
    top_k: int: maximum number of candidates passed to the threshold
    """
    if not 0.0 <= score_threshold <= 1.0 or not 0.0 <= nms_threshold <= 1.0:
        raise ValueError("Score and NMS thresholds must be in [0, 1].")
    # The input size is a placeholder: detect_faces sets the real one for every image
    return cv2.FaceDetectorYN.create(
        str(ensure_model()), "", (320, 320), score_threshold, nms_threshold, top_k
    )
    
def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# YuNet output (raw) contains the detection in an float32 array with shape (N, 15)
# N is the number of detected faces; the 15 are the elements of face: box arguments (4), landmarks positions (2x5) and the score (1)
def parse_faces(raw: np.ndarray | None) -> list[Face]:
    """Turn YuNet's (N, 15) output rows into Face objects.
    Row layout: x, y, w, h, 5 landmarks as (x, y) pairs, score.
    """
    if raw is None:
        return []
    return [
        Face(
            box=tuple(int(value) for value in row[:4]),
            landmarks=row[4:14].reshape(5, 2).astype(int),
            score=float(row[14]),
        )
        for row in raw
    ]



def detect_faces(detector: cv2.FaceDetectorYN, frame: np.ndarray) -> list[Face]:
    """Builds a list of Face objects, containing the detected elements from frame image."""
    height, width = frame.shape[:2]
    detector.setInputSize((width, height)) # scale to the frame's actual size
    _, raw = detector.detect(frame) # obtain the detector's output
    return parse_faces(raw)


def draw_faces(frame: np.ndarray, faces: list[Face]) -> np.ndarray:
    """Draws boxes around the provided detected faces on the frame"""
    output = frame.copy() # avoid drawing on the original image
    for face in faces:
        x, y, w, h = face.box 
        cv2.rectangle(output, (x, y), (x + w, y + h), BOX_COLOR, LINE_WIDTH)
        cv2.putText(
            output, f"Face: {face.score:.2f}", (x, max(20, y - 8)), # avoid text out of the image
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, LINE_WIDTH,
        )
        for px, py in face.landmarks:
            cv2.circle(output, (int(px), int(py)), LANDMARK_RADIUS, LANDMARK_COLOR, -1)
    return output

def run_webcam(detector: cv2.FaceDetectorYN, camera_index: int = WEBCAM_INDEX) -> int:
    """Detect faces live until the user presses q/Esc or closes the window.

    Returns the number of processed frames.
    """
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open the webcam (index {camera_index}).")
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            cv2.imshow(WEBCAM_WINDOW, draw_faces(frame, detect_faces(detector, frame)))
            key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WEBCAM_WINDOW, cv2.WND_PROP_VISIBLE) < 1
            if key in QUIT_KEYS or window_closed:
                break
    finally:
        capture.release()
        cv2.destroyWindow(WEBCAM_WINDOW)
    return frames


def describe_faces(faces: list[Face]) -> dict[str, str]:
    """Report detected face summary into a dict"""
    details: dict[str, str] = {} 
    for number, face in enumerate(faces, start=1): # not starting at face 0
        x, y, w, h = face.box
        details[f"Face {number} box"] = f"(x:{x}, y:{y}, w:{w}, h:{h})"
        details[f"Face {number} score"] = f"{face.score:.3f}"
        for (px, py), name in zip(face.landmarks, LANDMARK_NAMES): 
            details[f"Face {number} {name.lower()}"] = f"({px}, {py})"
    return details

# ---------- Dialog ----------

class FaceDetectionDialog(simpledialog.Dialog):
    def __init__(self, parent: tk.Misc, title: str) -> None:
        self.options: tuple[float, float] | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Frame) -> tk.Widget:
        frame = ttk.LabelFrame(master, text="Parameters", padding=8)
        frame.pack(fill="x", padx=8, pady=8)
        self.score = tk.StringVar(master, value=str(DEFAULT_SCORE_THRESHOLD))
        self.nms = tk.StringVar(master, value=str(DEFAULT_NMS_THRESHOLD))

        ttk.Label(frame, text="Score threshold (0-1):").grid(row=0, column=0, sticky="w", pady=2)
        first = ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.score, width=8)
        first.grid(row=0, column=1, padx=(6, 0), pady=2)
        ttk.Label(frame, text="NMS threshold (0-1):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.nms, width=8).grid(
            row=1, column=1, padx=(6, 0), pady=2
        )
        return first

    def validate(self) -> bool:
        try:
            score, nms = float(self.score.get()), float(self.nms.get())
            valid = 0.0 <= score <= 1.0 and 0.0 <= nms <= 1.0
        except ValueError:
            valid = False
        if not valid:
            messagebox.showwarning(self.title(), "Both thresholds must be numbers in [0, 1].", parent=self)
            return False
        self.options = (score, nms)
        return True


def _ask_options(parent: tk.Misc, title: str) -> tuple[float, float] | None:
    return FaceDetectionDialog(parent, title).options


# ---------- Tools ----------

class FaceDetectionTool(ForensicsTool):
    tool_id = "face_detection"
    title = "Face detection (YuNet)"
    category = "Faces"
    description = "Detect faces with the YuNet CNN: green boxes with their score and 5 blue landmarks."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_options(parent, self.title)
        if options is None:
            return None
        score_threshold, nms_threshold = options

        assert document.current is not None
        frame = pil_to_bgr(document.current)
        faces = detect_faces(create_detector(score_threshold, nms_threshold), frame)
        width, height = document.current.size
        return ToolResult(
            image=bgr_to_pil(draw_faces(frame, faces)),
            message=f"Detected {len(faces)} face(s) with YuNet (score >= {score_threshold:g}).",
            details={
                "Operation": "Face detection (YuNet)",
                "Image size (w x h)": f"{width} x {height}",
                "Score threshold": f"{score_threshold:g}",
                "NMS threshold": f"{nms_threshold:g}",
                "Faces": str(len(faces)),
                **describe_faces(faces),
            },
        )


class WebcamFaceDetectionTool(ForensicsTool):
    tool_id = "webcam_face_detection"
    title = "Webcam face detection (YuNet)"
    category = "Faces"
    description = "Run YuNet live on the webcam stream in a separate window (press q or Esc to stop)."
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        options = _ask_options(parent, self.title)
        if options is None:
            return None
        score_threshold, nms_threshold = options

        frames = run_webcam(create_detector(score_threshold, nms_threshold))
        return ToolResult(
            message=f"Webcam stream closed after {frames:,} frames.",
            details={
                "Operation": "Webcam face detection (YuNet)",
                "Camera index": str(WEBCAM_INDEX),
                "Score threshold": f"{score_threshold:g}",
                "NMS threshold": f"{nms_threshold:g}",
                "Frames processed": f"{frames:,}",
            },
        )