import unittest
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.yolo_detection import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    Detection,
    YoloDetectionTool,
    describe_detections,
    detect_objects,
    parse_detections,
    plot_detections,
)

NAMES = {0: "person", 5: "bus"}
# Two YOLO outputs: a person with 0.6 and a bus with 0.9 (not sorted, as the model may return them)
BOXES = np.array([[10, 20, 30, 60], [0, 5, 80, 50]], dtype=np.float32)
SCORES = np.array([0.6, 0.9], dtype=np.float32)
CLASSES = np.array([0, 5], dtype=np.float32)


def _document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def _tensor(array: np.ndarray) -> MagicMock:
    """Stand-in for a torch tensor: only .cpu().numpy() is used."""
    tensor = MagicMock()
    tensor.cpu.return_value.numpy.return_value = array
    return tensor


class ParseDetectionsTests(unittest.TestCase):
    def test_best_score_first_with_names(self) -> None:
        detections = parse_detections(BOXES, SCORES, CLASSES, NAMES)
        self.assertEqual([d.label for d in detections], ["bus", "person"])
        self.assertEqual(detections[0].box, (0.0, 5.0, 80.0, 50.0))
        self.assertEqual(detections[0].class_id, 5)
        self.assertAlmostEqual(detections[0].score, 0.9, places=5)

    def test_no_boxes_means_no_detections(self) -> None:
        empty = np.zeros((0, 4), dtype=np.float32)
        self.assertEqual(parse_detections(empty, np.zeros(0), np.zeros(0), NAMES), [])


class DetectObjectsTests(unittest.TestCase):
    def test_passes_thresholds_and_reads_boxes(self) -> None:
        result = MagicMock(names=NAMES)
        result.boxes.xyxy, result.boxes.conf, result.boxes.cls = _tensor(BOXES), _tensor(SCORES), _tensor(CLASSES)
        model = MagicMock()
        model.predict.return_value = [result]
        frame = np.zeros((60, 80, 3), dtype=np.uint8)

        detections = detect_objects(model, frame, 0.4, 0.5)

        model.predict.assert_called_once_with(frame, conf=0.4, iou=0.5, verbose=False)
        self.assertEqual(len(detections), 2)


class PlotAndDescribeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detections = parse_detections(BOXES, SCORES, CLASSES, NAMES)

    def test_one_rectangle_per_detection(self) -> None:
        figure = plot_detections(Image.new("RGB", (80, 60), "white"), self.detections)
        self.assertEqual(len(figure.axes[0].patches), 2)
        self.assertEqual(figure.axes[0].get_title(), "Detected Objects")
        plt.close(figure)

    def test_counts_per_class_and_boxes(self) -> None:
        details = describe_detections(self.detections + [Detection((1, 1, 2, 2), 0, "person", 0.3)])
        self.assertEqual(details["Count person"], "2")
        self.assertEqual(details["Count bus"], "1")
        self.assertEqual(details["Object 1"], "bus 0.90 (x1:0, y1:5, x2:80, y2:50)")


class YoloDetectionToolTests(unittest.TestCase):
    @patch("forensics_app.tools.yolo_detection.detect_objects")
    @patch("forensics_app.tools.yolo_detection.load_yolo")
    @patch("forensics_app.tools.yolo_detection._ask_options", return_value=(DEFAULT_CONFIDENCE, DEFAULT_IOU))
    def test_returns_plot_and_details(self, _ask_options, _load_yolo, detect) -> None:
        detect.return_value = parse_detections(BOXES, SCORES, CLASSES, NAMES)
        document = _document(Image.new("RGB", (80, 60), "white"))

        result = YoloDetectionTool().run(MagicMock(), document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.details["Objects"], "2")
        self.assertEqual(result.details["Count bus"], "1")
        frame = detect.call_args.args[1]
        self.assertEqual(frame.shape, (60, 80, 3))  # BGR numpy frame, as Ultralytics expects
        self.assertEqual(document.current.getpixel((5, 5)), (255, 255, 255))

    @patch("forensics_app.tools.yolo_detection._ask_options", return_value=None)
    def test_cancel_returns_none(self, _ask_options) -> None:
        self.assertIsNone(YoloDetectionTool().run(MagicMock(), _document(Image.new("RGB", (8, 8)))))


if __name__ == "__main__":
    unittest.main()
