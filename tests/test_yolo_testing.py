from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.yolo_detection import Detection
from forensics_app.tools.yolo_testing import (
    GroundTruthBox,
    YoloTestingTool,
    count_per_class,
    label_file_for,
    plot_ground_truth_and_predictions,
    read_ground_truth,
    yolo_to_xyxy,
)

NAMES = {0: "PC", 1: "RBC", 2: "WBC"}
# Same boxes as in test_yolo_training: a RBC and a WBC on a 100 x 50 image
LABELS = "1 0.200000 0.300000 0.200000 0.200000\n2 0.750000 0.500000 0.500000 1.000000"
DETECTIONS = [Detection(box=(11.0, 9.0, 31.0, 21.0), class_id=1, label="RBC", score=0.9)]


def _write_test_image(folder: Path) -> Path:
    """<folder>/test/images/a.jpg with its labels in <folder>/test/labels/a.txt."""
    (folder / "test" / "images").mkdir(parents=True)
    (folder / "test" / "labels").mkdir(parents=True)
    image_path = folder / "test" / "images" / "a.jpg"
    Image.new("RGB", (100, 50)).save(image_path)
    (folder / "test" / "labels" / "a.txt").write_text(LABELS)
    return image_path


class GroundTruthTests(unittest.TestCase):
    def test_yolo_box_back_to_pixels(self) -> None:
        for value, expected in zip(yolo_to_xyxy((0.2, 0.3, 0.2, 0.2), 100, 50), (10, 10, 30, 20)):
            self.assertAlmostEqual(value, expected)

    def test_label_file_next_to_images_folder(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            image_path = _write_test_image(Path(folder))
            self.assertEqual(label_file_for(image_path), Path(folder) / "test" / "labels" / "a.txt")
            self.assertIsNone(label_file_for(Path(folder) / "elsewhere" / "a.jpg"))
            self.assertIsNone(label_file_for(None))

    def test_reads_every_box_with_names(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            label_file = Path(folder) / "a.txt"
            label_file.write_text(LABELS + "\n")
            boxes = read_ground_truth(label_file, 100, 50, NAMES)
        self.assertEqual([box.label for box in boxes], ["RBC", "WBC"])
        for value, expected in zip(boxes[1].box, (50, 0, 100, 50)):
            self.assertAlmostEqual(value, expected)

    def test_counts_ground_truth_and_predictions_per_class(self) -> None:
        truth = [GroundTruthBox((0, 0, 1, 1), 1, "RBC"), GroundTruthBox((0, 0, 1, 1), 2, "WBC")]
        self.assertEqual(
            count_per_class(truth, DETECTIONS, NAMES),
            {"Count PC": "GT 0 / Pred 0", "Count RBC": "GT 1 / Pred 1", "Count WBC": "GT 1 / Pred 0"},
        )

    def test_plot_draws_one_rectangle_per_box(self) -> None:
        truth = [GroundTruthBox((10, 10, 30, 20), 1, "RBC")]
        figure = plot_ground_truth_and_predictions(Image.new("RGB", (100, 50)), truth, DETECTIONS)
        self.assertEqual(len(figure.axes[0].patches), 2)
        plt.close(figure)


class YoloTestingToolTests(unittest.TestCase):
    def test_draws_predictions_over_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            document = ImageDocument()
            document.load(_write_test_image(Path(folder)))
            model = MagicMock(names=NAMES)

            with patch("forensics_app.tools.yolo_testing._ask_options", return_value=(0.25, 0.7)), \
                    patch("forensics_app.tools.yolo_testing.load_trained", return_value=model), \
                    patch("forensics_app.tools.yolo_testing.detect_objects", return_value=DETECTIONS):
                result = YoloTestingTool().run(MagicMock(), document)

        self.assertIsNotNone(result.image)
        self.assertEqual(result.details["Split"], "test")
        self.assertEqual(result.details["Count RBC"], "GT 1 / Pred 1")
        self.assertIn("ground truth has 2", result.message)

    def test_without_labels_only_predictions(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (100, 50))
        with patch("forensics_app.tools.yolo_testing._ask_options", return_value=(0.25, 0.7)), \
                patch("forensics_app.tools.yolo_testing.load_trained", return_value=MagicMock(names=NAMES)), \
                patch("forensics_app.tools.yolo_testing.detect_objects", return_value=DETECTIONS):
            result = YoloTestingTool().run(MagicMock(), document)

        self.assertEqual(result.details["Ground truth file"], "not found")
        self.assertIn("No ground truth found", result.message)

    @patch("forensics_app.tools.yolo_testing._ask_options", return_value=None)
    def test_cancel_returns_none(self, _ask_options) -> None:
        self.assertIsNone(YoloTestingTool().run(MagicMock(), ImageDocument()))


if __name__ == "__main__":
    unittest.main()
