import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.yolo_training import (
    COCO_ANNOTATIONS,
    DATA_FILE,
    YoloTrainingTool,
    coco_class_names,
    coco_to_yolo_box,
    convert_coco_to_yolo,
    load_trained,
    save_trained,
    train_yolo,
    yolo_dataset_dir,
)

# Roboflow COCO export: category 0 is the parent of the real classes and has no boxes
CATEGORIES = [
    {"id": 0, "name": "cells", "supercategory": "none"},
    {"id": 1, "name": "PC", "supercategory": "cells"},
    {"id": 2, "name": "RBC", "supercategory": "cells"},
    {"id": 3, "name": "WBC", "supercategory": "cells"},
]
METRICS = {
    "metrics/precision(B)": 0.8, "metrics/recall(B)": 0.7, "metrics/mAP50(B)": 0.75,
    "metrics/mAP50-95(B)": 0.5, "fitness": 0.5,
}


def _write_split(folder: Path) -> None:
    """Two 100 x 50 images: a.jpg with a RBC and a WBC, b.jpg with no boxes."""
    folder.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (100, 50)).save(folder / name)
    coco = {
        "categories": CATEGORIES,
        "images": [
            {"id": 0, "file_name": "a.jpg", "width": 100, "height": 50},
            {"id": 1, "file_name": "b.jpg", "width": 100, "height": 50},
        ],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 2, "bbox": [10, 10, 20, 10]},
            {"id": 1, "image_id": 0, "category_id": 3, "bbox": [50, 0, 50, 50]},
        ],
    }
    (folder / COCO_ANNOTATIONS).write_text(json.dumps(coco))


class CocoToYoloTests(unittest.TestCase):
    def test_box_is_centre_and_size_over_image_size(self) -> None:
        self.assertEqual(coco_to_yolo_box([10, 10, 20, 10], 100, 50), (0.2, 0.3, 0.2, 0.2))

    def test_parent_category_is_dropped(self) -> None:
        self.assertEqual(coco_class_names(CATEGORIES), {1: "PC", 2: "RBC", 3: "WBC"})

    def test_converts_every_split(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "cells_coco", Path(folder) / "cells_yolo"
            for split in ("train", "valid"):
                _write_split(source / split)

            data_file, names, counts = convert_coco_to_yolo(source, output)

            self.assertEqual(names, ["PC", "RBC", "WBC"])
            self.assertEqual(counts, {"train": (2, 2), "valid": (2, 2)})
            self.assertTrue((output / "valid" / "images" / "a.jpg").is_file())
            labels = (output / "train" / "labels" / "a.txt").read_text().splitlines()
            self.assertEqual(labels, ["1 0.200000 0.300000 0.200000 0.200000", "2 0.750000 0.500000 0.500000 1.000000"])
            self.assertEqual((output / "train" / "labels" / "b.txt").read_text(), "")
            data = data_file.read_text()
            self.assertIn(f"path: {output.resolve()}", data)
            self.assertIn("val: valid/images", data)
            self.assertIn("  2: WBC", data)

    def test_needs_train_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _write_split(Path(folder) / "train")
            with self.assertRaises(ValueError):
                convert_coco_to_yolo(folder, Path(folder) / "out")

    def test_dataset_dir_name(self) -> None:
        self.assertEqual(yolo_dataset_dir(Path("blood_cell_coco")).name, "blood_cell_yolo")


class TrainAndSaveTests(unittest.TestCase):
    @patch("forensics_app.tools.yolo_training.training_device", return_value="cpu")
    def test_train_passes_options_and_cleans_metric_names(self, _device) -> None:
        model = MagicMock()
        model.train.return_value.results_dict = METRICS

        metrics = train_yolo(model, Path("data.yaml"), 3, 320, Path("runs"))

        model.train.assert_called_once_with(
            data="data.yaml", epochs=3, imgsz=320, device="cpu",
            project="runs", name="blood_cell", exist_ok=True, verbose=False,
        )
        self.assertEqual(metrics["mAP50"], 0.75)
        self.assertEqual(metrics["mAP50-95"], 0.5)

    def test_save_copies_best_weights_and_load_needs_them(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            best = Path(folder) / "best.pt"
            best.write_bytes(b"weights")
            model = MagicMock()
            model.trainer.best = str(best)

            path = save_trained(model, Path(folder) / "models" / "trained.pt")

            self.assertEqual(path.read_bytes(), b"weights")
            with self.assertRaises(ValueError):
                load_trained(Path(folder) / "missing.pt")


class YoloTrainingToolTests(unittest.TestCase):
    def test_converts_trains_and_saves(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "run"
            run_dir.mkdir()
            Image.new("RGB", (30, 20), "white").save(run_dir / "results.png")
            model = MagicMock(names={0: "PC", 1: "RBC", 2: "WBC"})
            model.trainer.save_dir = str(run_dir)
            metrics = {"precision": 0.8, "recall": 0.7, "mAP50": 0.75, "mAP50-95": 0.5}

            with patch("forensics_app.tools.yolo_training._ask_coco_folder", return_value=Path(folder) / "cells_coco"), \
                    patch("forensics_app.tools.yolo_training._ask_options", return_value=(3, 320)), \
                    patch("forensics_app.tools.yolo_training.DATASETS_DIR", Path(folder)), \
                    patch("forensics_app.tools.yolo_training.convert_coco_to_yolo") as convert, \
                    patch("forensics_app.tools.yolo_training.load_yolo", return_value=model), \
                    patch("forensics_app.tools.yolo_training.train_yolo", return_value=metrics) as train, \
                    patch("forensics_app.tools.yolo_training.save_trained", return_value=Path(folder) / "trained.pt"), \
                    patch("forensics_app.tools.yolo_training.training_device", return_value="cpu"):
                result = YoloTrainingTool().run(MagicMock(), ImageDocument())

            convert.assert_called_once_with(Path(folder) / "cells_coco", Path(folder) / "cells_yolo")
            self.assertEqual(train.call_args.args[1:], (Path(folder) / "cells_yolo" / DATA_FILE, 3, 320))
            self.assertEqual(result.image.size, (30, 20))
            self.assertEqual(result.details["Classes"], "PC, RBC, WBC")
            self.assertEqual(result.details["mAP50"], "0.750")
            self.assertIn("trained.pt", result.message)

    @patch("forensics_app.tools.yolo_training._ask_coco_folder", return_value=None)
    def test_cancel_returns_none(self, _ask_coco_folder) -> None:
        self.assertIsNone(YoloTrainingTool().run(MagicMock(), ImageDocument()))


if __name__ == "__main__":
    unittest.main()
