from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ToolResult
from forensics_app.tools.bow_classification import (
    ClassifyImageTool,
    SplitDatasetTool,
    classify,
    export_split,
    image_histograms,
    render_prediction,
    train_classifier,
)
from forensics_app.tools.codebook import load_training_data, save_codebook
from forensics_app.tools.feature_matching import FeatureParams

IMAGES_PER_CLASS = 8
DESCRIPTORS_PER_IMAGE = 20
DESCRIPTOR_LENGTH = 8
K = 4


def _write_images(root: Path, classes: dict[str, int]) -> None:
    for label, count in classes.items():
        (root / label).mkdir(parents=True)
        for index in range(count):
            Image.new("L", (32, 32), index * 10).save(root / label / f"image_{index:02d}.png")


def _class_descriptors(offset: float, seed: int) -> np.ndarray:
    """Descriptors of one image of a class: noise around `offset`, so both classes use different words."""
    rng = np.random.default_rng(seed)
    return offset + 0.1 * rng.random((DESCRIPTORS_PER_IMAGE, DESCRIPTOR_LENGTH))


def _write_codebook(root: Path) -> Path:
    """Codebook of 2 classes: 'cats' descriptors near 0 and 'dogs' near 1."""
    descriptors, owners, paths = [], [], []
    for index in range(2 * IMAGES_PER_CLASS):
        label, offset = ("cats", 0.0) if index < IMAGES_PER_CLASS else ("dogs", 1.0)
        descriptors.append(_class_descriptors(offset, index))
        owners.extend([index] * DESCRIPTORS_PER_IMAGE)
        paths.append(root / "train" / label / f"image_{index:02d}.png")
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=3).fit(np.vstack(descriptors))
    return save_codebook(kmeans, FeatureParams(), owners, paths, root / "codebook.npz")


class SplitTests(unittest.TestCase):
    def test_split_is_stratified_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_images(root / "source", {"cats": 8, "dogs": 4})
            n_train, n_test = export_split(root / "source", root / "split")
            train = {path.relative_to(root / "split" / "train") for path in (root / "split" / "train").rglob("*.png")}
            test = {path.relative_to(root / "split" / "test") for path in (root / "split" / "test").rglob("*.png")}

        self.assertEqual((n_train, n_test), (9, 3))
        self.assertEqual(len(train), 9)
        self.assertFalse(train & test)
        self.assertEqual(sum(1 for path in test if path.parent.name == "cats"), 2)
        self.assertEqual(sum(1 for path in test if path.parent.name == "dogs"), 1)

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_images(root / "source", {"cats": 4, "dogs": 4})
            (root / "split").mkdir()
            with self.assertRaises(ValueError):
                export_split(root / "source", root / "split")

    def test_needs_two_classes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_images(root / "source", {"cats": 4})
            with self.assertRaises(ValueError):
                export_split(root / "source", root / "split")


class TrainingDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.codebook_file = _write_codebook(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_and_load_keep_owners_and_classes(self) -> None:
        words, owners, labels, paths = load_training_data(self.codebook_file)
        self.assertEqual(len(words), len(owners))
        self.assertEqual(len(labels), 2 * IMAGES_PER_CLASS)
        self.assertEqual(sorted(set(labels)), ["cats", "dogs"])
        self.assertTrue(paths[0].endswith("image_00.png"))

    def test_old_codebook_is_rejected(self) -> None:
        old_file = Path(self.tmp.name) / "old.npz"
        np.savez(old_file, centers=np.zeros((K, DESCRIPTOR_LENGTH)), words=np.zeros(3, dtype=int))
        with self.assertRaises(ValueError):
            load_training_data(old_file)

    def test_histograms_are_normalized_and_skip_empty_images(self) -> None:
        words = np.array([0, 1, 1, 2, 2, 2])
        owners = np.array([0, 0, 0, 2, 2, 2])  # image 1 has no descriptors
        histograms, labels = image_histograms(words, owners, np.array(["a", "b", "c"]), 3)

        np.testing.assert_allclose(histograms.sum(axis=1), 1.0)
        np.testing.assert_allclose(histograms[0], [1 / 3, 2 / 3, 0])
        self.assertEqual(list(labels), ["a", "c"])


class ClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        words, owners, labels, _ = load_training_data(_write_codebook(Path(tempfile.mkdtemp())))
        self.histograms, self.labels = image_histograms(words, owners, labels, K)

    def test_separable_classes_are_recognised(self) -> None:
        classifier = train_classifier(self.histograms, self.labels)
        for histogram, label in zip(self.histograms, self.labels):
            self.assertEqual(classify(classifier, histogram)[0][0], label)

    def test_probabilities_are_sorted_and_sum_to_one(self) -> None:
        ranking = classify(train_classifier(self.histograms, self.labels), self.histograms[0])
        probabilities = [probability for _, probability in ranking]
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_needs_two_classes(self) -> None:
        with self.assertRaises(ValueError):
            train_classifier(self.histograms, np.array(["cats"] * len(self.histograms)))

    def test_prediction_image_keeps_the_aspect_ratio(self) -> None:
        image = render_prediction(Image.new("RGB", (600, 300), "orange"), "kangaroo", 0.64)
        width, height = image.size
        self.assertEqual(width, 600)
        self.assertGreater(height, 300)  # room for the title


class ClassifyImageToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.codebook_file = _write_codebook(root)
        _write_images(root / "test", {"cats": 1, "dogs": 1})
        self.document = ImageDocument()
        self.document.load(root / "test" / "cats" / "image_00.png")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, descriptors: np.ndarray) -> ToolResult | None:
        keypoints = np.full((len(descriptors), 2), 16.0)
        with patch("forensics_app.tools.bow_classification.CODEBOOK_FILE", self.codebook_file), \
                patch("forensics_app.tools.bow_classification.extract_features", return_value=(keypoints, descriptors)):
            return ClassifyImageTool().run(MagicMock(), self.document)

    @patch("forensics_app.tools.bow_classification._ask_classify", return_value=True)
    def test_classifies_the_current_image(self, ask_classify) -> None:
        result = self._run(_class_descriptors(0.0, seed=99))

        self.assertEqual(result.details["Prediction"], "cats")
        self.assertEqual(result.details["True class"], "cats")
        self.assertEqual(result.details["Correct"], "yes")
        self.assertIn("p(dogs)", result.details)
        self.assertEqual(result.details["Training images"], "cats 8, dogs 8")
        self.assertNotIn("Note", result.details)
        self.assertTrue(result.message.startswith("p(cats) = "))
        self.assertIsNotNone(result.image)
        ask_classify.assert_called_once()

    @patch("forensics_app.tools.bow_classification._ask_classify", return_value=True)
    def test_wrong_prediction_is_reported(self, _ask_classify) -> None:
        result = self._run(_class_descriptors(1.0, seed=99))  # a 'dog' image inside test/cats/
        self.assertEqual(result.details["Prediction"], "dogs")
        self.assertEqual(result.details["Correct"], "no")
        self.assertIn("wrong, it is cats", result.message)

    @patch("forensics_app.tools.bow_classification._ask_classify", return_value=False)
    def test_cancel_returns_none(self, _ask_classify) -> None:
        self.assertIsNone(self._run(_class_descriptors(0.0, seed=99)))

    def test_image_without_keypoints_is_reported(self) -> None:
        with self.assertRaises(ValueError):
            self._run(np.empty((0, DESCRIPTOR_LENGTH)))


class SplitDatasetToolTests(unittest.TestCase):
    def test_exports_to_the_datasets_folder(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_images(root / "animals", {"cats": 4, "dogs": 4})
            with patch("forensics_app.tools.bow_classification._ask_source_folder", return_value=root / "animals"), \
                    patch("forensics_app.tools.bow_classification.DATASETS_DIR", root / "datasets"):
                result = SplitDatasetTool().run(MagicMock(), ImageDocument())
            self.assertTrue((root / "datasets" / "animals_split" / "test" / "dogs").is_dir())

        self.assertEqual(result.details["Classes"], "cats, dogs")
        self.assertEqual(result.details["Train images"], "6")
        self.assertEqual(result.details["Test images"], "2")

    @patch("forensics_app.tools.bow_classification._ask_source_folder", return_value=None)
    def test_cancel_returns_none(self, _ask_source_folder) -> None:
        self.assertIsNone(SplitDatasetTool().run(MagicMock(), ImageDocument()))


if __name__ == "__main__":
    unittest.main()
