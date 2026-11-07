from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ToolResult
from forensics_app.tools.eigenfaces import (
    EigenfaceBuilderTool,
    EigenfaceModel,
    FaceRecognitionTool,
    build_eigenfaces,
    find_closest,
    load_face_folder,
    plot_eigenfaces,
    plot_matches,
    prepare_face,
    project,
    save_faces,
    train_face,
)

FACE_SHAPE = (12, 10)  # (H, W)


def _random_faces(n: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, *FACE_SHAPE)).astype(np.float32)


def _write_folder(root: Path) -> None:
    save_faces(_random_faces(), ["Ada Lovelace"] * 4 + ["Alan Turing"] * 4, root)


class FolderTests(unittest.TestCase):
    def test_save_and_load_keep_names_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _write_folder(Path(folder))
            faces = load_face_folder(folder)

        self.assertEqual(faces.images.shape, (8, *FACE_SHAPE))
        self.assertEqual(faces.image_shape, FACE_SHAPE)
        self.assertEqual(sorted(set(faces.labels)), ["Ada Lovelace", "Alan Turing"])
        self.assertLessEqual(faces.images.max(), 1.0)

    def test_empty_folder_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                load_face_folder(folder)


class BuildEigenfacesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        _write_folder(Path(self.tmp.name))
        self.faces = load_face_folder(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_shapes(self) -> None:
        model = build_eigenfaces(self.faces, 5)
        pixels = FACE_SHAPE[0] * FACE_SHAPE[1]
        self.assertEqual(model.mean.shape, (pixels,))
        self.assertEqual(model.eigenfaces.shape, (5, pixels))
        self.assertEqual(model.coefficients.shape, (8, 5))

    def test_eigenfaces_are_orthonormal(self) -> None:
        model = build_eigenfaces(self.faces, 5)
        np.testing.assert_allclose(model.eigenfaces @ model.eigenfaces.T, np.eye(5), atol=1e-6)

    def test_coefficients_follow_the_formula(self) -> None:
        model = build_eigenfaces(self.faces, 5)
        first = self.faces.images[0].ravel()
        expected = model.eigenfaces @ (first - model.mean)  # w = U_K^T (x - mu)
        np.testing.assert_allclose(model.coefficients[0], expected, atol=1e-6)
        np.testing.assert_allclose(project(first[None], model.mean, model.eigenfaces)[0], expected)

    def test_rejects_too_many_components(self) -> None:
        with self.assertRaises(ValueError):
            build_eigenfaces(self.faces, 9)  # only 8 faces

    def test_save_and_load_round_trip(self) -> None:
        model = build_eigenfaces(self.faces, 3)
        path = model.save(Path(self.tmp.name) / "model.npz")
        loaded = EigenfaceModel.load(path)
        np.testing.assert_allclose(loaded.coefficients, model.coefficients)
        self.assertEqual(loaded.image_shape, FACE_SHAPE)
        self.assertEqual(list(loaded.labels), list(model.labels))

    def test_gallery_has_mean_plus_k_faces(self) -> None:
        figure = plot_eigenfaces(build_eigenfaces(self.faces, 5), k=3)
        self.assertEqual(sum(1 for axis in figure.axes if axis.images), 4)


class EigenfaceBuilderToolTests(unittest.TestCase):
    @patch("forensics_app.tools.eigenfaces._show_gallery")
    @patch("forensics_app.tools.eigenfaces._ask_options", return_value=(4, 3))
    def test_builds_and_saves_model(self, _ask_options, show_gallery) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_folder(root / "train")
            model_file = root / "model.npz"
            with patch("forensics_app.tools.eigenfaces._ask_train_folder", return_value=root / "train"), \
                    patch("forensics_app.tools.eigenfaces.MODEL_FILE", model_file):
                result = EigenfaceBuilderTool().run(MagicMock(), ImageDocument())
            self.assertTrue(model_file.exists())

        self.assertIsNone(result.image)
        self.assertEqual(result.details["Eigenfaces (K)"], "4")
        self.assertEqual(result.details["Coefficients shape"], "(8, 4)")
        show_gallery.assert_called_once()

    @patch("forensics_app.tools.eigenfaces._ask_train_folder", return_value=None)
    def test_cancel_returns_none(self, _ask_train_folder) -> None:
        self.assertIsNone(EigenfaceBuilderTool().run(MagicMock(), ImageDocument()))


class RecognitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        _write_folder(Path(self.tmp.name))
        self.model = build_eigenfaces(load_face_folder(self.tmp.name), 5)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_prepare_face_converts_and_resizes(self) -> None:
        image = Image.new("RGB", (40, 30), (255, 255, 255))
        face = prepare_face(image, FACE_SHAPE)
        self.assertEqual(face.shape, (FACE_SHAPE[0] * FACE_SHAPE[1],))
        np.testing.assert_allclose(face, 1.0)

    def test_training_face_finds_itself(self) -> None:
        with Image.open(self.model.paths[5]) as image:
            face = prepare_face(image, FACE_SHAPE)
        matches = find_closest(self.model, project(face[None], self.model.mean, self.model.eigenfaces)[0])
        self.assertEqual(matches[0].index, 5)
        self.assertEqual(matches[0].label, "Alan Turing")
        self.assertAlmostEqual(matches[0].distance, 0.0, places=6)

    def test_matches_are_sorted_and_k_is_limited(self) -> None:
        matches = find_closest(self.model, self.model.coefficients[0], k=20)  # only 8 faces
        distances = [match.distance for match in matches]
        self.assertEqual(len(matches), 8)
        self.assertEqual(distances, sorted(distances))

    def test_train_face_is_rebuilt_when_the_file_is_missing(self) -> None:
        model = EigenfaceModel(**{**vars(self.model), "paths": np.array(["missing.png"] * 8)})
        expected = (model.mean + model.coefficients[2] @ model.eigenfaces).reshape(FACE_SHAPE)
        np.testing.assert_allclose(train_face(model, 2), expected)

    def test_plot_has_test_face_plus_k_matches(self) -> None:
        matches = find_closest(self.model, self.model.coefficients[0], k=3)
        figure = plot_matches(self.model.mean, "Ada Lovelace", matches, self.model)
        self.assertEqual(sum(1 for axis in figure.axes if axis.images), 4)


class FaceRecognitionToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        _write_folder(root / "train")
        self.model_file = build_eigenfaces(load_face_folder(root / "train"), 5).save(root / "model.npz")
        self.document = ImageDocument()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self) -> ToolResult | None:
        with patch("forensics_app.tools.eigenfaces.MODEL_FILE", self.model_file):
            return FaceRecognitionTool().run(MagicMock(), self.document)

    @patch("forensics_app.tools.eigenfaces._show_matches")
    @patch("forensics_app.tools.eigenfaces._ask_neighbours", return_value=2)
    def test_recognises_a_face_of_a_known_person(self, _ask_neighbours, show_matches) -> None:
        self.document.load(Path(self.tmp.name) / "train" / "Ada_Lovelace" / "0001.png")
        result = self._run()

        self.assertIsNone(result.image)
        self.assertEqual(result.details["Prediction"], "Ada Lovelace")
        self.assertEqual(result.details["True name"], "Ada Lovelace")
        self.assertEqual(result.details["Correct"], "yes")
        self.assertIn("Match 2", result.details)
        self.assertNotIn("Match 3", result.details)
        self.assertIn("Note", result.details)  # it is a training image
        show_matches.assert_called_once()

    @patch("forensics_app.tools.eigenfaces._show_matches")
    @patch("forensics_app.tools.eigenfaces._ask_neighbours", return_value=3)
    def test_unknown_folder_has_no_true_name(self, _ask_neighbours, _show_matches) -> None:
        path = Path(self.tmp.name) / "holiday" / "photo.png"
        path.parent.mkdir()
        Image.new("RGB", (40, 30), (90, 120, 150)).save(path)
        self.document.load(path)
        result = self._run()

        self.assertEqual(result.details["True name"], "Unknown")
        self.assertEqual(result.details["Correct"], "-")
        self.assertEqual(result.details["Input size (W x H)"], "40 x 30")

    @patch("forensics_app.tools.eigenfaces._ask_neighbours", return_value=None)
    def test_cancel_returns_none(self, _ask_neighbours) -> None:
        self.document.load(Path(self.tmp.name) / "train" / "Ada_Lovelace" / "0001.png")
        self.assertIsNone(self._run())

    def test_missing_model_is_reported(self) -> None:
        with patch("forensics_app.tools.eigenfaces.MODEL_FILE", Path(self.tmp.name) / "none.npz"):
            with self.assertRaises(ValueError):
                FaceRecognitionTool().run(MagicMock(), self.document)


if __name__ == "__main__":
    unittest.main()
