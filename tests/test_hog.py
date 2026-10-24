import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.hog import (
    EUCLIDEAN_COLOR,
    Detection,
    HogPersonDetectionTool,
    HogTool,
    compute_hog,
    detect_person,
    draw_detections,
    hog_to_image,
    sliding_windows,
)


def _square(mode: str = "RGB") -> Image.Image:
    image = Image.new(mode, (36, 48), 0)
    fill = 200 if mode == "L" else (200,) * len(mode)
    image.paste(fill, (9, 12, 27, 36))
    return image


class ComputeHogTests(unittest.TestCase):
    def test_feature_length_matches_blocks_cells_and_orientations(self) -> None:
        features, hog_image = compute_hog(_square(), 9, 6, 3)
        # 48x36 px -> 8x6 cells -> 6x4 blocks of 3x3 cells
        self.assertEqual(features.size, 6 * 4 * 3 * 3 * 9)
        self.assertEqual(hog_image.shape, (48, 36))

    def test_more_orientations_give_longer_vector(self) -> None:
        two, _ = compute_hog(_square(), 2, 6, 3)
        four, _ = compute_hog(_square(), 4, 6, 3)
        self.assertEqual(four.size, 2 * two.size)

    def test_accepts_grayscale_and_rgba(self) -> None:
        rgb, _ = compute_hog(_square("RGB"))
        for mode in ("L", "RGBA"):
            with self.subTest(mode=mode):
                features, _ = compute_hog(_square(mode))
                self.assertEqual(features.size, rgb.size)

    def test_flat_image_has_no_gradients(self) -> None:
        features, _ = compute_hog(Image.new("RGB", (36, 48), (90, 90, 90)))
        self.assertFalse(features.any())

    def test_rejects_image_smaller_than_a_block(self) -> None:
        with self.assertRaises(ValueError):
            compute_hog(Image.new("RGB", (10, 10)), 9, 6, 3)


class HogToImageTests(unittest.TestCase):
    def test_returns_visible_grayscale_image(self) -> None:
        _, hog_image = compute_hog(_square())
        output = hog_to_image(hog_image)
        self.assertEqual(output.mode, "L")
        self.assertEqual(output.size, (36, 48))
        self.assertEqual(np.asarray(output).max(), 255)


class HogToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = _square()

    @patch("forensics_app.tools.hog._ask_options", return_value=(4, 6, 3))
    def test_uses_selected_parameters(self, _ask_options) -> None:
        result = HogTool().run(None, self.document)
        _, expected = compute_hog(self.document.current, 4, 6, 3)
        np.testing.assert_array_equal(np.asarray(result.image), np.asarray(hog_to_image(expected)))
        self.assertEqual(result.details["Orientations"], "4")
        self.assertEqual(result.details["Cells (w x h)"], "6 x 8")
        self.assertEqual(result.details["Blocks (w x h)"], "4 x 6")
        self.assertEqual(result.details["Feature vector length"], f"{4 * 6 * 9 * 4:,}")

    @patch("forensics_app.tools.hog._ask_options", return_value=None)
    def test_cancel_returns_none(self, _ask_options) -> None:
        self.assertIsNone(HogTool().run(None, self.document))


def _figure() -> Image.Image:
    """Stick-like figure (head + body + legs) on a 48x64 black background."""
    figure = Image.new("RGB", (48, 64), 0)
    figure.paste((220, 220, 220), (18, 4, 30, 16))
    figure.paste((220, 220, 220), (14, 18, 34, 44))
    figure.paste((220, 220, 220), (14, 46, 22, 62))
    figure.paste((220, 220, 220), (26, 46, 34, 62))
    return figure


def _scene(x: int, y: int) -> Image.Image:
    scene = Image.new("RGB", (160, 120), 0)
    scene.paste((120, 120, 120), (100, 10, 150, 30))  # distractor
    scene.paste(_figure(), (x, y))
    return scene


class SlidingWindowTests(unittest.TestCase):
    def test_includes_last_position(self) -> None:
        corners = sliding_windows((10, 6), (4, 4), 3)
        self.assertEqual(corners, [(0, 0), (3, 0), (6, 0)])

    def test_window_same_size_as_image(self) -> None:
        self.assertEqual(sliding_windows((8, 8), (8, 8), 4), [(0, 0)])

    def test_rejects_template_larger_than_image(self) -> None:
        with self.assertRaises(ValueError):
            sliding_windows((10, 10), (12, 4), 2)

    def test_rejects_invalid_step(self) -> None:
        with self.assertRaises(ValueError):
            sliding_windows((10, 10), (4, 4), 0)


class DetectPersonTests(unittest.TestCase):
    def test_finds_the_planted_figure_with_both_measures(self) -> None:
        euclidean, dot_product, windows = detect_person(_scene(40, 32), _figure(), step=4)
        self.assertEqual((euclidean.x, euclidean.y), (40, 32))
        self.assertEqual((dot_product.x, dot_product.y), (40, 32))
        self.assertAlmostEqual(euclidean.score, 0.0, places=6)
        self.assertEqual(windows, len(sliding_windows((160, 120), (48, 64), 4)))

    def test_grayscale_scene_and_template(self) -> None:
        euclidean, _, _ = detect_person(_scene(40, 32).convert("L"), _figure().convert("L"), step=8)
        self.assertEqual((euclidean.x, euclidean.y), (40, 32))


class DrawDetectionsTests(unittest.TestCase):
    def test_draws_box_without_modifying_source(self) -> None:
        source = Image.new("RGB", (40, 40), 0)
        output = draw_detections(source, (10, 20), Detection(5, 5, 0.0), Detection(20, 10, 1.0))
        self.assertEqual(output.getpixel((5, 5)), EUCLIDEAN_COLOR)
        self.assertEqual(output.getpixel((14, 24)), EUCLIDEAN_COLOR)
        self.assertNotEqual(output.getpixel((20, 10)), (0, 0, 0))
        self.assertEqual(source.getpixel((5, 5)), (0, 0, 0))


@patch("forensics_app.tools.hog._show_template_hog")
class HogPersonDetectionToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = _scene(40, 32)

    def _run(self, path: str, options):
        with patch("forensics_app.tools.hog._ask_template_path", return_value=path), patch(
            "forensics_app.tools.hog._ask_detection_options", return_value=options
        ):
            return HogPersonDetectionTool().run(None, self.document)

    def test_reports_detections(self, _show) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "person.png")
            _figure().save(path)
            result = self._run(path, (4, 16, 8, 2))
        self.assertEqual(result.image.size, (160, 120))
        self.assertEqual(result.details["Euclidean best (x, y)"], "(40, 32)")
        self.assertEqual(result.details["Dot product best (x, y)"], "(40, 32)")
        self.assertEqual(result.details["Window size (w x h)"], "48 x 64")
        _show.assert_called_once()

    def test_template_larger_than_image_reports_error(self, _show) -> None:
        self.document.current = Image.new("RGB", (30, 30))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "person.png")
            _figure().save(path)
            result = self._run(path, (4, 16, 8, 2))
        self.assertIsNone(result.image)
        self.assertIn("larger than the image", result.message)

    def test_cancel_template_returns_none(self, _show) -> None:
        self.assertIsNone(self._run("", (4, 16, 8, 2)))

    def test_cancel_options_returns_none(self, _show) -> None:
        self.assertIsNone(self._run("unused.png", None))


if __name__ == "__main__":
    unittest.main()
