import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.hog import HogTool, compute_hog, hog_to_image


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


if __name__ == "__main__":
    unittest.main()
