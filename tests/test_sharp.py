import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.convolution import convolve
from forensics_app.tools.sharp import SharpeningTool, gaussian_size, sharpen


def _step(mode: str = "L") -> Image.Image:
    """Vertical step edge: dark on the left, bright on the right."""
    data = np.full((9, 9), 100, dtype=np.uint8)
    data[:, 5:] = 150
    image = Image.fromarray(data)
    return image if mode == "L" else image.convert(mode)


class SharpenTests(unittest.TestCase):
    def test_increases_contrast_across_edges(self) -> None:
        result = np.asarray(sharpen(_step(), 1.0)).astype(int)
        self.assertLess(result[4, 4], 100)  # dark side gets darker at the border
        self.assertGreater(result[4, 5], 150)  # bright side gets brighter
        self.assertEqual(result[4, 0], 100)  # flat regions are unchanged
        self.assertEqual(result[4, 8], 150)

    def test_equals_original_plus_border_mask(self) -> None:
        image = _step()
        original = np.asarray(image, dtype=int)
        mask = original - np.asarray(convolve(image, "gaussian", gaussian_size(2.0), 2.0), dtype=int)
        expected = np.clip(original + mask, 0, 255)
        np.testing.assert_array_equal(np.asarray(sharpen(image, 2.0)), expected)

    def test_constant_image_is_unchanged_and_mode_preserved(self) -> None:
        image = Image.new("RGB", (6, 5), (10, 120, 200))
        result = sharpen(image, 1.5)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (6, 5))
        self.assertEqual(result.getpixel((0, 0)), (10, 120, 200))

    def test_rejects_non_positive_sigma(self) -> None:
        with self.assertRaises(ValueError):
            sharpen(_step(), 0)


class SharpeningToolTests(unittest.TestCase):
    @patch("forensics_app.tools.sharp.simpledialog.askfloat", return_value=1.0)
    def test_returns_image_without_mutating_document(self, _ask_float) -> None:
        document = ImageDocument()
        document.current = _step("RGB")
        original = document.current.copy()

        result = SharpeningTool().run(None, document)

        self.assertEqual(result.details["Sigma"], "1")
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(list(document.current.getdata()), list(original.getdata()))

    @patch("forensics_app.tools.sharp.simpledialog.askfloat", return_value=None)
    def test_cancel_returns_none(self, _ask_float) -> None:
        document = ImageDocument()
        document.current = _step()
        self.assertIsNone(SharpeningTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
