import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.convolution import (
    GaussianConvolutionTool,
    gaussian_convolve,
    gaussian_kernel_1d,
    gaussian_kernel_2d,
)


def _impulse(mode: str = "L") -> Image.Image:
    image = Image.new(mode, (9, 9), 0)
    image.putpixel((4, 4), 255 if mode == "L" else (255, 255, 255))
    return image


class KernelTests(unittest.TestCase):
    def test_kernels_are_normalised_and_symmetric(self) -> None:
        kernel = gaussian_kernel_1d(1.0)
        self.assertEqual(len(kernel), 7)
        self.assertAlmostEqual(kernel.sum(), 1.0)
        np.testing.assert_allclose(kernel, kernel[::-1])
        self.assertAlmostEqual(gaussian_kernel_2d(1.0).sum(), 1.0)

    def test_rejects_non_positive_sigma(self) -> None:
        with self.assertRaises(ValueError):
            gaussian_kernel_1d(0)


class GaussianConvolveTests(unittest.TestCase):
    def test_horizontal_spreads_only_along_rows(self) -> None:
        result = np.asarray(gaussian_convolve(_impulse(), 1.0, "horizontal"))
        self.assertGreater(result[4, 3], 0)
        self.assertEqual(result[3, 4], 0)

    def test_vertical_spreads_only_along_columns(self) -> None:
        result = np.asarray(gaussian_convolve(_impulse(), 1.0, "vertical"))
        self.assertGreater(result[3, 4], 0)
        self.assertEqual(result[4, 3], 0)

    def test_normal_spreads_in_both_directions(self) -> None:
        result = np.asarray(gaussian_convolve(_impulse(), 1.0, "normal"))
        self.assertGreater(result[3, 4], 0)
        self.assertGreater(result[4, 3], 0)
        self.assertGreater(result[3, 3], 0)

    def test_constant_image_is_unchanged_and_mode_preserved(self) -> None:
        image = Image.new("RGB", (5, 4), (10, 120, 200))
        result = gaussian_convolve(image, 2.0, "normal")
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (5, 4))
        self.assertEqual(result.getpixel((0, 0)), (10, 120, 200))


class GaussianConvolutionToolTests(unittest.TestCase):
    @patch("forensics_app.tools.convolution._ask_options", return_value=("vertical", 1.0))
    def test_returns_image_without_mutating_document(self, _ask_options) -> None:
        document = ImageDocument()
        document.current = _impulse("RGB")
        original = document.current.copy()

        result = GaussianConvolutionTool().run(None, document)

        self.assertEqual(result.details["Direction"], "Vertical convolution")
        self.assertEqual(result.details["Kernel size"], "7x1")
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(list(document.current.getdata()), list(original.getdata()))

    @patch("forensics_app.tools.convolution._ask_options", return_value=None)
    def test_cancel_returns_none(self, _ask_options) -> None:
        document = ImageDocument()
        document.current = _impulse()
        self.assertIsNone(GaussianConvolutionTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
