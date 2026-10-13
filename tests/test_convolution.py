import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.convolution import (
    FILTERS,
    ConvolutionTool,
    _parse_filter,
    convolve,
)


def _impulse(mode: str = "L") -> Image.Image:
    image = Image.new(mode, (9, 9), 0)
    image.putpixel((4, 4), 255 if mode == "L" else (255, 255, 255))
    return image


def _vertical_edge() -> Image.Image:
    data = np.zeros((9, 9), dtype=np.uint8)
    data[:, 5:] = 255
    return Image.fromarray(data)


class GaussianTests(unittest.TestCase):
    def test_spreads_in_both_directions(self) -> None:
        result = np.asarray(convolve(_impulse(), "gaussian", 5, 1.0))
        self.assertGreater(result[3, 4], 0)
        self.assertGreater(result[4, 3], 0)
        self.assertGreater(result[3, 3], 0)

    def test_kernel_size_limits_spread(self) -> None:
        result = np.asarray(convolve(_impulse(), "gaussian", 3, 5.0))
        self.assertGreater(result[3, 3], 0)
        self.assertEqual(result[2, 4], 0)
        self.assertEqual(result[4, 2], 0)


class MedianTests(unittest.TestCase):
    def test_removes_isolated_impulse(self) -> None:
        result = np.asarray(convolve(_impulse(), "median", 3))
        self.assertEqual(result.max(), 0)


class EdgeFilterTests(unittest.TestCase):
    def test_edge_filters_detect_vertical_edge(self) -> None:
        for name in ("sobel", "prewitt", "scharr"):
            with self.subTest(filter=name):
                result = np.asarray(convolve(_vertical_edge(), name))
                self.assertEqual(result[4, 4], 255)
                self.assertEqual(result[4, 0], 0)


class ConvolveTests(unittest.TestCase):
    def test_constant_image_mode_preserved(self) -> None:
        image = Image.new("RGB", (5, 4), (10, 120, 200))
        for name in ("gaussian", "median"):
            with self.subTest(filter=name):
                result = convolve(image, name, 5, 2.0)
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.size, (5, 4))
                self.assertEqual(result.getpixel((0, 0)), (10, 120, 200))
        self.assertEqual(convolve(image, "sobel").getpixel((0, 0)), (0, 0, 0))

    def test_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            convolve(_impulse(), "gaussian", 4)
        with self.assertRaises(ValueError):
            convolve(_impulse(), "gaussian", 3, sigma=0)
        with self.assertRaises(ValueError):
            convolve(_impulse(), "laplace")

    def test_parse_filter(self) -> None:
        self.assertEqual(_parse_filter(""), "gaussian")
        self.assertEqual(_parse_filter("Med"), "median")
        self.assertEqual(_parse_filter("sc"), "scharr")
        with self.assertRaises(ValueError):
            _parse_filter("s")  # ambiguous: sobel or scharr
        self.assertEqual(set(FILTERS), {"gaussian", "median", "sobel", "prewitt", "scharr"})


class ConvolutionToolTests(unittest.TestCase):
    @patch("forensics_app.tools.convolution.simpledialog.askfloat", return_value=1.0)
    @patch("forensics_app.tools.convolution.simpledialog.askinteger", return_value=5)
    @patch("forensics_app.tools.convolution.simpledialog.askstring", return_value="gaussian")
    def test_returns_image_without_mutating_document(self, _ask_string, _ask_int, _ask_float) -> None:
        document = ImageDocument()
        document.current = _impulse("RGB")
        original = document.current.copy()

        result = ConvolutionTool().run(None, document)

        self.assertEqual(result.details["Filter"], "gaussian")
        self.assertEqual(result.details["Kernel size"], "5x5")
        self.assertEqual(result.details["Sigma"], "1")
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(list(document.current.getdata()), list(original.getdata()))

    @patch("forensics_app.tools.convolution.simpledialog.askinteger")
    @patch("forensics_app.tools.convolution.simpledialog.askstring", return_value="sobel")
    def test_edge_filter_skips_kernel_size_prompt(self, _ask_string, ask_int) -> None:
        document = ImageDocument()
        document.current = _vertical_edge()
        result = ConvolutionTool().run(None, document)
        ask_int.assert_not_called()
        self.assertEqual(result.details["Kernel size"], "3x3")
        self.assertNotIn("Sigma", result.details)

    @patch("forensics_app.tools.convolution.simpledialog.askinteger", return_value=4)
    @patch("forensics_app.tools.convolution.simpledialog.askstring", return_value="median")
    def test_even_kernel_size_is_rejected(self, _ask_string, _ask_int) -> None:
        document = ImageDocument()
        document.current = _impulse()
        result = ConvolutionTool().run(None, document)
        self.assertIsNone(result.image)
        self.assertIn("odd", result.message)

    @patch("forensics_app.tools.convolution.simpledialog.askstring", return_value=None)
    def test_cancel_returns_none(self, _ask_string) -> None:
        document = ImageDocument()
        document.current = _impulse()
        self.assertIsNone(ConvolutionTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
