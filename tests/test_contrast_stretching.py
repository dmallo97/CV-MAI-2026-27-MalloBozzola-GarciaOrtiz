import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.contrast_stretching import ContrastStretchingTool, stretch_contrast


def _low_contrast_pixels() -> np.ndarray:
    return np.random.default_rng(0).integers(100, 151, (50, 50, 3)).astype(np.uint8)


class StretchContrastTests(unittest.TestCase):
    def _stretch(self, pixels: np.ndarray, low: float = 10.0, high: float = 90.0) -> np.ndarray:
        output, _ = stretch_contrast(Image.fromarray(pixels), low, high)
        return np.asarray(output)

    def test_stretches_low_contrast_image_to_full_range(self) -> None:
        output = self._stretch(_low_contrast_pixels())
        self.assertEqual(output.min(), 0)
        self.assertEqual(output.max(), 255)

    def test_outlier_pixels_do_not_cancel_the_stretch(self) -> None:
        pixels = _low_contrast_pixels()
        pixels[0, 0] = 0
        pixels[0, 1] = 255
        output = self._stretch(pixels)
        self.assertGreater(output[1:].std(), pixels[1:].std() * 2)

    def test_full_percentile_range_maps_min_and_max_exactly(self) -> None:
        pixels = _low_contrast_pixels()
        output = self._stretch(pixels, 0, 100)
        for channel in range(3):
            values = pixels[:, :, channel].astype(np.float64)
            expected = np.rint((values - values.min()) * 255.0 / (values.max() - values.min()))
            # float32 vs float64 can round .5 cases differently.
            np.testing.assert_allclose(output[:, :, channel], expected, atol=1)

    def test_narrower_percentiles_saturate_more_pixels(self) -> None:
        pixels = _low_contrast_pixels()
        narrow, wide = self._stretch(pixels, 25, 75), self._stretch(pixels, 5, 95)
        self.assertGreater(np.isin(narrow, (0, 255)).sum(), np.isin(wide, (0, 255)).sum())

    def test_reports_input_range_per_channel(self) -> None:
        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        pixels[0, 0] = (10, 20, 30)
        pixels[1, 1] = (110, 120, 130)
        _, ranges = stretch_contrast(Image.fromarray(pixels), 0, 100)
        self.assertEqual(ranges["Red input range"], "0–110")
        self.assertEqual(ranges["Blue input range"], "0–130")

    def test_uniform_channel_is_left_unchanged(self) -> None:
        pixels = np.full((4, 4, 3), 80, dtype=np.uint8)
        np.testing.assert_array_equal(self._stretch(pixels), pixels)

    def test_rejects_invalid_percentiles(self) -> None:
        image = Image.fromarray(_low_contrast_pixels())
        for low, high in ((50, 50), (90, 10), (-1, 90), (10, 101)):
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                stretch_contrast(image, low, high)


@patch("matplotlib.pyplot.show")
class ContrastStretchingToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = Image.fromarray(_low_contrast_pixels())

    @patch("forensics_app.tools.contrast_stretching._ask_options", return_value=(5.0, 95.0))
    def test_uses_selected_percentiles(self, _ask_options, show_plot) -> None:
        result = ContrastStretchingTool().run(None, self.document)
        expected, _ = stretch_contrast(self.document.current, 5.0, 95.0)
        np.testing.assert_array_equal(np.asarray(result.image), np.asarray(expected))
        self.assertEqual(result.details["Low percentile"], "5")
        self.assertEqual(result.details["High percentile"], "95")

    @patch("forensics_app.tools.contrast_stretching._ask_options", return_value=None)
    def test_cancel_returns_none(self, _ask_options, show_plot) -> None:
        self.assertIsNone(ContrastStretchingTool().run(None, self.document))
        show_plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
