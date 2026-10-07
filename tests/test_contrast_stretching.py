import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.contrast_stretching import ContrastStretchingTool


@patch("matplotlib.pyplot.show")
class ContrastStretchingTests(unittest.TestCase):
    def _run(self, pixels: np.ndarray) -> np.ndarray:
        document = ImageDocument()
        document.current = Image.fromarray(pixels)
        return np.asarray(ContrastStretchingTool().run(None, document).image)

    def test_stretches_low_contrast_image_to_full_range(self, show_plot) -> None:
        pixels = np.random.default_rng(0).integers(100, 151, (50, 50, 3)).astype(np.uint8)
        output = self._run(pixels)
        self.assertEqual(output.min(), 0)
        self.assertEqual(output.max(), 255)

    def test_outlier_pixels_do_not_cancel_the_stretch(self, show_plot) -> None:
        pixels = np.random.default_rng(0).integers(100, 151, (50, 50, 3)).astype(np.uint8)
        pixels[0, 0] = 0
        pixels[0, 1] = 255
        output = self._run(pixels)
        self.assertGreater(output[1:].std(), pixels[1:].std() * 2)

    def test_uniform_channel_is_left_unchanged(self, show_plot) -> None:
        pixels = np.full((4, 4, 3), 80, dtype=np.uint8)
        np.testing.assert_array_equal(self._run(pixels), pixels)


if __name__ == "__main__":
    unittest.main()
