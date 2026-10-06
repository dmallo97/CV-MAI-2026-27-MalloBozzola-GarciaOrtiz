import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.masking import MaskingTool, apply_mask, build_mask


def _gray_sample() -> Image.Image:
    return Image.fromarray(np.array([[50, 135], [136, 200]], dtype=np.uint8), mode="L")


def _rgb_sample() -> Image.Image:
    image = Image.new("RGB", (2, 1))
    image.putpixel((0, 0), (10, 20, 30))
    image.putpixel((1, 0), (200, 210, 220))
    return image


class BuildMaskTests(unittest.TestCase):
    def test_threshold_is_strict(self) -> None:
        mask = build_mask(_gray_sample(), 135)
        np.testing.assert_array_equal(mask, [[False, False], [True, True]])

    def test_rgb_uses_grayscale_intensity(self) -> None:
        mask = build_mask(_rgb_sample(), 135)
        np.testing.assert_array_equal(mask, [[False, True]])

    def test_out_of_range_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_mask(_gray_sample(), 256)


class ApplyMaskTests(unittest.TestCase):
    def test_keeps_masked_pixels_and_zeroes_the_rest(self) -> None:
        source = _gray_sample()
        result = apply_mask(source, build_mask(source, 135))
        self.assertEqual(result.mode, "L")
        np.testing.assert_array_equal(np.asarray(result), [[0, 0], [136, 200]])

    def test_invert_keeps_the_complement(self) -> None:
        source = _gray_sample()
        result = apply_mask(source, build_mask(source, 135), invert=True)
        np.testing.assert_array_equal(np.asarray(result), [[50, 135], [0, 0]])

    def test_rgb_mask_applies_to_every_channel(self) -> None:
        source = _rgb_sample()
        result = apply_mask(source, build_mask(source, 135))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((1, 0)), (200, 210, 220))

    def test_input_image_is_not_mutated(self) -> None:
        source = _rgb_sample()
        pixels_before = list(source.getdata())
        apply_mask(source, build_mask(source, 135))
        self.assertEqual(list(source.getdata()), pixels_before)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_mask(_gray_sample(), np.ones((3, 3), dtype=bool))


@patch("matplotlib.pyplot.show")
class MaskingToolTests(unittest.TestCase):
    def _document(self) -> ImageDocument:
        document = ImageDocument()
        document.current = _gray_sample()
        return document

    @patch("tkinter.messagebox.askyesnocancel", return_value=False)
    @patch("tkinter.simpledialog.askinteger", return_value=135)
    def test_run_returns_dark_pixels_without_changing_document(self, _ask_int, _ask_yes, show_plot) -> None:
        document = self._document()
        original = document.current

        result = MaskingTool().run(None, document)

        np.testing.assert_array_equal(np.asarray(result.image), [[50, 135], [0, 0]])
        self.assertIs(document.current, original)
        self.assertEqual(result.details["Pixels kept"], "2 of 4 (50.0%)")
        show_plot.assert_called_once_with(block=False)

    @patch("tkinter.simpledialog.askinteger", return_value=None)
    def test_cancelled_threshold_returns_none(self, _ask_int, show_plot) -> None:
        self.assertIsNone(MaskingTool().run(None, self._document()))
        show_plot.assert_not_called()

    @patch("tkinter.messagebox.askyesnocancel", return_value=None)
    @patch("tkinter.simpledialog.askinteger", return_value=135)
    def test_cancelled_mode_returns_none(self, _ask_int, _ask_yes, show_plot) -> None:
        self.assertIsNone(MaskingTool().run(None, self._document()))
        show_plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
