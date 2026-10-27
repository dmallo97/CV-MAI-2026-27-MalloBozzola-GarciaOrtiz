import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.corners import (
    CORNER_COLOR,
    CornerDetectionTool,
    corner_response,
    draw_corners,
    find_corners,
)


def _square(mode: str = "RGB") -> Image.Image:
    image = Image.new(mode, (40, 40), 0)
    fill = 200 if mode == "L" else (200, 200, 200)
    image.paste(fill, (10, 10, 30, 30))
    return image


def _document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


def _near_square_corners(corners: np.ndarray, tolerance: int = 2) -> bool:
    expected = np.array([[10, 10], [10, 29], [29, 10], [29, 29]])
    distances = np.abs(corners[:, None, :] - expected[None, :, :]).max(axis=2)
    return bool((distances.min(axis=1) <= tolerance).all())


class CornerResponseTests(unittest.TestCase):
    def test_response_has_image_shape(self) -> None:
        for method in ("fast", "harris"):
            with self.subTest(method=method):
                self.assertEqual(corner_response(_square(), method).shape, (40, 40))

    def test_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            corner_response(_square(), "sobel")


class FindCornersTests(unittest.TestCase):
    def test_harris_finds_the_four_corners_of_a_square(self) -> None:
        corners = find_corners(corner_response(_square(), "harris"), 5, 0.1)
        self.assertEqual(len(corners), 4)
        self.assertTrue(_near_square_corners(corners))

    def test_fast_finds_an_isolated_bright_spot(self) -> None:
        # a 90 degree corner only covers ~11 of the 16 circle pixels, below FAST's default n=12
        image = Image.new("L", (40, 40), 0)
        image.paste(200, (19, 19, 22, 22))
        corners = find_corners(corner_response(image, "fast"), 5, 0.1)
        self.assertEqual(len(corners), 1)
        self.assertTrue((np.abs(corners[0] - 20) <= 1).all())

    def test_constant_image_has_no_corners(self) -> None:
        response = corner_response(Image.new("L", (20, 20), 128), "harris")
        self.assertEqual(len(find_corners(response)), 0)

    def test_higher_threshold_keeps_fewer_corners(self) -> None:
        rng = np.random.default_rng(0)
        noise = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8))
        response = corner_response(noise, "harris")
        self.assertGreater(len(find_corners(response, 1, 0.05)), len(find_corners(response, 1, 0.5)))

    def test_rejects_invalid_parameters(self) -> None:
        response = corner_response(_square(), "fast")
        with self.assertRaises(ValueError):
            find_corners(response, 0, 0.1)
        with self.assertRaises(ValueError):
            find_corners(response, 5, 1.5)


class DrawCornersTests(unittest.TestCase):
    def test_draws_red_circles_on_grayscale(self) -> None:
        output = draw_corners(_square(), np.array([[20, 20]]), radius=3)

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.getpixel((23, 20)), CORNER_COLOR)
        self.assertEqual(output.getpixel((20, 20)), (200, 200, 200))  # circle is not filled

    def test_no_corners_returns_grayscale_copy(self) -> None:
        output = np.asarray(draw_corners(_square(), np.empty((0, 2), dtype=int)))
        np.testing.assert_array_equal(output[..., 0], output[..., 1])
        np.testing.assert_array_equal(output[..., 1], output[..., 2])


class CornerDetectionToolTests(unittest.TestCase):
    @patch("forensics_app.tools.corners._show_response")
    @patch("forensics_app.tools.corners._ask_options", return_value=("harris", 5, 0.1))
    def test_returns_overlay_and_details(self, _ask_options, show_response) -> None:
        document = _document(_square())
        original = document.current.copy()

        result = CornerDetectionTool().run(None, document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.details["Detector"], "HARRIS")
        self.assertEqual(result.details["Corners"], "4")
        self.assertEqual(list(document.current.getdata()), list(original.getdata()))
        show_response.assert_called_once()

    @patch("forensics_app.tools.corners._ask_options", return_value=None)
    def test_cancel_returns_none(self, _ask_options) -> None:
        self.assertIsNone(CornerDetectionTool().run(None, _document(_square())))


if __name__ == "__main__":
    unittest.main()
