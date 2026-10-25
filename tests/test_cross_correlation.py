import unittest

import numpy as np

from forensics_app.tools.cross_correlation import (
    count_regions,
    crop_template,
    default_threshold,
    downscale_any,
    normalized_cross_correlation,
    preview_factor,
)
from forensics_app.tools.template_dialogs import angle_from_point, normalize_box, point_on_circle


class CropTemplateTests(unittest.TestCase):
    def test_crop_uses_left_top_right_bottom(self) -> None:
        image = np.arange(20, dtype=float).reshape(4, 5)  # (height, width)
        np.testing.assert_array_equal(crop_template(image, (1, 2, 3, 4)), [[11, 12], [16, 17]])

    def test_empty_or_outside_region_raises(self) -> None:
        image = np.zeros((4, 5))
        with self.assertRaises(ValueError):
            crop_template(image, (2, 2, 2, 3))  # zero width
        with self.assertRaises(ValueError):
            crop_template(image, (0, 0, 6, 2))  # wider than the image

    def test_template_cut_from_image_matches_itself(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.random((40, 50))
        template = crop_template(image, (10, 5, 21, 16))  # 11 x 11, centre at (15, 10)
        correlation = normalized_cross_correlation(image, template)
        row, col = np.unravel_index(np.argmax(correlation), correlation.shape)
        self.assertEqual((col, row), (15, 10))
        self.assertAlmostEqual(float(correlation.max()), 1.0, places=6)


class ThresholdPreviewTests(unittest.TestCase):
    def test_default_threshold_drops_below_a_low_peak(self) -> None:
        self.assertEqual(default_threshold(np.array([[0.2, 0.99]])), 0.9)
        self.assertEqual(default_threshold(np.array([[0.2, 0.6]])), 0.55)

    def test_preview_factor_fits_the_box(self) -> None:
        self.assertEqual(preview_factor((100, 100), (900, 600)), 1)
        self.assertEqual(preview_factor((3000, 4000), (900, 600)), 5)

    def test_downscale_any_keeps_isolated_pixels(self) -> None:
        mask = np.zeros((6, 6), dtype=bool)
        mask[4, 1] = True
        small = downscale_any(mask, 3)
        np.testing.assert_array_equal(small, [[False, False], [True, False]])

    def test_count_regions(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = mask[1, 1] = True  # diagonal neighbours: one region
        mask[4, 4] = True
        self.assertEqual(count_regions(mask), 2)


class DialogGeometryTests(unittest.TestCase):
    def test_normalize_box_orders_and_clamps(self) -> None:
        self.assertEqual(normalize_box(50, 40, -5, 10, 30, 30), (0, 10, 30, 30))

    def test_angle_is_counter_clockwise_on_screen(self) -> None:
        self.assertAlmostEqual(angle_from_point(0, 0, 10, 0), 0.0)
        self.assertAlmostEqual(angle_from_point(0, 0, 0, -10), 90.0)  # up on screen
        self.assertAlmostEqual(angle_from_point(0, 0, 0, 10), 270.0)  # down on screen

    def test_point_on_circle_is_inverse_of_angle(self) -> None:
        x, y = point_on_circle(100, 100, 50, 135)
        self.assertAlmostEqual(angle_from_point(100, 100, x, y), 135.0)


if __name__ == "__main__":
    unittest.main()