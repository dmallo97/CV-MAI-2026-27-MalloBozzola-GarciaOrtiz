import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.canny import (
    CannyEdgeTool,
    _parse_visualization,
    canny_edges,
    edge_map,
    overlay_edges,
)


def _square(mode: str = "RGB") -> Image.Image:
    image = Image.new(mode, (40, 40), 0)
    fill = 200 if mode == "L" else (200, 200, 200)
    image.paste(fill, (10, 10, 30, 30))
    return image


class CannyEdgesTests(unittest.TestCase):
    def test_detects_edges_of_a_square(self) -> None:
        edges = canny_edges(_square(), 1.0)
        self.assertEqual(edges.dtype, bool)
        self.assertEqual(edges.shape, (40, 40))
        self.assertTrue(edges.any())
        self.assertFalse(edges[20, 20])  # flat interior
        self.assertFalse(edges[0, 0])  # flat background

    def test_constant_image_has_no_edges(self) -> None:
        self.assertFalse(canny_edges(Image.new("L", (20, 20), 128), 1.0).any())

    def test_larger_sigma_detects_fewer_edges(self) -> None:
        rng = np.random.default_rng(0)
        noise = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8))
        self.assertGreater(canny_edges(noise, 1.0).sum(), canny_edges(noise, 4.0).sum())

    def test_rejects_non_positive_sigma(self) -> None:
        with self.assertRaises(ValueError):
            canny_edges(_square(), 0)


class EdgeMapTests(unittest.TestCase):
    def test_edges_are_white_on_black(self) -> None:
        edges = canny_edges(_square(), 1.0)
        output = edge_map(edges)

        self.assertEqual(output.mode, "L")
        self.assertEqual(output.size, (40, 40))
        result = np.asarray(output)
        self.assertTrue((result[edges] == 255).all())
        self.assertTrue((result[~edges] == 0).all())

    def test_parse_visualization(self) -> None:
        self.assertEqual(_parse_visualization(""), "edges")
        self.assertEqual(_parse_visualization(" Overlay "), "overlay")
        self.assertEqual(_parse_visualization("e"), "edges")
        with self.assertRaises(ValueError):
            _parse_visualization("blue")


class OverlayEdgesTests(unittest.TestCase):
    def test_edges_are_red_and_rest_is_grayscale(self) -> None:
        image = _square()
        edges = canny_edges(image, 1.0)
        output = np.asarray(overlay_edges(image, edges))

        self.assertEqual(output.shape, (40, 40, 3))
        np.testing.assert_array_equal(output[edges], np.tile([255, 0, 0], (edges.sum(), 1)))
        rest = output[~edges]
        np.testing.assert_array_equal(rest[:, 0], rest[:, 1])
        np.testing.assert_array_equal(rest[:, 1], rest[:, 2])


class CannyEdgeToolTests(unittest.TestCase):
    @patch("forensics_app.tools.canny.simpledialog.askfloat", return_value=2.0)
    @patch("forensics_app.tools.canny.simpledialog.askstring", return_value="edges")
    def test_edges_returns_edge_map_without_mutating_document(self, _ask_string, _ask_float) -> None:
        document = ImageDocument()
        document.current = _square("L")
        original = document.current.copy()

        result = CannyEdgeTool().run(None, document)

        self.assertEqual(result.details["Visualization"], "edges")
        self.assertEqual(result.details["Sigma"], "2")
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(set(np.unique(np.asarray(result.image))), {0, 255})
        self.assertEqual(list(document.current.getdata()), list(original.getdata()))

    @patch("forensics_app.tools.canny.simpledialog.askfloat", return_value=1.0)
    @patch("forensics_app.tools.canny.simpledialog.askstring", return_value="overlay")
    def test_overlay_returns_rgb_image(self, _ask_string, _ask_float) -> None:
        document = ImageDocument()
        document.current = _square()

        result = CannyEdgeTool().run(None, document)

        self.assertEqual(result.details["Visualization"], "overlay")
        self.assertEqual(result.image.mode, "RGB")

    @patch("forensics_app.tools.canny.simpledialog.askstring", return_value=None)
    def test_cancel_visualization_returns_none(self, _ask_string) -> None:
        document = ImageDocument()
        document.current = _square()
        self.assertIsNone(CannyEdgeTool().run(None, document))

    @patch("forensics_app.tools.canny.simpledialog.askfloat", return_value=None)
    @patch("forensics_app.tools.canny.simpledialog.askstring", return_value="edges")
    def test_cancel_sigma_returns_none(self, _ask_string, _ask_float) -> None:
        document = ImageDocument()
        document.current = _square()
        self.assertIsNone(CannyEdgeTool().run(None, document))


if __name__ == "__main__":
    unittest.main()
