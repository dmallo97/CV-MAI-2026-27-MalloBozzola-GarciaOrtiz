import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.canny import (
    ANALYSIS_KEY,
    NO_EDGES_MESSAGE,
    CannyEdgeTool,
    EdgeVisualizationTool,
    canny_edges,
    edge_map,
    overlay_edges,
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


class EdgeImageTests(unittest.TestCase):
    def test_edge_map_is_white_on_black(self) -> None:
        edges = canny_edges(_square(), 1.0)
        output = edge_map(edges)

        self.assertEqual(output.mode, "L")
        self.assertEqual(output.size, (40, 40))
        result = np.asarray(output)
        self.assertTrue((result[edges] == 255).all())
        self.assertTrue((result[~edges] == 0).all())

    def test_overlay_edges_are_red_and_rest_is_grayscale(self) -> None:
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
    def test_returns_edge_map_and_stores_edges(self, _ask_float) -> None:
        document = _document(_square())
        original = document.current.copy()

        result = CannyEdgeTool().run(None, document)

        self.assertEqual(result.details["Sigma"], "2")
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(set(np.unique(np.asarray(result.image))), {0, 255})
        self.assertEqual(list(document.current.getdata()), list(original.getdata()))
        stored = document.analyses[ANALYSIS_KEY]
        self.assertEqual(stored.sigma, 2.0)
        np.testing.assert_array_equal(stored.edges, np.asarray(result.image) == 255)

    @patch("forensics_app.tools.canny.simpledialog.askfloat", return_value=None)
    def test_cancel_returns_none_and_stores_nothing(self, _ask_float) -> None:
        document = _document(_square())
        self.assertIsNone(CannyEdgeTool().run(None, document))
        self.assertNotIn(ANALYSIS_KEY, document.analyses)


class EdgeVisualizationToolTests(unittest.TestCase):
    def test_unavailable_before_edge_detection(self) -> None:
        document = _document(_square())
        tool = EdgeVisualizationTool()

        self.assertFalse(tool.is_available(document))
        self.assertEqual(tool.unavailable_message, NO_EDGES_MESSAGE)
        result = tool.run(None, document)
        self.assertIsNone(result.image)
        self.assertEqual(result.message, NO_EDGES_MESSAGE)

    @patch("forensics_app.tools.canny.simpledialog.askfloat", return_value=1.0)
    def test_overlays_edges_on_image_used_for_detection(self, _ask_float) -> None:
        document = _document(_square())
        detection = CannyEdgeTool().run(None, document)
        document.apply(detection.image)  # as the main window would
        tool = EdgeVisualizationTool()

        self.assertTrue(tool.is_available(document))
        result = tool.run(None, document)

        self.assertEqual(result.image.mode, "RGB")
        edges = document.analyses[ANALYSIS_KEY].edges
        expected = overlay_edges(_square(), edges)
        self.assertEqual(list(result.image.getdata()), list(expected.getdata()))


if __name__ == "__main__":
    unittest.main()
