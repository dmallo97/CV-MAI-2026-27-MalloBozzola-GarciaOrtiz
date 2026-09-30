import unittest
from unittest.mock import patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.histogram import HistogramVisualizationTool
from forensics_app.tools.registry import ToolRegistry


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])

    @patch("matplotlib.pyplot.show")
    def test_histogram_displays_plot_without_changing_document(self, show_plot) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        original = document.current

        result = HistogramVisualizationTool().run(None, document)

        self.assertIsNone(result.image)
        self.assertIs(document.current, original)
        self.assertEqual(result.details["Pixels"], "12")
        show_plot.assert_called_once_with(block=False)

if __name__ == "__main__":
    unittest.main()
