"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .blur import BlurTool
from .histogram import HistogramVisualizationTool
from .contrast_stretching import ContrastStretchingTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            BlurTool(),
            HistogramVisualizationTool(),
            ContrastStretchingTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
