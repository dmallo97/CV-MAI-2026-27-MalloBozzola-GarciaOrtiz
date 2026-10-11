"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .histogram import HistogramVisualizationTool
from .contrast_stretching import ContrastStretchingTool
from .masking import MaskingTool
from .convolution import GaussianConvolutionTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            HistogramVisualizationTool(),
            ContrastStretchingTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            GaussianConvolutionTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
