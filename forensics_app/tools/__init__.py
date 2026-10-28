"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .histogram import HistogramVisualizationTool
from .contrast_stretching import ContrastStretchingTool
from .masking import MaskingTool
from .canny import CannyEdgeTool, EdgeVisualizationTool
from .convolution import ConvolutionTool
from .sharp import SharpeningTool
from .cross_correlation import CrossCorrelationTool, TemplateMatchThresholdTool
from .hog import HogPersonDetectionTool, HogTool
from .corners import CornerDetectionTool
from .feature_matching import FeatureMatchingTool, KeypointsTool
from .image_lookup import ImageLookupTool
from .face_detection import FaceDetectionTool, WebcamFaceDetectionTool
from .eigenfaces import EigenfaceBuilderTool



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
            ConvolutionTool(),
            CannyEdgeTool(),
            EdgeVisualizationTool(),
            SharpeningTool(),
            CrossCorrelationTool(),
            TemplateMatchThresholdTool(),
            HogTool(),
            HogPersonDetectionTool(),
            CornerDetectionTool(),
            KeypointsTool(),
            FeatureMatchingTool(),
            ImageLookupTool(),
            FaceDetectionTool(),
            WebcamFaceDetectionTool(),
            EigenfaceBuilderTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
