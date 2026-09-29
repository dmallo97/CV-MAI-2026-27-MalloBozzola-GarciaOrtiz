from __future__ import annotations

from collections.abc import Sequence
import tkinter as tk

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


CHANNELS = ("R", "G", "B")


def _parse_selection(raw: str) -> list[str]:
    ...


def extract_channels(image: Image.Image, channels: Sequence[str]) -> Image.Image:
    ...


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Split channels (RGB)"
    category = "Filtering"
    description = "Show one, several, or all RGB channels of the working image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        return None
