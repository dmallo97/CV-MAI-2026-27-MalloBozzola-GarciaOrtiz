from __future__ import annotations

from collections.abc import Sequence
import tkinter as tk

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


CHANNELS = ("R", "G", "B")


def _parse_selection(raw: str) -> list[str]:
    cleaned = raw.strip().upper().replace(" ", "").replace(",", "")
    if not cleaned:
        raise ValueError("Choose at least one channel (R, G, or B).")
    seen: list[str] = []
    for char in cleaned:
        if char not in CHANNELS:
            raise ValueError(f"Unknown channel '{char}'. Use R, G, or B.")
        if char in seen:
            raise ValueError(f"Channel '{char}' selected more than once.")
        seen.append(char)
    return seen


def extract_channels(image: Image.Image, channels: Sequence[str]) -> Image.Image:
    selection = list(channels)
    if not selection:
        raise ValueError("Choose at least one channel (R, G, or B).")
    for name in selection:
        if name not in CHANNELS:
            raise ValueError(f"Unknown channel '{name}'. Use R, G, or B.")

    rgb = image if image.mode == "RGB" else image.convert("RGB")
    r, g, b = rgb.split()
    lookup = {"R": r, "G": g, "B": b}

    if len(selection) == 1:
        return lookup[selection[0]].copy()

    zero = Image.new("L", rgb.size, 0)
    bands = tuple(lookup[name] if name in selection else zero for name in CHANNELS)
    return Image.merge("RGB", bands)


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Split channels (RGB)"
    category = "Filtering"
    description = "Show one, several, or all RGB channels of the working image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        return None
