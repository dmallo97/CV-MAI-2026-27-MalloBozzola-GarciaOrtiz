from __future__ import annotations

from collections.abc import Sequence
import tkinter as tk
from tkinter import simpledialog

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .channel_split import CHANNELS


def _parse_order(raw: str) -> list[str]:
    cleaned = raw.strip().upper().replace(" ", "").replace(",", "")
    if not cleaned:
        raise ValueError("Enter a channel order (e.g. GRB) or two channels to swap (e.g. RG).")
    for char in cleaned:
        if char not in CHANNELS:
            raise ValueError(f"Unknown channel '{char}'. Use R, G, or B.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Each channel may appear only once.")

    if len(cleaned) == 3:
        return list(cleaned)
    if len(cleaned) == 2:
        first, second = cleaned
        swapped = {first: second, second: first}
        return [swapped.get(name, name) for name in CHANNELS]
    raise ValueError("Use three letters for a new order (e.g. GRB) or two to swap (e.g. RG).")


def swap_channels(image: Image.Image, order: Sequence[str]) -> Image.Image:
    selection = list(order)
    if sorted(selection) != sorted(CHANNELS):
        raise ValueError("Channel order must use each of R, G, and B exactly once.")

    rgb = image if image.mode == "RGB" else image.convert("RGB")
    r, g, b = rgb.split()
    lookup = {"R": r, "G": g, "B": b}
    return Image.merge("RGB", tuple(lookup[name] for name in selection))


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Swap channels (RGB)"
    category = "Filtering"
    description = "Reorder or swap the RGB channels of the working image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        raw = simpledialog.askstring(
            "Swap channels",
            "New channel order (e.g. GRB, BGR) or two channels to swap (e.g. RG):",
            parent=parent,
            initialvalue="GRB",
        )
        if raw is None:
            return None

        try:
            order = _parse_order(raw)
        except ValueError as error:
            return ToolResult(
                message=f"Channel swap cancelled: {error}",
                details={"Input": raw, "Error": str(error)},
            )

        assert document.current is not None
        output = swap_channels(document.current, order)
        mapping = ", ".join(f"{target}←{source}" for target, source in zip(CHANNELS, order))
        return ToolResult(
            image=output,
            message=f"Swapped channels: {mapping}.",
            details={
                "Original mode": document.current.mode,
                "Channel mapping": mapping,
                "Output mode": output.mode,
            },
        )
