"""Section with a clickable header that shows or hides its body."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

EXPANDED_ICON = "▾"
COLLAPSED_ICON = "▸"


class CollapsibleSection(ttk.Frame):
    """Place children in ``self.body``; clicking the header toggles it."""

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        *,
        expanded: bool = True,
        style: str = "TFrame",
        header_style: str = "TButton",
    ) -> None:
        super().__init__(parent, style=style)
        self._title = title
        self._count: int | None = None
        self._expanded = expanded
        self.header = ttk.Button(self, style=header_style, command=self.toggle, takefocus=True)
        self.header.pack(fill="x")
        self.body = ttk.Frame(self, style=style)
        if expanded:
            self.body.pack(fill="x")
        self._update_header()

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_count(self, count: int | None) -> None:
        self._count = count
        self._update_header()

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        if expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()
        self._update_header()

    def _update_header(self) -> None:
        icon = EXPANDED_ICON if self._expanded else COLLAPSED_ICON
        suffix = f"  ({self._count})" if self._count is not None and not self._expanded else ""
        self.header.configure(text=f"{icon}  {self._title}{suffix}")
