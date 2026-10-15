"""Vertically scrollable container for long lists of widgets."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """Place children in ``self.content``; a scrollbar appears when needed."""

    def __init__(self, parent: tk.Misc, style: str = "TFrame") -> None:
        super().__init__(parent, style=style)
        background = ttk.Style(self).lookup(style, "background") or None
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, background=background)
        self._scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)

        self.content = ttk.Frame(self._canvas, style=style)
        self._window = self._canvas.create_window(0, 0, window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_content_resize)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(sequence, self._on_wheel, add="+")

    def _on_content_resize(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._update_scrollbar()

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._update_scrollbar()

    def _overflows(self) -> bool:
        return self.content.winfo_reqheight() > self._canvas.winfo_height()

    def _update_scrollbar(self) -> None:
        if self._overflows():
            if not self._scrollbar.winfo_ismapped():
                self._scrollbar.pack(side="right", fill="y", before=self._canvas)
        else:
            self._scrollbar.pack_forget()
            self._canvas.yview_moveto(0)

    def _contains(self, widget: object) -> bool:
        if not isinstance(widget, tk.Misc):
            return False
        path, own = str(widget), str(self)
        return path == own or path.startswith(own + ".")

    def _on_wheel(self, event: tk.Event) -> None:
        if not self._contains(event.widget) or not self._overflows():
            return
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        self._canvas.yview_scroll(step * 2, "units")
