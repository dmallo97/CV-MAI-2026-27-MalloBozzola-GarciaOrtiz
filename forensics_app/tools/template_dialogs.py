from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from PIL import Image, ImageTk

Box = tuple[int, int, int, int]  # (left, top, right, bottom)

SOURCE_FILE = "file"
SOURCE_SELECTION = "selection"

CANVAS_BACKGROUND = "#20242b"
DIAL_COLOR = "#c7ccd4"
ACCENT_COLOR = "#ffcc00"

HANDLE_SIZE = 6
MIN_SELECTION_SIDE = 3
MAX_PREVIEW_SIZE = (900, 600)
DIAL_SIZE = 240
DIAL_MARGIN = 26
SNAP_STEP = 15.0
PREVIEW_DELAY_MS = 30
SHIFT_MASK = 0x0001

# ---------- Geometry helpers ----------


def clamp(value: float, low: float, high: float) -> float:
    """Limit `value` to the closed interval [low, high]."""
    return min(max(value, low), high)


def fit_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    """Scale factor that fits (width, height) inside the maximum size,
    never enlarging."""
    return min(max_width / width, max_height / height, 1.0)


def normalize_box(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> Box:
    """Order two opposite corners and clamp them to an image of size (width x height)."""
    left, right = sorted((x0, x1))  # sort so that left <= right and top <= bottom
    top, bottom = sorted((y0, y1))
    return (
        round(clamp(left, 0, width)),
        round(clamp(top, 0, height)),
        round(clamp(right, 0, width)),
        round(clamp(bottom, 0, height)),
    )


def angle_from_point(cx: float, cy: float, x: float, y: float) -> float:
    """Counter-clockwise angle in degrees [0, 360) of point (x, y) around (cx, cy).

    Screen y grows downwards, so it is flipped to follow the math convention.
    """
    return math.degrees(math.atan2(-(y - cy), x - cx)) % 360.0


def point_on_circle(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    """Screen coordinates of the point at `angle` degrees (counter-clockwise) on a circle.

    Inverse of `angle_from_point`: y is flipped because screen y grows downwards.
    """
    radians = math.radians(angle)
    return cx + radius * math.cos(radians), cy - radius * math.sin(radians)


# ---------- Dialogs ----------


class _ModalDialog(tk.Toplevel):
    """Base class for blocking dialogs: `show()` waits until the window closes."""

    def __init__(self, parent: tk.Misc, title: str) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent.winfo_toplevel())
        self.resizable(False, False)
        self.result: Any = None
        self._job: str | None = None
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Escape>", lambda _event: self.cancel())

    def show(self) -> Any:
        self.wait_visibility()
        self.grab_set()  # block clicks on the main window while open
        self.focus_set()
        self.wait_window()  # runs a local event loop until destroy()
        return self.result

    def accept(self, result: Any) -> None:
        self.result = result
        self.destroy()

    def cancel(self) -> None:
        self.result = None
        self.destroy()

    def schedule(self, callback: Callable[[], None]) -> None:
        """Run `callback` soon, dropping any call still pending (debounce)."""
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(PREVIEW_DELAY_MS, self._run_job, callback)

    def _run_job(self, callback: Callable[[], None]) -> None:
        self._job = None
        callback()

    def destroy(self) -> None:
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        super().destroy()


class TemplateSourceDialog(_ModalDialog):
    """Ask whether the template comes from a file or from the current image."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, "Template source")
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Where should the template come from?").pack(anchor="w", pady=(0, 10))
        ttk.Button(
            frame, text="Provide external template...",
            command=lambda: self.accept(SOURCE_FILE),
        ).pack(fill="x", pady=2)
        ttk.Button(
            frame, text="Select from the image...",
            command=lambda: self.accept(SOURCE_SELECTION),
        ).pack(fill="x", pady=2)
        ttk.Button(
            frame, text="Cancel", command=self.cancel,
        ).pack(fill="x", pady=(10, 0))


class RegionSelectorDialog(_ModalDialog):
    """Let the user draw, move or resize a rectangle over the image."""

    def __init__(self, parent: tk.Misc, image: Image.Image) -> None:
        super().__init__(parent, "Select template region")
        self._image_size = image.size
        self._scale = fit_scale(image.width, image.height, *MAX_PREVIEW_SIZE)
        preview_size = (
            max(1, round(image.width * self._scale)),
            max(1, round(image.height * self._scale)),
        )
        preview = image.convert("RGB").resize(preview_size, Image.Resampling.LANCZOS)
        self._canvas_size = preview_size
        self._photo = ImageTk.PhotoImage(preview, master=self)
        self._box: list[float] | None = None  # [x0, y0, x1, y1] in canvas pixels
        self._mode: str | None = None  # "draw", "move" or None
        self._anchor = (0.0, 0.0)
        self._info = tk.StringVar(
            value="Drag to draw. Drag inside to move, drag a corner to resize."
        )

        self._canvas = tk.Canvas(
            self,
            width=preview_size[0],
            height=preview_size[1],
            background=CANVAS_BACKGROUND,
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.pack(padx=8, pady=8)
        self._canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>", self._on_motion)

        bottom = ttk.Frame(self, padding=(8, 0, 8, 8))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self._info).pack(side="left")
        ttk.Button(bottom, text="Cancel", command=self.cancel).pack(side="right")
        self._ok_button = ttk.Button(
            bottom, text="Use selection", command=self._confirm, state="disabled"
        )
        self._ok_button.pack(side="right", padx=(0, 6))
        self.bind("<Return>", lambda _event: self._confirm())

    # --- coordinate helpers ---
    def _clamp_point(self, x: float, y: float) -> tuple[float, float]:
        return clamp(x, 0, self._canvas_size[0]), clamp(y, 0, self._canvas_size[1])

    def _image_box(self) -> Box | None:
        """Selection converted from canvas pixels to image pixels."""
        if self._box is None:
            return None
        x0, y0, x1, y1 = (value / self._scale for value in self._box)
        return normalize_box(x0, y0, x1, y1, *self._image_size)

    def _hit_test(self, x: float, y: float) -> str | None:
        """Return the corner name ("nw", "ne", "sw", "se"), "move", or None."""
        if self._box is None:
            return None
        x0, y0, x1, y1 = self._box
        corners = {"nw": (x0, y0), "ne": (x1, y0), "sw": (x0, y1), "se": (x1, y1)}
        for name, (cx, cy) in corners.items():
            if abs(x - cx) <= HANDLE_SIZE and abs(y - cy) <= HANDLE_SIZE:
                return name
        if x0 <= x <= x1 and y0 <= y <= y1:
            return "move"
        return None

    # --- mouse events ---
    def _on_motion(self, event: tk.Event) -> None:
        hit = self._hit_test(event.x, event.y)
        cursor = "crosshair" if hit is None else ("fleur" if hit == "move" else "sizing")
        self._canvas.configure(cursor=cursor)

    def _on_press(self, event: tk.Event) -> None:
        x, y = self._clamp_point(event.x, event.y)
        hit = self._hit_test(x, y)
        if hit == "move":
            self._mode = "move"
            self._anchor = (x, y)
        elif hit is not None and self._box is not None:
            # Resizing = drawing again while the opposite corner stays fixed.
            x0, y0, x1, y1 = self._box
            self._anchor = (x1 if "w" in hit else x0, y1 if "n" in hit else y0)
            self._mode = "draw"
        else:
            self._anchor = (x, y)
            self._box = [x, y, x, y]
            self._mode = "draw"
        self._redraw()

    def _on_drag(self, event: tk.Event) -> None:
        if self._mode is None or self._box is None:
            return
        x, y = self._clamp_point(event.x, event.y)
        if self._mode == "draw":
            ax, ay = self._anchor
            self._box = [min(ax, x), min(ay, y), max(ax, x), max(ay, y)]
        else:
            x0, y0, x1, y1 = self._box
            # Limit the displacement so the rectangle never leaves the canvas.
            dx = clamp(x - self._anchor[0], -x0, self._canvas_size[0] - x1)
            dy = clamp(y - self._anchor[1], -y0, self._canvas_size[1] - y1)
            self._box = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
            self._anchor = (self._anchor[0] + dx, self._anchor[1] + dy)
        self._redraw()

    def _on_release(self, _event: tk.Event) -> None:
        self._mode = None
        self._redraw()

    # --- drawing and result ---
    def _redraw(self) -> None:
        self._canvas.delete("selection")
        box = self._image_box()
        if self._box is None or box is None:
            return
        x0, y0, x1, y1 = self._box
        self._canvas.create_rectangle(
            x0, y0, x1, y1, outline=ACCENT_COLOR, width=2, dash=(6, 3), tags="selection"
        )
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            self._canvas.create_rectangle(
                cx - HANDLE_SIZE, cy - HANDLE_SIZE, cx + HANDLE_SIZE, cy + HANDLE_SIZE,
                fill=ACCENT_COLOR, outline="black", tags="selection",
            )
        left, top, right, bottom = box
        width, height = right - left, bottom - top
        self._info.set(f"Selection: {width} x {height} px at ({left}, {top})")
        valid = min(width, height) >= MIN_SELECTION_SIDE
        self._ok_button.configure(state="normal" if valid else "disabled")

    def _confirm(self) -> None:
        box = self._image_box()
        if box is None or min(box[2] - box[0], box[3] - box[1]) < MIN_SELECTION_SIDE:
            self.bell()
            return
        self.accept(box)


class AngleDialog(_ModalDialog):
    """Choose a rotation angle on a circular dial with a live template preview."""

    def __init__(
        self,
        parent: tk.Misc,
        render_preview: Callable[[float], Image.Image],
        initial_angle: float = 0.0,
    ) -> None:
        super().__init__(parent, "Template rotation")
        self._render_preview = render_preview
        self._center = DIAL_SIZE / 2
        self._radius = DIAL_SIZE / 2 - DIAL_MARGIN
        self._photo: ImageTk.PhotoImage | None = None
        self._angle = tk.DoubleVar(value=initial_angle % 360.0)

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        self._dial = tk.Canvas(
            body, width=DIAL_SIZE, height=DIAL_SIZE,
            background=CANVAS_BACKGROUND,
            highlightthickness=0, cursor="hand2",
        )
        self._dial.grid(row=0, column=0, padx=(0, 12))
        self._preview = tk.Canvas(
            body, width=DIAL_SIZE, height=DIAL_SIZE,
            background=CANVAS_BACKGROUND,
            highlightthickness=0,
        )
        self._preview.grid(row=0, column=1)

        controls = ttk.Frame(body, padding=(0, 10, 0, 0))
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Label(controls, text="Angle (degrees, counter-clockwise): ").pack(side="left")
        ttk.Spinbox(
            controls, from_=0.0, to=360.0, increment=1.0, width=7, textvariable=self._angle
        ).pack(side="left", padx=6)
        ttk.Label(controls, text="Shift + drag snaps to 15°").pack(side="left")
        ttk.Button(controls, text="Cancel", command=self.cancel).pack(side="right")
        ttk.Button(controls, text="OK", command=self._confirm).pack(side="right", padx=(0, 6))
        self.bind("<Return>", lambda _event: self._confirm())

        self._draw_dial_background()
        self._dial.bind("<ButtonPress-1>", self._on_mouse)
        self._dial.bind("<B1-Motion>", self._on_mouse)
        self._angle.trace_add("write", self._on_angle_changed)
        self._on_angle_changed()

    def _current_angle(self) -> float | None:
        try:
            return float(self._angle.get()) % 360.0
        except (tk.TclError, ValueError):
            return None  # the spinbox contains text that is not a number

    def _draw_dial_background(self) -> None:
        c, r = self._center, self._radius
        self._dial.create_oval(c - r, c - r, c + r, c + r, outline=DIAL_COLOR, width=2)
        for tick in range(0, 360, 30):
            inner = r - (10 if tick % 90 == 0 else 5)
            x0, y0 = point_on_circle(c, c, inner, tick)
            x1, y1 = point_on_circle(c, c, r, tick)
            self._dial.create_line(x0, y0, x1, y1, fill=DIAL_COLOR, width=2)
        for tick in (0, 90, 180, 270):
            x, y = point_on_circle(c, c, r + 13, tick)
            self._dial.create_text(
                x, y, text=f"{tick}°", fill=DIAL_COLOR, font=("TkDefaultFont", 8)
            )

    def _on_mouse(self, event: tk.Event) -> None:
        angle = angle_from_point(self._center, self._center, event.x, event.y)
        if event.state & SHIFT_MASK:
            angle = round(angle / SNAP_STEP) * SNAP_STEP % 360.0
        self._angle.set(round(angle, 1))

    def _on_angle_changed(self, *_args: object) -> None:
        angle = self._current_angle()
        if angle is None:
            return
        self._draw_needle(angle)
        self.schedule(self._update_preview)

    def _draw_needle(self, angle: float) -> None:
        c, r = self._center, self._radius
        self._dial.delete("needle")
        self._dial.create_arc(
            c - r / 2, c - r / 2, c + r / 2, c + r / 2,
            start=0, extent=angle, outline=ACCENT_COLOR,
            style="arc", width=2, tags="needle",
        )
        x, y = point_on_circle(c, c, r, angle)
        self._dial.create_line(c, c, x, y, fill=ACCENT_COLOR, width=3, tags="needle")
        self._dial.create_oval(
            x - 7, y - 7, x + 7, y + 7, fill=ACCENT_COLOR, outline="", tags="needle"
        )
        self._dial.create_text(c, c + 18, text=f"{angle:.1f}°", fill="white", tags="needle")

    def _update_preview(self) -> None:
        angle = self._current_angle()
        if angle is None:
            return
        # Copy so thumbnail() (in-place) never modifies an image owned by the caller.
        preview = self._render_preview(angle).copy()
        preview.thumbnail((DIAL_SIZE - 16, DIAL_SIZE - 16), Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(preview, master=self)
        self._preview.delete("all")
        self._preview.create_image(
            DIAL_SIZE / 2, DIAL_SIZE / 2, image=self._photo, anchor="center"
        )

    def _confirm(self) -> None:
        angle = self._current_angle()
        if angle is None:
            self.bell()
            return
        self.accept(angle)


class ThresholdDialog(_ModalDialog):
    """Slider that re-renders a preview every time the threshold changes."""

    def __init__(
        self,
        parent: tk.Misc,
        render_preview: Callable[[float], tuple[Image.Image, str]],
        initial: float,
        low: float = -1.0,
        high: float = 1.0,
    ) -> None:
        super().__init__(parent, "Thresholded matches")
        self._render_preview = render_preview
        self._low, self._high = low, high  # set before any callback can read them
        self._value = tk.DoubleVar(value=initial)
        self._value_text = tk.StringVar()
        self._info = tk.StringVar()
        self._photo: ImageTk.PhotoImage | None = None

        self._image_label = tk.Label(self, background=CANVAS_BACKGROUND, padx=8, pady=8)
        self._image_label.pack(padx=8, pady=8)

        controls = ttk.Frame(self, padding=(8, 0, 8, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="Threshold:").pack(side="left")
        self._scale = ttk.Scale(
            controls, from_=low, to=high, orient="horizontal", length=360,
            variable=self._value, command=self._on_slider,
        )
        self._scale.pack(side="left", padx=6)
        ttk.Label(controls, textvariable=self._value_text, width=6).pack(side="left")
        ttk.Label(controls, textvariable=self._info).pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="Cancel", command=self.cancel).pack(side="right")
        ttk.Button(controls, text="Apply", command=self._confirm).pack(side="right", padx=(0, 6))

        self.bind("<Return>", lambda _event: self._confirm())
        self._on_slider()

    def _threshold(self) -> float:
        return round(clamp(self._value.get(), self._low, self._high), 2)

    def _on_slider(self, *_args: object) -> None:
        self._value_text.set(f"{self._threshold():.2f}")
        self.schedule(self._update_preview)

    def _update_preview(self) -> None:
        preview, info = self._render_preview(self._threshold())
        self._photo = ImageTk.PhotoImage(preview, master=self)
        self._image_label.configure(image=self._photo)
        self._info.set(info)

    def _confirm(self) -> None:
        self.accept(self._threshold())


# ---------- Functions used by the tools ----------


def ask_template_source(parent: tk.Misc) -> str | None:
    """Return SOURCE_FILE, SOURCE_SELECTION, or None if cancelled."""
    return TemplateSourceDialog(parent).show()


def ask_region(parent: tk.Misc, image: Image.Image) -> Box | None:
    """Return the selected (left, top, right, bottom) box in image pixels, or None."""
    return RegionSelectorDialog(parent, image).show()


def ask_angle(parent: tk.Misc, render_preview: Callable[[float], Image.Image]) -> float | None:
    """Return the angle in degrees [0, 360), or None if cancelled."""
    return AngleDialog(parent, render_preview).show()


def ask_threshold(
    parent: tk.Misc,
    render_preview: Callable[[float], tuple[Image.Image, str]],
    initial: float,
) -> float | None:
    """Return the chosen threshold in [-1, 1], or None if cancelled."""
    return ThresholdDialog(parent, render_preview, initial).show()
