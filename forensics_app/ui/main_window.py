"""The application's main window and event coordination."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool
from forensics_app.tools.registry import ToolRegistry
from .image_view import ImageView
from .widgets import CollapsibleSection, ScrollableFrame


OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
SAVE_TYPES = [("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("TIFF image", "*.tiff")]


class MainWindow:
    def __init__(self, root: tk.Tk, registry: ToolRegistry) -> None:
        self.root = root
        self.registry = registry
        self.document = ImageDocument()
        self.status = tk.StringVar(value="Ready. Open an image to begin.")
        self._tool_buttons: list[tuple[ForensicsTool, ttk.Button]] = []
        self._tool_menu_entries: list[tuple[ForensicsTool, tk.Menu, int]] = []

        self._configure_window()
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        self._refresh()

    def _configure_window(self) -> None:
        self.root.title("ForensicsApp")
        self.root.geometry("1180x720")
        self.root.minsize(820, 520)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Sidebar.TFrame", background="#eef1f5")
        style.configure("Tool.TButton", anchor="w", padding=(10, 7))
        style.configure(
            "Section.TButton",
            anchor="w",
            padding=(4, 6),
            relief="flat",
            borderwidth=0,
            background="#eef1f5",
            font=("TkDefaultFont", 10, "bold"),
        )
        style.map("Section.TButton", background=[("active", "#dde3ea"), ("pressed", "#d0d7e0")])
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open image…", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save result as…", command=self.save_image, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu, underline=0)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_command(label="Reset to original", command=self.reset)
        menu.add_cascade(label="Edit", menu=edit_menu, underline=0)

        menu.add_cascade(label="Tools", menu=self._build_tools_menu(menu), underline=0)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menu.add_cascade(label="Help", menu=help_menu, underline=0)
        self.root.configure(menu=menu)

    def _build_tools_menu(self, menubar: tk.Menu) -> tk.Menu:
        tools_menu = tk.Menu(menubar, tearoff=False)
        for category, tools in self.registry.categories():
            submenu = tk.Menu(tools_menu, tearoff=False)
            for tool in tools:
                submenu.add_command(label=tool.title, command=lambda selected=tool: self.run_tool(selected))
                self._tool_menu_entries.append((tool, submenu, submenu.index("end")))
            tools_menu.add_cascade(label=category, menu=submenu)
        return tools_menu

    def _build_sidebar(self, parent: ttk.Panedwindow) -> ttk.Frame:
        sidebar = ttk.Frame(parent, style="Sidebar.TFrame", padding=12, width=235)
        sidebar.pack_propagate(False)
        ttk.Label(sidebar, text="Forensics tools", style="Title.TLabel", background="#eef1f5").pack(
            anchor="w", pady=(0, 8)
        )
        scroller = ScrollableFrame(sidebar, style="Sidebar.TFrame")
        scroller.pack(fill="both", expand=True)
        for category, tools in self.registry.categories():
            section = CollapsibleSection(
                scroller.content,
                category,
                expanded=True,
                style="Sidebar.TFrame",
                header_style="Section.TButton",
            )
            section.set_count(len(tools))
            section.pack(fill="x", pady=(4, 0))
            for tool in tools:
                button = ttk.Button(
                    section.body,
                    text=tool.title,
                    style="Tool.TButton",
                    command=lambda selected=tool: self.run_tool(selected),
                )
                button.pack(fill="x", padx=(12, 0), pady=2)
                button.bind("<Enter>", lambda _event, selected=tool: self.status.set(self._tool_hint(selected)))
                button.bind("<Leave>", lambda _event: self.status.set("Ready."))
                self._tool_buttons.append((tool, button))
        return sidebar

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        toolbar = ttk.Frame(container, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open image", command=self.open_image).pack(side="left")
        ttk.Button(toolbar, text="Save result", command=self.save_image).pack(side="left", padx=(6, 0))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.undo_button = ttk.Button(toolbar, text="Undo", command=self.undo)
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(toolbar, text="Redo", command=self.redo)
        self.redo_button.pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="left", padx=(6, 0))

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill="both", expand=True)

        body.add(self._build_sidebar(body), weight=0)

        self.image_view = ImageView(body)
        body.add(self.image_view, weight=1)

        inspector = ttk.Frame(body, padding=12, width=250)
        inspector.pack_propagate(False)
        body.add(inspector, weight=0)
        ttk.Label(inspector, text="Results", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        self.results = ttk.Treeview(inspector, columns=("value",), show="tree headings", height=15)
        self.results.heading("#0", text="Property")
        self.results.heading("value", text="Value")
        self.results.column("#0", width=95, stretch=True)
        self.results.column("value", width=120, stretch=True)
        self.results.pack(fill="both", expand=True)

        ttk.Label(container, textvariable=self.status, anchor="w", padding=(10, 6), relief="sunken").pack(fill="x")

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-o>", lambda _event: self.open_image())
        self.root.bind_all("<Control-Shift-S>", lambda _event: self.save_image())
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-y>", lambda _event: self.redo())

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(title="Open evidence image", filetypes=OPEN_TYPES)
        if not filename:
            return
        try:
            self.document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=self.root)
            return
        self.status.set(f"Opened {Path(filename).name}")
        self._show_default_details()
        self._refresh()

    def save_image(self) -> None:
        if not self._require_image():
            return
        source = self.document.path
        initial = f"{source.stem}_result.png" if source else "result.png"
        filename = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            initialfile=initial,
            filetypes=SAVE_TYPES,
        )
        if not filename:
            return
        try:
            self.document.save(filename)
        except OSError as error:
            messagebox.showerror("Could not save image", str(error), parent=self.root)
            return
        self.status.set(f"Saved result as {Path(filename).name}")

    def run_tool(self, tool: ForensicsTool) -> None:
        if tool.requires_image and not self._require_image():
            return
        if not tool.is_available(self.document):
            messagebox.showinfo(tool.title, tool.unavailable_message, parent=self.root)
            return
        try:
            result = tool.run(self.root, self.document)
        except Exception as error:  # keep one student feature from crashing the shell
            messagebox.showerror(f"{tool.title} failed", str(error), parent=self.root)
            self.status.set(f"Error in {tool.title}.")
            return
        if result is None:
            self.status.set(f"Cancelled {tool.title}.")
            return
        if result.image is not None:
            self.document.apply(result.image)
        self._show_details(result.details)
        self.status.set(result.message)
        self._refresh()

    def undo(self) -> None:
        if self.document.undo():
            self.status.set("Undid the last image operation.")
            self._show_default_details()
            self._refresh()

    def redo(self) -> None:
        if self.document.redo():
            self.status.set("Redid the image operation.")
            self._show_default_details()
            self._refresh()

    def reset(self) -> None:
        if self.document.reset():
            self.status.set("Restored the original image.")
            self._show_default_details()
            self._refresh()

    def _require_image(self) -> bool:
        if self.document.is_loaded:
            return True
        messagebox.showinfo("No image loaded", "Open an image first.", parent=self.root)
        return False

    def _refresh(self) -> None:
        self.image_view.show(self.document.current)
        self.undo_button.configure(state="normal" if self.document.can_undo else "disabled")
        self.redo_button.configure(state="normal" if self.document.can_redo else "disabled")
        for tool, button in self._tool_buttons:
            button.configure(state="normal" if tool.is_available(self.document) else "disabled")
        for tool, menu, index in self._tool_menu_entries:
            menu.entryconfigure(index, state="normal" if tool.is_available(self.document) else "disabled")
        title = self.document.path.name if self.document.path else "No image"
        self.root.title(f"ForensicsApp — {title}")

    def _tool_hint(self, tool: ForensicsTool) -> str:
        return tool.description if tool.is_available(self.document) else tool.unavailable_message

    def _show_default_details(self) -> None:
        image = self.document.current
        if image is None:
            self._show_details({})
            return
        self._show_details(
            {
                "File": self.document.path.name if self.document.path else "—",
                "Size": f"{image.width} × {image.height}",
                "Mode": image.mode,
            }
        )

    def _show_details(self, details: dict[str, object]) -> None:
        for item in self.results.get_children():
            self.results.delete(item)
        for name, value in details.items():
            self.results.insert("", "end", text=str(name), values=(str(value),))

    def show_about(self) -> None:
        messagebox.showinfo(
            "About ForensicsApp",
            "A modular image-forensics application for the Computer Vision course.\n\n"
            "Add each weekly feature as a tool in forensics_app/tools/.",
            parent=self.root,
        )
