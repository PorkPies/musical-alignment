"""
Score display: a tkinter window that shows rendered score pages and
responds to page-change events queued from the inference pipeline.

Standalone usage:
    python display/app.py [path/to/score.musicxml]

Keyboard shortcuts:
    Right arrow / Left arrow — manual page flip
    Escape                    — quit
"""
import os
import sys
import queue
import tkinter as tk

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Image display will not work.")


class ScoreDisplay:
    """
    tkinter window that displays score page images.

    Page changes can be requested from any thread via request_page(page_idx).
    """

    def __init__(self, page_images, fullscreen=True):
        """
        Parameters:
            page_images (dict[int, str]): {page_index: png_path}
            fullscreen (bool): Start fullscreen.
        """
        self.page_images = page_images
        self.current_page = 0
        self._page_queue = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Musical Score")
        self.root.configure(bg="white")

        if fullscreen:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry("1024x768")

        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<Right>", lambda e: self._manual_turn(1))
        self.root.bind("<Left>", lambda e: self._manual_turn(-1))

        self._label = tk.Label(self.root, bg="white")
        self._label.pack(expand=True, fill=tk.BOTH)

        self._tk_images = {}
        if _PIL_AVAILABLE:
            self._load_images()

        if page_images:
            self._show_page(min(page_images.keys()))

        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_images(self):
        """Pre-load and cache all page images scaled to screen size."""
        self.root.update_idletasks()
        w = self.root.winfo_screenwidth() or 1024
        h = self.root.winfo_screenheight() or 768
        for idx, path in self.page_images.items():
            if os.path.exists(path):
                img = Image.open(path)
                img.thumbnail((w, h), Image.LANCZOS)
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, "white")
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                self._tk_images[idx] = ImageTk.PhotoImage(img)
            else:
                print(f"Warning: page image not found: {path}")

    # ------------------------------------------------------------------
    # Page navigation
    # ------------------------------------------------------------------

    def _show_page(self, page_idx):
        if page_idx in self._tk_images:
            self._label.configure(image=self._tk_images[page_idx])
            self.current_page = page_idx
            self.root.title(f"Musical Score — Page {page_idx + 1} / {len(self._tk_images)}")
        elif not self._tk_images:
            self._label.configure(
                text=f"Page {page_idx + 1}\n(no image)",
                fg="white",
                font=("Helvetica", 48),
            )
            self.current_page = page_idx
            self.root.title(f"Musical Score — Page {page_idx + 1}")

    def _manual_turn(self, delta):
        new_page = self.current_page + delta
        max_page = max(self.page_images.keys()) if self.page_images else 0
        if 0 <= new_page <= max_page:
            self._show_page(new_page)

    # ------------------------------------------------------------------
    # Thread-safe page change
    # ------------------------------------------------------------------

    def request_page(self, page_idx):
        """Enqueue a page change from any thread."""
        self._page_queue.put(page_idx)

    def _poll_queue(self):
        """Called by tkinter event loop every 100 ms to drain the queue."""
        try:
            while True:
                page_idx = self._page_queue.get_nowait()
                self._show_page(page_idx)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        """Block until the window is closed."""
        self.root.mainloop()


def run_display(xml_path, fullscreen=True):
    """
    Convenience function: render pages and return an initialised ScoreDisplay
    together with bar_to_page and bar_times for wiring up the inference pipeline.

    Returns:
        (ScoreDisplay, bar_to_page, bar_times)
    """
    from display.score_renderer import render_score_pages, build_bar_to_page
    from data.scripts.extract_bar_times import extract_bar_times

    out_dir = os.path.join(base_dir, "display", "rendered")
    page_images = render_score_pages(xml_path, out_dir)
    if not page_images:
        print("Warning: no page images rendered.")

    bar_to_page = build_bar_to_page(xml_path, page_images)
    bar_times = extract_bar_times(xml_path)

    display = ScoreDisplay(page_images, fullscreen=fullscreen)
    return display, bar_to_page, bar_times


if __name__ == "__main__":
    xml = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(base_dir, "data", "scores", "bach_chorale_0.musicxml")
    )
    display, bar_to_page, bar_times = run_display(xml, fullscreen=False)
    print(f"Displaying score with {len(bar_to_page)} bars across {len(display.page_images)} page(s).")
    print("Use arrow keys to flip pages manually. Press Escape to quit.")
    display.run()
