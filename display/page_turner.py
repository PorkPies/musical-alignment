"""
Page turner: receives predicted bar numbers, applies predictive logic
(turning LEAD_TIME seconds before the first bar of the next page),
and calls a page_change_callback.
"""
import os
import sys
import queue
import threading

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_dir)

from data.scripts.extract_bar_times import extract_bar_times

# Default lead time in seconds before the page boundary to trigger a turn.
LEAD_TIME = 2.5


def _build_page_first_and_last(bar_to_page):
    """
    Return two dicts:
      page_first_bar[page_idx] = smallest bar number on that page
      page_last_bar[page_idx]  = largest bar number on that page
    """
    page_first_bar = {}
    page_last_bar = {}
    for bar_num, page_idx in bar_to_page.items():
        if page_idx not in page_first_bar or bar_num < page_first_bar[page_idx]:
            page_first_bar[page_idx] = bar_num
        if page_idx not in page_last_bar or bar_num > page_last_bar[page_idx]:
            page_last_bar[page_idx] = bar_num
    return page_first_bar, page_last_bar


class PageTurner:
    """
    Receives predicted bar numbers (via push_prediction) and decides when
    to trigger page turns, including a predictive early-turn based on
    known bar durations.

    Usage:
        turner = PageTurner(bar_to_page, bar_times, on_page_change)
        turner.start()
        # … from inference callback:
        turner.push_prediction(bar_number)
        turner.stop()
    """

    def __init__(self, bar_to_page, bar_times, page_change_callback, lead_time=LEAD_TIME):
        """
        Parameters:
            bar_to_page (dict[int, int]): {bar_number: page_index}
            bar_times (list[float]): Bar onset times in seconds (0-indexed, from extract_bar_times).
            page_change_callback (callable): Called with (page_index) when a turn is needed.
            lead_time (float): Seconds before the page boundary to trigger the turn.
        """
        self.bar_to_page = bar_to_page
        self.bar_times = bar_times
        self.page_change_callback = page_change_callback
        self.lead_time = lead_time

        self.current_page = 0
        self._page_first_bar, self._page_last_bar = _build_page_first_and_last(bar_to_page)
        self._prediction_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def push_prediction(self, bar_number, confidence=1.0):
        """Thread-safe: enqueue a new bar prediction."""
        self._prediction_queue.put((bar_number, confidence))

    def _bar_time(self, bar_number):
        """Return onset time (s) for a 1-indexed bar number, or None if out of range."""
        idx = bar_number - 1
        if 0 <= idx < len(self.bar_times):
            return self.bar_times[idx]
        return None

    def _try_predictive_turn(self, bar_number):
        """
        If the predicted bar is close enough to the end of the current page,
        pre-fire a turn to the next page.
        """
        next_page = self.current_page + 1
        if next_page not in self._page_first_bar:
            return  # already on the last page

        # The turn should happen just before the first bar of the next page.
        first_bar_next = self._page_first_bar[next_page]
        target_time = self._bar_time(first_bar_next)
        current_time = self._bar_time(bar_number)

        if target_time is None or current_time is None:
            return

        time_remaining = target_time - current_time
        if 0 <= time_remaining <= self.lead_time:
            self.current_page = next_page
            self.page_change_callback(next_page)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                bar_number, _conf = self._prediction_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            predicted_page = self.bar_to_page.get(bar_number, self.current_page)

            if predicted_page > self.current_page:
                # Reactive turn: model says we're already on a new page.
                self.current_page = predicted_page
                self.page_change_callback(predicted_page)
            else:
                # Predictive turn: check if we should turn early.
                self._try_predictive_turn(bar_number)


if __name__ == "__main__":
    import time

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xml = os.path.join(base_dir, "data", "scores", "bach_chorale_0.musicxml")
    bar_times = extract_bar_times(xml)

    from display.score_renderer import build_bar_to_page, render_score_pages

    out_dir = os.path.join(base_dir, "display", "rendered")
    pages = render_score_pages(xml, out_dir)
    bar_to_page = build_bar_to_page(xml, pages)

    def on_page_change(page_idx):
        print(f">>> Page turn: now showing page {page_idx + 1}")

    turner = PageTurner(bar_to_page, bar_times, on_page_change)
    turner.start()

    print("Simulating bar predictions 1–30...")
    for bar in range(1, 31):
        turner.push_prediction(bar)
        time.sleep(0.3)

    turner.stop()
    print("Done.")
