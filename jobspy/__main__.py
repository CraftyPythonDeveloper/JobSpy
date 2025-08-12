"""
scrape_jobs_gui.py
PyQt5 UI for scrape_jobs (threaded, with logging capture)

Dependencies:
    pip install PyQt5
Usage:
    python scrape_jobs_gui.py
"""

import sys
import logging
import queue
import threading
import traceback
from typing import List, Optional

from PyQt5 import QtWidgets, QtCore

# Try to import your scrape_jobs function from the jobspy package.
# If your package path is different, adjust the import or set PYTHONPATH accordingly.
try:
    from jobspy import scrape_jobs
except Exception:
    # Fallback: if jobspy is not importable as a package, try to import from local file.
    # Edit the path below if needed (point to your __init__.py that defines scrape_jobs).
    import importlib.util
    import os

    local_path = os.path.join(os.path.dirname(__file__), "__init__.py")
    if os.path.exists(local_path):
        spec = importlib.util.spec_from_file_location("jobspy_local", local_path)
        jobspy_local = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(jobspy_local)
        scrape_jobs = jobspy_local.scrape_jobs
    else:
        raise ImportError(
            "Could not import scrape_jobs from jobspy. "
            "Ensure this script is run from the project's root or adjust imports."
        )

# --- Logging -> UI glue ----------------------------------------------------
LOG_QUEUE: "queue.Queue[str]" = queue.Queue()

class QueueLoggingHandler(logging.Handler):
    """A logging handler that puts formatted log records into a queue."""
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        self.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.q.put(msg)
        except Exception:
            self.q.put("Logging error:\n" + traceback.format_exc())

class StdoutRedirector:
    """Redirects writes (print) to logging so it ends up in the queue as well."""
    def __init__(self, level=logging.INFO):
        self.level = level
    def write(self, s: str) -> None:
        s = str(s)
        if s.strip():
            logging.getLogger("stdout").log(self.level, s.rstrip())
    def flush(self) -> None:
        pass

# Configure root logger to forward to queue
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers (avoids duplicate logs)
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

queue_handler = QueueLoggingHandler(LOG_QUEUE)
root_logger.addHandler(queue_handler)

# Also attach to all existing loggers (so JobSpy:* loggers work too)
for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.addHandler(queue_handler)
    logger.propagate = True  # ensure it bubbles up to root

# Redirect stdout/stderr so print() also shows in logs
sys.stdout = StdoutRedirector(logging.INFO)
sys.stderr = StdoutRedirector(logging.ERROR)

# --- Worker thread ---------------------------------------------------------
class ScrapeWorker(QtCore.QThread):
    """
    Worker runs scrape_jobs in a separate thread and emits finished/error signals.
    """

    finished_signal = QtCore.pyqtSignal(object)  # emits the returned DataFrame or None
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, kwargs: dict):
        super().__init__()
        self.kwargs = kwargs
        self._result = None

    def run(self):
        try:
            logging.getLogger("scrape_worker").info("Starting scrape_jobs with kwargs: %s", self.kwargs)
            # Call the actual function (this may use threads internally)
            result = scrape_jobs(
                site_name = self.kwargs["site_name"],
                search_term = self.kwargs["search_term"],
                google_search_term = self.kwargs.get("google_search_term"),
                location = self.kwargs.get("location"),
                results_wanted = self.kwargs.get("results_wanted"),
                country_indeed = "india",
                hours_old = self.kwargs.get("hours_old"),
                verbose = 2,
                export_to_excel = True,
            )
            self._result = result
            logging.getLogger("scrape_worker").info("scrape_jobs completed.")
            self.finished_signal.emit(result)
        except Exception:
            tb = traceback.format_exc()
            logging.getLogger("scrape_worker").exception("Exception in scrape_worker")
            self.error_signal.emit(tb)


# --- Main Window -----------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Job Scraper UI")
        self.resize(900, 700)

        # Central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # --- Input form (scrollable) ---
        form_widget = QtWidgets.QWidget()
        form_layout = QtWidgets.QGridLayout(form_widget)
        row = 0

        # Sites multi-select list (site value strings expected by scrape_jobs)
        # Based on your uploaded code's mapping: LinkedIn, Indeed, ZipRecruiter, Glassdoor, Google, Bayt, Naukri, BDJobs.
        # We expose user-friendly labels but store values that match the expected site strings.
        self.site_map = [
            ("LinkedIn", "linkedin"),
            ("Indeed", "indeed"),
            ("ZipRecruiter", "zip_recruiter"),
            ("Glassdoor", "glassdoor"),
            ("Google", "google"),
            ("Bayt", "bayt"),
            ("Naukri", "naukri"),
            ("BDJobs", "bdjobs"),
        ]
        site_label = QtWidgets.QLabel("Sites (multi-select):")
        form_layout.addWidget(site_label, row, 0)
        self.sites_list = QtWidgets.QListWidget()
        self.sites_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for label, _val in self.site_map:
            it = QtWidgets.QListWidgetItem(label)
            self.sites_list.addItem(it)
        # pre-select a couple
        for i in (0, 1):
            self.sites_list.item(i).setSelected(True)
        form_layout.addWidget(self.sites_list, row, 1)
        row += 1

        # Search term
        form_layout.addWidget(QtWidgets.QLabel("Search term:"), row, 0)
        self.search_input = QtWidgets.QLineEdit()
        form_layout.addWidget(self.search_input, row, 1)
        row += 1

        # Google search term
        form_layout.addWidget(QtWidgets.QLabel("Google search term:"), row, 0)
        self.google_search_input = QtWidgets.QLineEdit()
        form_layout.addWidget(self.google_search_input, row, 1)
        row += 1

        # Location
        form_layout.addWidget(QtWidgets.QLabel("Location:"), row, 0)
        self.location_input = QtWidgets.QLineEdit()
        form_layout.addWidget(self.location_input, row, 1)
        row += 1

        # Is remote (tri-state via combo: Any/True/False)
        form_layout.addWidget(QtWidgets.QLabel("Is remote:"), row, 0)
        self.is_remote_combo = QtWidgets.QComboBox()
        self.is_remote_combo.addItems(["Any", "True", "False"])
        form_layout.addWidget(self.is_remote_combo, row, 1)
        row += 1

        # Results wanted
        form_layout.addWidget(QtWidgets.QLabel("Results wanted:"), row, 0)
        self.results_wanted = QtWidgets.QSpinBox()
        self.results_wanted.setRange(1, 1000)
        self.results_wanted.setValue(15)
        form_layout.addWidget(self.results_wanted, row, 1)
        row += 1

        # Hours old
        form_layout.addWidget(QtWidgets.QLabel("Hours old (None=leave blank):"), row, 0)
        self.hours_old_input = QtWidgets.QSpinBox()
        self.hours_old_input.setRange(0, 1000000)
        self.hours_old_input.setValue(0)
        h_hours = QtWidgets.QWidget()
        h_hours_l = QtWidgets.QHBoxLayout(h_hours)
        h_hours_l.addWidget(self.hours_old_input)
        form_layout.addWidget(h_hours, row, 1)
        row += 1

        # Submit button + status label
        self.start_button = QtWidgets.QPushButton("Start Scrape")
        self.start_button.clicked.connect(self.on_start)
        self.status_label = QtWidgets.QLabel("Ready")
        hb = QtWidgets.QWidget()
        hb_l = QtWidgets.QHBoxLayout(hb)
        hb_l.addWidget(self.start_button)
        hb_l.addWidget(self.status_label)
        hb_l.addStretch()
        form_layout.addWidget(hb, row, 0, 1, 2)
        row += 1

        # Add the form widget inside a scroll area (useful when many options)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        main_layout.addWidget(scroll, 70)

        # --- Logs area ---
        logs_label = QtWidgets.QLabel("Logs:")
        main_layout.addWidget(logs_label)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text, 30)

        # Timer to poll LOG_QUEUE and append to the QTextEdit
        self.log_timer = QtCore.QTimer()
        self.log_timer.setInterval(150)
        self.log_timer.timeout.connect(self.flush_log_queue)
        self.log_timer.start()

        # Keep worker reference
        self.worker: Optional[ScrapeWorker] = None

    # --- UI Actions ------------------------------------------------------
    def browse_ca_cert(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CA cert")
        if path:
            self.ca_cert_input.setText(path)

    def flush_log_queue(self):
        appended = False
        while True:
            try:
                msg = LOG_QUEUE.get_nowait()
            except queue.Empty:
                break
            else:
                self.log_text.append(msg)
                appended = True
        if appended:
            # auto-scroll
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def on_start(self):
        # disable the button immediately
        self.start_button.setEnabled(False)
        self.status_label.setText("Starting...")

        # collect inputs and convert typesse
        kwargs = {}

        # sites: multi-select
        selected_items = self.sites_list.selectedItems()
        if selected_items:
            # map label back to value
            label_to_val = {label: val for label, val in self.site_map}
            sites = [label_to_val[it.text()] for it in selected_items]
            kwargs["site_name"] = sites
        else:
            kwargs["site_name"] = None

        def none_if_empty(s):
            s = s.strip() if isinstance(s, str) else s
            return s if s else None

        search_term = none_if_empty(self.search_input.text())
        if search_term is None:
            # if the search input is empty, display a warning
            self.status_label.setText("Please enter a search term.")
            self.start_button.setEnabled(True)
            return

        kwargs["search_term"] = search_term
        kwargs["google_search_term"] = none_if_empty(self.google_search_input.text())
        kwargs["location"] = none_if_empty(self.location_input.text())

        # tri-state combos
        def combo_to_opt(combo: QtWidgets.QComboBox):
            text = combo.currentText()
            if text == "Any":
                return None
            return True if text == "True" else False

        kwargs["is_remote"] = combo_to_opt(self.is_remote_combo)
        kwargs["results_wanted"] = int(self.results_wanted.value())
        kwargs["hours_old"] = None if int(self.hours_old_input.value()) < 1 else int(self.hours_old_input.value())


        # Start the worker thread
        self.worker = ScrapeWorker(kwargs)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()
        logging.getLogger("ui").info("Worker started.")

    def on_finished(self, result):
        # result is likely a DataFrame or None
        try:
            import pandas as pd

            if isinstance(result, pd.DataFrame):
                rows = len(result)
                logging.getLogger("ui").info(f"Scraping finished. Rows obtained: {rows}")
                self.status_label.setText(f"Finished: {rows} rows")
            else:
                logging.getLogger("ui").info("Scraping finished (no rows).")
                self.status_label.setText("Finished.")
        except Exception:
            logging.getLogger("ui").info("Scraping finished (couldn't inspect result).")
            self.status_label.setText("Finished (unknown result).")

        self.start_button.setEnabled(True)

    def on_error(self, tb_text: str):
        logging.getLogger("ui").error("Worker error:\n" + tb_text)
        self.status_label.setText("Error - check logs")
        self.start_button.setEnabled(True)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
