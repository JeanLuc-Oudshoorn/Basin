"""Local viewer stays bounded while selecting the newest generated day."""

import json
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/01_data_generation"))
from view_generated_plots import make_server


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for day in ("2026-11-06", "2026-11-13"):
            folder = self.root / day
            folder.mkdir()
            (folder / "notes.txt").write_text("not a table")
            for index in range(5):
                table = folder / f"table_{index}"
                table.mkdir()
                (table / "series.png").write_bytes(b"example image")
        self.server = make_server(self.root, page_size=2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def get(self, endpoint):
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        try:
            connection.request("GET", endpoint)
            with connection.getresponse() as response:
                return json.load(response)
        finally:
            connection.close()

    def test_only_latest_by_default_and_optional_pin(self):
        self.assertEqual(self.get("/api/dates")["dates"], ["2026-11-13"])
        self.server.initial_date = "2026-11-06"
        self.assertEqual(self.get("/api/dates")["dates"], ["2026-11-06"])
        self.server.latest_only = False
        self.assertEqual(self.get("/api/dates")["dates"], ["2026-11-13", "2026-11-06"])

    def test_pages_cover_tables_once_and_report_saved_plots(self):
        names, offset = [], 0
        while offset is not None:
            result = self.get(f"/api/tables?date=2026-11-13&offset={offset}")
            self.assertLessEqual(len(result["tables"]), 2)
            for table in result["tables"]:
                self.assertEqual(table["plots"], ["series.png"])
                names.append(table["name"])
            offset = result["next_offset"]
        self.assertEqual(sorted(names), [f"table_{i}" for i in range(5)])
        self.assertEqual(len(names), len(set(names)))

    def test_no_plots_and_empty_day_are_explicit(self):
        for path in (self.root / "2026-11-13").glob("*/series.png"):
            path.unlink()
        result = self.get("/api/tables?date=2026-11-13")
        self.assertTrue(result["tables"])
        self.assertTrue(all(table["plots"] == [] for table in result["tables"]))
        (self.root / "2026-11-14").mkdir()
        self.assertEqual(self.get("/api/dates")["dates"], ["2026-11-14"])
        self.assertEqual(
            self.get("/api/tables?date=2026-11-14"), {"tables": [], "next_offset": None}
        )


if __name__ == "__main__":
    unittest.main()
