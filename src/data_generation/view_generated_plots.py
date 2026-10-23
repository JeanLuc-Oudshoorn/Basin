"""Run this file to browse generated series and DAGs in your default browser.

Uses only Python's standard library. Stop the local server with Ctrl+C.
"""

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

# Settings: select a day here, or switch dates inside the viewer.
GENERATED_DIR = Path(__file__).resolve().parents[2] / "data" / "generated"
INITIAL_DATE = None  # None selects the latest available day, e.g. "2026-10-23".
PORT = 0  # Zero chooses an available local port automatically.
OPEN_BROWSER = True


class PlotHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Keep the console quiet while scrolling through lazy-loaded images.

    def respond(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        request = urlsplit(self.path)
        root = self.server.generated_dir.resolve()
        if request.path == "/":
            self.respond(
                Path(__file__).with_name("plot_viewer.html").read_bytes(),
                "text/html; charset=utf-8",
            )
            return
        if request.path == "/api/dates":
            dates = (
                sorted((p.name for p in root.iterdir() if p.is_dir()), reverse=True)
                if root.is_dir()
                else []
            )
            self.respond(
                json.dumps(
                    {"dates": dates, "initial_date": self.server.initial_date}
                ).encode(),
                "application/json",
            )
            return
        if request.path == "/api/tables":
            day = parse_qs(request.query).get("date", [""])[0]
            folder = (root / day).resolve()
            if folder.parent != root or not folder.is_dir():
                self.send_error(404, "Date folder not found")
                return
            tables = []
            for table in sorted(folder.iterdir()):
                if table.is_dir() and table.resolve().parent == folder:
                    tables.append(
                        {
                            "name": table.name,
                            "plots": [
                                name
                                for name in ("series.png", "dag.png")
                                if (table / name).is_file()
                                and (table / name).resolve().parent == table.resolve()
                            ],
                        }
                    )
            self.respond(json.dumps(tables).encode(), "application/json")
            return
        if request.path.startswith("/image/"):
            parts = unquote(request.path[len("/image/") :]).split("/")
            # Serve only the two generated plot files, confined to day/table.
            if len(parts) == 3 and parts[-1] in ("series.png", "dag.png"):
                path = root.joinpath(*parts).resolve()
                if (
                    path.is_relative_to(root)
                    and len(path.relative_to(root).parts) == 3
                    and path.is_file()
                ):
                    self.respond(path.read_bytes(), "image/png")
                    return
        self.send_error(404)


def make_server(generated_dir=GENERATED_DIR, port=PORT, initial_date=INITIAL_DATE):
    # Loopback binding keeps this a local desktop utility.
    server = ThreadingHTTPServer(("127.0.0.1", port), PlotHandler)
    server.generated_dir = Path(generated_dir)
    server.initial_date = initial_date
    return server


def main():
    with make_server() as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(
            f"Plot viewer: {url}\nData: {GENERATED_DIR}\nPress Ctrl+C to close the viewer server.",
            flush=True,
        )
        if OPEN_BROWSER:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nViewer closed.")


if __name__ == "__main__":
    main()
