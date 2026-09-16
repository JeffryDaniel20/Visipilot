"""A minimal local HTTP server for the controlled test pages under
``testpages/``. No external network dependency — purely for local,
reproducible evaluation (Instructions.md #1, implementation-plan.md A.1).
"""
from __future__ import annotations

import functools
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

TESTPAGES_DIR = Path(__file__).resolve().parent.parent / "testpages"


class TestPageServer:
    """Serves ``testpages/`` on localhost on a background thread.

    Usable as a context manager::

        with TestPageServer(port=8765) as server:
            url = server.url("search_basic.html")
            ...
    """

    __test__ = False  # not a pytest test class, despite the name

    def __init__(self, directory: Path = TESTPAGES_DIR, host: str = "127.0.0.1", port: int = 8765):
        self.directory = directory
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(self.directory))
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        # Reassign the ephemeral port back in case port=0 was requested.
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def url(self, page: str) -> str:
        return f"http://{self.host}:{self.port}/{page}"

    def __enter__(self) -> "TestPageServer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
