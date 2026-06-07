"""Tests for the health-check HTTP server (US-5)."""

import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _find_free_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_health_server(port: int, timeout: float = 1.0) -> bool:
    """Start the health server in a background thread on the given port."""
    import src.health

    src.health.HEALTH_PORT = port
    thread = threading.Thread(target=src.health.main, daemon=True)
    thread.start()

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/healthz")
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    return False


class TestHealthEndpoint:
    """Tests for the /healthz endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Assign a unique port and start the server."""
        self.port = _find_free_port()
        assert _start_health_server(self.port), "Health server failed to start"

    def test_healthz_returns_ok(self):
        """AC2: GET /healthz returns 200 with {"status": "ok"}."""
        resp = urllib.request.urlopen(f"http://localhost:{self.port}/healthz")
        assert resp.status == 200

        body = json.loads(resp.read().decode("utf-8"))
        assert body == {"status": "ok"}

    def test_healthz_content_type(self):
        """GET /healthz returns Content-Type: application/json."""
        resp = urllib.request.urlopen(f"http://localhost:{self.port}/healthz")
        ct = resp.headers.get("Content-Type", "")
        assert "application/json" in ct

    def test_other_path_returns_404(self):
        """GET /anything-else returns 404."""
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://localhost:{self.port}/other")
        assert exc_info.value.code == 404

    def test_health_module_importable(self):
        """The health module is importable and has HEALTH_PORT."""
        import src.health

        assert hasattr(src.health, "HEALTH_PORT")
        assert isinstance(src.health.HEALTH_PORT, int)
        assert src.health.HEALTH_PORT > 0
