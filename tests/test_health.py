"""Tests for the health-check HTTP server (US-5)."""

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _start_health_server():
    """Start the health server in a background thread for testing."""
    from src.health import HEALTH_PORT, main

    thread = threading.Thread(target=main, daemon=True)
    thread.start()
    time.sleep(0.5)  # Let the server bind
    return HEALTH_PORT


class TestHealthEndpoint:
    """Tests for the /healthz endpoint."""

    def test_healthz_returns_ok(self):
        """AC2: GET /healthz returns 200 with {"status": "ok"}."""
        port = _start_health_server()
        url = f"http://localhost:{port}/healthz"

        resp = urllib.request.urlopen(url)
        assert resp.status == 200

        body = json.loads(resp.read().decode("utf-8"))
        assert body == {"status": "ok"}

    def test_healthz_content_type(self):
        """GET /healthz returns Content-Type: application/json."""
        port = _start_health_server()
        url = f"http://localhost:{port}/healthz"

        resp = urllib.request.urlopen(url)
        ct = resp.headers.get("Content-Type", "")
        assert "application/json" in ct

    def test_other_path_returns_404(self):
        """GET /anything-else returns 404."""
        port = _start_health_server()

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://localhost:{port}/other")
        assert exc_info.value.code == 404

    def test_health_module_importable(self):
        """The health module is importable and has HEALTH_PORT."""
        import src.health

        assert hasattr(src.health, "HEALTH_PORT")
        assert isinstance(src.health.HEALTH_PORT, int)
        assert src.health.HEALTH_PORT > 0
