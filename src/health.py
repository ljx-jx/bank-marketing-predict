"""Minimal health-check HTTP server for Docker / Kubernetes.

Serves a single endpoint `/healthz` that returns {"status": "ok"}.
Runs alongside the Streamlit main process in the container.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

HEALTH_PORT = 8502


class HealthHandler(BaseHTTPRequestHandler):
    """Respond to /healthz with 200 OK; everything else gets 404."""

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        """Suppress access logs."""
        pass


def main() -> None:
    """Start the health-check HTTP server."""
    server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
    print(f"Health check listening on port {HEALTH_PORT} /healthz")
    server.serve_forever()


if __name__ == "__main__":
    main()
