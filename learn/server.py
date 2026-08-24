"""Local guided-learn HTTP server over the two-criterion unconstrained engine."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from admissions_simulation import (
    TwoCriterionScenario,
    UniformCostDistribution,
    evaluate_threshold_pair,
)
from explorer.server import (
    ILLUSTRATION_SHARES,
    PINNED,
    _require_number,
    _serialize_outcomes,
    analyze_request,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def before_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Outcomes at (S0, S1) = (0, 0) for the pinned illustration population."""
    university_quota = _require_number(payload, "university_quota")
    diversity_weight = _require_number(payload, "diversity_weight")
    if set(payload) - {"university_quota", "diversity_weight"}:
        raise ValueError("only university_quota and diversity_weight are allowed")
    scenario = TwoCriterionScenario(
        benefit=1.0,
        university_quota=university_quota,
        diversity_weight=diversity_weight,
        underlying_share_00=ILLUSTRATION_SHARES[0],
        underlying_share_01=ILLUSTRATION_SHARES[1],
        underlying_share_10=ILLUSTRATION_SHARES[2],
        underlying_share_11=ILLUSTRATION_SHARES[3],
    )
    outcomes = evaluate_threshold_pair(
        scenario,
        UniformCostDistribution(upper=1.0),
        0.0,
        0.0,
    )
    return {
        "pinned": dict(PINNED),
        "request": {
            "university_quota": university_quota,
            "diversity_weight": diversity_weight,
            "tutoring_threshold_0": 0.0,
            "tutoring_threshold_1": 0.0,
        },
        "outcomes": _serialize_outcomes(outcomes),
    }


class LearnHandler(BaseHTTPRequestHandler):
    server_version = "AdmissionsLearn/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"ok": True})
            return
        if path in {"/", "/index.html"}:
            index_path = STATIC_DIR / "index.html"
            self._send_bytes(200, index_path.read_bytes(), "text/html; charset=utf-8")
            return
        static_candidate = (STATIC_DIR / path.lstrip("/")).resolve()
        if (
            static_candidate.is_file()
            and STATIC_DIR.resolve() in static_candidate.parents
        ):
            content_type = "application/octet-stream"
            if static_candidate.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif static_candidate.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            self._send_bytes(200, static_candidate.read_bytes(), content_type)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/analyze", "/before"}:
            self._send_json(404, {"error": "not found"})
            return
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header else 0
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "request body must be JSON"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "request body must be a JSON object"})
            return
        try:
            if path == "/analyze":
                result = analyze_request(payload)
            else:
                result = before_request(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception:
            self._send_json(500, {"error": "internal error"})
            return
        self._send_json(200, result)


def create_learn_server(
    host: str = "127.0.0.1",
    port: int = 8766,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), LearnHandler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Guided learn surface for admissions Simulation labs (local)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    server = create_learn_server(host=args.host, port=args.port)
    host, bound_port = server.server_address[:2]
    print(f"Learn at http://{host}:{bound_port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
