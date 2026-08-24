"""Local Interactive explorer HTTP server for the two-criterion unconstrained engine."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from admissions_simulation import (
    TwoCriterionAnalysis,
    TwoCriterionEquilibrium,
    TwoCriterionOutcomes,
    TwoCriterionScenario,
    UniformCostDistribution,
    analyze_two_criterion_scenario,
)

ILLUSTRATION_SHARES = (0.448, 0.252, 0.192, 0.108)
STATIC_DIR = Path(__file__).resolve().parent / "static"
PINNED = {
    "benefit": 1.0,
    "underlying_share_00": ILLUSTRATION_SHARES[0],
    "underlying_share_01": ILLUSTRATION_SHARES[1],
    "underlying_share_10": ILLUSTRATION_SHARES[2],
    "underlying_share_11": ILLUSTRATION_SHARES[3],
    "cost_distribution": "uniform",
}


def _serialize_outcomes(outcomes: TwoCriterionOutcomes) -> dict[str, float]:
    return asdict(outcomes)


def _serialize_equilibrium(equilibrium: TwoCriterionEquilibrium) -> dict[str, Any]:
    return {
        "regime": equilibrium.regime.value,
        "stability": equilibrium.stability.value,
        "outcomes": _serialize_outcomes(equilibrium.outcomes),
    }


def serialize_analysis(
    analysis: TwoCriterionAnalysis,
    university_quota: float,
    diversity_weight: float,
) -> dict[str, Any]:
    equilibria = [_serialize_equilibrium(eq) for eq in analysis.equilibria]
    selected_index: int | None = None
    selected_wire: dict[str, Any] | None = None
    if analysis.selected_equilibrium is not None:
        for index, equilibrium in enumerate(analysis.equilibria):
            if equilibrium is analysis.selected_equilibrium:
                selected_index = index
                selected_wire = equilibria[index]
                break
        if selected_wire is None:
            selected_wire = _serialize_equilibrium(analysis.selected_equilibrium)
    return {
        "pinned": dict(PINNED),
        "request": {
            "university_quota": university_quota,
            "diversity_weight": diversity_weight,
        },
        "equilibria": equilibria,
        "selected_equilibrium": selected_wire,
        "selected_index": selected_index,
    }


def _require_number(payload: dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"missing field: {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def analyze_request(payload: dict[str, Any]) -> dict[str, Any]:
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
    analysis = analyze_two_criterion_scenario(
        scenario,
        UniformCostDistribution(upper=1.0),
    )
    return serialize_analysis(analysis, university_quota, diversity_weight)


class ExplorerHandler(BaseHTTPRequestHandler):
    server_version = "TwoCriterionExplorer/1.0"

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
        if path != "/analyze":
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
            result = analyze_request(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception:
            self._send_json(500, {"error": "internal error"})
            return
        self._send_json(200, result)


def create_explorer_server(
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), ExplorerHandler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Two-criterion Interactive explorer (local)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = create_explorer_server(host=args.host, port=args.port)
    host, bound_port = server.server_address[:2]
    print(f"Explorer at http://{host}:{bound_port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
