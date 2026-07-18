#!/usr/bin/env python3
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class DebugHandler(BaseHTTPRequestHandler):
    ui_dir: Path
    bridge_json: Path
    pipeline_json: Path

    def log_message(self, fmt, *args):
        return

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (self.ui_dir / "index.html").read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
            return

        if self.path == "/telemetry":
            data = {
                "bridge": self._read_json(self.bridge_json),
                "pipeline": self._read_json(self.pipeline_json),
            }
            self._send(
                200,
                "application/json; charset=utf-8",
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
            )
            return

        self._send(404, "text/plain; charset=utf-8", b"not found")

    @staticmethod
    def _read_json(path):
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            return {"error": f"json decode failed: {exc}"}
        except OSError as exc:
            return {"error": f"read failed: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ui-dir", type=Path, default=root / "debug_ui")
    parser.add_argument(
        "--bridge-json",
        type=Path,
        default=root / "build" / "debug" / "aim_bridge.json",
    )
    parser.add_argument(
        "--pipeline-json",
        type=Path,
        default=root / "build" / "debug" / "aim_pipeline.json",
    )
    args = parser.parse_args()

    DebugHandler.ui_dir = args.ui_dir
    DebugHandler.bridge_json = args.bridge_json
    DebugHandler.pipeline_json = args.pipeline_json

    server = ThreadingHTTPServer((args.host, args.port), DebugHandler)
    print(f"Aim debug UI: http://{args.host}:{args.port}/")
    print(f"Bridge telemetry: {args.bridge_json}")
    print(f"Pipeline telemetry: {args.pipeline_json}")
    server.serve_forever()


if __name__ == "__main__":
    main()
