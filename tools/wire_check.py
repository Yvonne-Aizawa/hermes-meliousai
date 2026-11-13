"""Reproducible wire check: what the installed plugin's client actually sends.

Runs a local echo server (no provider traffic, no credential), points the INSTALLED
plugin's client at it, and prints the exact request body fields and headers. This is the
one contract the offline suite cannot prove — it blocks sockets by design.

    python3 tools/wire_check.py --home "${HERMES_HOME:-$HOME/.hermes}"

Exit 0 = the wire format is right: a top-level ``request_id`` body field, the sentinel
header, streaming through, and the stream's ``.response`` (where core reads Melious'
``X-Request-ID`` from) intact.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

captured: dict = {}


class Echo(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 — http.server's naming
        length = int(self.headers.get("content-length") or 0)
        captured["path"] = self.path
        captured["headers"] = {k.lower(): v for k, v in self.headers.items()}
        captured["body"] = json.loads(self.rfile.read(length) or b"{}")
        chunk = {
            "id": "cmpl", "object": "chat.completion.chunk", "created": 0, "model": "glm-5.3-flash",
            "choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": "stop"}],
        }
        payload = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("x-request-id", "echo-upstream-id")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep the check quiet
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--home", required=True, type=Path, help='Hermes home (e.g. "$HERMES_HOME")')
    args = parser.parse_args()

    plugins_root = args.home.expanduser().resolve() / "plugins" / "model-providers"
    if not (plugins_root / "melious" / "__init__.py").is_file():
        print(f"not installed under {plugins_root}; run install.py first")
        return 2
    sys.path.insert(0, str(plugins_root))

    # Import the INSTALLED package exactly as Hermes' discovery does.
    package = importlib.import_module("melious")
    module = importlib.import_module("melious.melious_provider")

    server = HTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"

    client = package.profile.create_client(base_url=base_url, api_key="local-check-key", timeout=30)
    try:
        stream = client.chat.completions.create(
            model="glm-5.3-flash",
            messages=[{"role": "user", "content": "say pong"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = list(stream)
        upstream_id = stream.response.headers.get("x-request-id")
    finally:
        client.close()
        server.shutdown()

    body = captured.get("body", {})
    headers = captured.get("headers", {})
    request_id = body.get("request_id")
    sentinel = headers.get(module.SENTINEL_HEADER.lower())
    content = chunks[0].choices[0].delta.content if chunks else None

    print("plugin slug        :", module.SLUG)
    print("client class       :", type(client).__name__)
    print("POST path          :", captured.get("path"))
    print("body.request_id    :", request_id)
    print("body keys          :", sorted(body))
    print("sentinel header    :", sentinel)
    print("stream_options     :", body.get("stream_options"))
    print("auth header sent   :", "Bearer" in str(headers.get("authorization", "")))
    print("first chunk        :", content)
    print("stream .response   :", upstream_id)

    ok = (
        isinstance(request_id, str) and len(request_id) == 36
        and sentinel == module.SENTINEL_VALUE
        and json.dumps(body).count("request_id") == 1
        and str(captured.get("path", "")).endswith("/chat/completions")
        and content == "pong"
        and upstream_id == "echo-upstream-id"
    )
    print("WIRE CHECK         :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
