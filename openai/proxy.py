"""Local OpenAI-compatible HTTP proxy backed by AnyServe workers.

Exposes a standard OpenAI HTTP API on localhost so existing clients (the OpenAI
Python/JS SDKs, curl, Open WebUI, Continue, LangChain) can talk to models behind
NAT/firewalls via AnyServe without changing client code:

    POST /v1/chat/completions   -> openai/chat/<model>
    POST /v1/completions        -> openai/completions/<model>
    POST /v1/embeddings         -> openai/embeddings/<model>
    GET  /healthz               -> 200 ok
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from anyserve import AnyserveClient, ExecutionOptions

ROUTES = {
    "/v1/chat/completions": "chat",
    "/v1/completions": "completions",
    "/v1/embeddings": "embeddings",
}


def parse_attrs(items: list[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid attribute {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        attrs[key] = value
    return attrs


def make_handler(
    client: AnyserveClient,
    options: ExecutionOptions,
    default_model: str | None,
):
    class OpenAIProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            print(f"[proxy] {self.address_string()} {fmt % args}", flush=True)

        def _send_json(self, status: int, body: bytes | dict) -> None:
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(404, {"error": {"message": f"unknown path {self.path}"}})

        def do_POST(self) -> None:
            mode = ROUTES.get(self.path.split("?", 1)[0])
            if mode is None:
                self._send_json(404, {"error": {"message": f"unknown endpoint {self.path}"}})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw_req = self.rfile.read(length)
            try:
                body = json.loads(raw_req)
            except ValueError:
                self._send_json(400, {"error": {"message": "request body must be valid JSON"}})
                return
            if not isinstance(body, dict):
                self._send_json(400, {"error": {"message": "request body must be a JSON object"}})
                return
            model = body.get("model") or default_model
            if not model:
                self._send_json(400, {"error": {"message": "missing 'model' field (or set --default-model)"}})
                return
            body["model"] = model
            interface = f"openai/{mode}/{model}"
            payload = json.dumps(body).encode()

            if body.get("stream") and mode in ("chat", "completions"):
                self._handle_stream(interface, payload)
                return

            try:
                reply = client.call(interface, payload, options=options)
            except RuntimeError as exc:
                self._send_json(502, {"error": {"message": str(exc), "type": "anyserve_error"}})
                return
            self._send_json(200, reply)

        def _handle_stream(self, interface: str, payload: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                with client.invoke(interface, payload, options=options) as events:
                    for event in events:
                        if event["type"] == "output":
                            line = event["data"].rstrip(b"\r\n") + b"\n\n"
                            self.wfile.write(line)
                            self.wfile.flush()
                        elif event["type"] == "finished":
                            if event["state"] != "succeeded":
                                err = event.get("error") or {}
                                err_payload = json.dumps(
                                    {"error": {"message": err.get("message", "stream failed"), "code": err.get("code")}}
                                ).encode()
                                self.wfile.write(b"data: " + err_payload + b"\n\n")
                                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Exiting the `with client.invoke(...)` block cancels generation on the remote worker.
                pass

    return OpenAIProxyHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose AnyServe OpenAI workers as a local HTTP OpenAI API.")
    parser.add_argument("--endpoint", help="anyserve control plane; default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--listen", default="127.0.0.1:8080", help="host:port to listen on; default: 127.0.0.1:8080")
    parser.add_argument("--default-model", help="fallback model when requests omit 'model'")
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="preferred worker attribute (repeatable)")
    parser.add_argument("--require", action="append", default=[], metavar="KEY=VALUE", help="required worker attribute (repeatable)")
    args = parser.parse_args()

    host, _, port_str = args.listen.rpartition(":")
    host = host or "127.0.0.1"
    port = int(port_str)

    client = AnyserveClient(args.endpoint)
    options = ExecutionOptions(
        total_timeout_ms=args.timeout_ms,
        preferred_attributes=parse_attrs(args.attr),
        required_attributes=parse_attrs(args.require),
    )
    server = ThreadingHTTPServer((host, port), make_handler(client, options, args.default_model))
    print(f"OpenAI HTTP proxy listening on http://{host}:{port}/v1 -> AnyServe", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
