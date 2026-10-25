"""Run ComfyUI workflows through anyserve.

The payload is a workflow in ComfyUI's API format (the "Save (API Format)"
export). The worker queues it, waits for the run and returns the first image
the workflow saved. ComfyUI keeps models and the graph runtime; anyserve adds
queueing across machines, deadlines and cancellation.
"""
from __future__ import annotations
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from anyserve import WorkerError, serve, worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ComfyUI workflows through anyserve.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:50052", help="anyserve control plane")
    parser.add_argument("--comfyui", default="http://127.0.0.1:8188", help="ComfyUI base URL")
    parser.add_argument("--interface", default="comfyui")
    parser.add_argument("--poll", type=float, default=0.5, help="seconds between history checks")
    parser.add_argument("--worker-id")
    args = parser.parse_args()
    base = args.comfyui.rstrip("/")

    def call(path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{base}{path}", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            raise WorkerError(f"comfyui returned HTTP {error.code} for {path}: {error.read().decode(errors='replace')[:300]}")
        except (urllib.error.URLError, TimeoutError) as error:
            raise WorkerError(f"comfyui unreachable: {error}", retryable=True)

    # One workflow at a time: ComfyUI runs its queue serially on one GPU.
    @worker(interface=args.interface, attributes={"backend": "comfyui"}, max_active_leases=1, pass_context=True)
    def generate(payload: bytes, context) -> bytes:
        workflow = json.loads(payload)
        prompt_id = json.loads(call("/prompt", {"prompt": workflow, "client_id": uuid.uuid4().hex}))["prompt_id"]
        while True:
            history = json.loads(call(f"/history/{prompt_id}"))
            if prompt_id in history:
                break
            if context.cancelled:
                call("/queue", {"delete": [prompt_id]})
                call("/interrupt", {})
                return b""
            time.sleep(args.poll)
        entry = history[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise WorkerError(f"workflow failed: {json.dumps(status.get('messages', []))[:300]}")
        for node in entry.get("outputs", {}).values():
            for image in node.get("images", []):
                query = urllib.parse.urlencode({"filename": image["filename"], "subfolder": image.get("subfolder", ""),
                                                "type": image.get("type", "output")})
                return call(f"/view?{query}")
        raise WorkerError("the workflow saved no image")

    serve(generate, args.endpoint, worker_id=args.worker_id)


if __name__ == "__main__":
    main()
