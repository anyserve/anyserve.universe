"""Queue a ComfyUI workflow through anyserve and save the image it produces."""
from __future__ import annotations
import argparse
from pathlib import Path

from anyserve import AnyserveClient, ExecutionOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a workflow and save its first image.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:50052")
    parser.add_argument("--workflow", required=True, help="workflow JSON exported with Save (API Format)")
    parser.add_argument("--out", default="output.png")
    args = parser.parse_args()
    client = AnyserveClient(args.endpoint)
    image = client.call("comfyui", Path(args.workflow).read_bytes(),
                        options=ExecutionOptions(queue_timeout_ms=600_000, total_timeout_ms=1_800_000))
    Path(args.out).write_bytes(image)
    print(f"saved {args.out} ({len(image)} bytes)")


if __name__ == "__main__":
    main()
