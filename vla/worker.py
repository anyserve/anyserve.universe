"""Serve a vision-language-action policy through anyserve.

The payload is one observation (instruction, proprioceptive state and a camera
frame) and the result is a chunk of actions, the shape robot control loops use
with VLA models such as OpenVLA or pi0. The built-in policy stands in for a
model: it finds the target in the frame and plans a few steps toward it, after
a configurable inference delay. ``--policy module:callable`` swaps in a real
model behind the same contract.

Observation: {"instruction": str, "state": [x, y], "image_png": base64}
Action chunk: {"actions": [[dx, dy], ...], "policy": str}
"""
from __future__ import annotations
import argparse
import base64
import importlib
import time

from anyserve import WorkerError, serve, worker

from common import TARGET, decode_png, locate


class SimulatedPolicy:
    """Looks for the red marker in the frame and moves toward it, one chunk at a time."""

    name = "simulated"

    def __init__(self, chunk: int, step: float, latency_ms: float) -> None:
        self.chunk, self.step, self.latency = chunk, step, latency_ms / 1000

    def __call__(self, observation: dict) -> dict:
        width, height, pixels = decode_png(base64.b64decode(observation["image_png"]))
        target = locate(pixels, width, height, TARGET)
        if target is None:
            raise WorkerError("no target in view")
        time.sleep(self.latency)
        x, y = observation["state"]
        actions = []
        for _ in range(self.chunk):
            dx, dy = target[0] - x, target[1] - y
            scale = min(1.0, self.step / max(1e-9, (dx * dx + dy * dy) ** 0.5))
            dx, dy = dx * scale, dy * scale
            x, y = x + dx, y + dy
            actions.append([round(dx, 4), round(dy, 4)])
        return {"actions": actions, "policy": self.name}


def load_policy(spec: str, chunk: int, step: float, latency_ms: float):
    if spec == "simulated":
        return SimulatedPolicy(chunk, step, latency_ms)
    module_name, _, attribute = spec.partition(":")
    policy = getattr(importlib.import_module(module_name), attribute or "act")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a VLA policy through anyserve.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--interface", default="vla.v1")
    parser.add_argument("--site", default="edge", help="attribute robots use to prefer the near policy server")
    parser.add_argument("--policy", default="simulated", help="simulated, or module:callable taking an observation")
    parser.add_argument("--chunk", type=int, default=8, help="actions per inference")
    parser.add_argument("--step", type=float, default=0.03, help="largest move per action, in unit coordinates")
    parser.add_argument("--latency-ms", type=float, default=20, help="simulated inference time")
    parser.add_argument("--worker-id")
    args = parser.parse_args()
    policy = load_policy(args.policy, args.chunk, args.step, args.latency_ms)

    # One inference at a time: a policy model serves one robot step per call.
    @worker(interface=args.interface, attributes={"site": args.site, "policy": getattr(policy, "name", args.policy)},
            codec="json", max_active_leases=1)
    def act(observation: dict) -> dict:
        return policy(observation)

    serve(act, args.endpoint, worker_id=args.worker_id)


if __name__ == "__main__":
    main()
