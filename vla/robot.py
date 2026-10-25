"""A simulated robot control loop that asks a VLA policy for action chunks through anyserve.

Each control step the robot renders its camera frame, sends an observation with
a latency budget, and executes the returned action chunk one action per tick.
When no policy answers within the budget it falls back to holding still, so a
lost worker or a slow network degrades the episode instead of ending it.
"""
from __future__ import annotations
import argparse
import json
import random
import time

from anyserve import AnyserveClient, ExecutionOptions

from common import observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simulated robot against a VLA policy.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--interface", default="vla.v1")
    parser.add_argument("--hz", type=float, default=10, help="control rate")
    parser.add_argument("--steps", type=int, default=300, help="give up after this many ticks")
    parser.add_argument("--budget-ms", type=int, default=150, help="time a policy call may take")
    parser.add_argument("--prefer-site", help="prefer policy servers with this site attribute")
    parser.add_argument("--spillover-ms", type=int, default=30, help="wait this long for the preferred site before any server may answer")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="print the summary as one JSON line")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    effector = (rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3))
    target = (rng.uniform(0.6, 0.9), rng.uniform(0.6, 0.9))
    client = AnyserveClient(args.endpoint)
    options = ExecutionOptions(
        queue_timeout_ms=args.budget_ms,
        total_timeout_ms=args.budget_ms,
        max_attempts=1,
        preferred_attributes={"site": args.prefer_site} if args.prefer_site else {},
        spillover_after_ms=args.spillover_ms if args.prefer_site else 0,
    )
    period = 1 / args.hz
    chunk: list[list[float]] = []
    latencies: list[float] = []
    inferences = fallbacks = 0
    success = False
    for step in range(args.steps):
        started = time.perf_counter()
        if not chunk:
            try:
                reply = json.loads(client.call(args.interface, json.dumps(observation(effector, target)).encode(), options=options))
                chunk = list(reply["actions"])
                latencies.append((time.perf_counter() - started) * 1000)
                inferences += 1
            except RuntimeError as error:
                fallbacks += 1
                chunk = [[0.0, 0.0]]  # hold still until a policy answers again
                if not args.json:
                    print(f"step {step}: no policy within {args.budget_ms} ms ({str(error)[:60]}); holding")
        dx, dy = chunk.pop(0)
        effector = (effector[0] + dx, effector[1] + dy)
        if (effector[0] - target[0]) ** 2 + (effector[1] - target[1]) ** 2 < 0.03 ** 2:
            success = True
            break
        time.sleep(max(0.0, period - (time.perf_counter() - started)))
    latencies.sort()
    summary = {
        "success": success,
        "ticks": step + 1,
        "inferences": inferences,
        "fallbacks": fallbacks,
        "latency_p50_ms": round(latencies[len(latencies) // 2], 1) if latencies else None,
        "latency_max_ms": round(latencies[-1], 1) if latencies else None,
    }
    if args.json:
        print(json.dumps(summary))
    else:
        print(f"{'reached the target' if success else 'gave up'} after {summary['ticks']} ticks: "
              f"{inferences} inferences, {fallbacks} fallbacks, policy latency p50 {summary['latency_p50_ms']} ms, max {summary['latency_max_ms']} ms")


if __name__ == "__main__":
    main()
