"""Replay a recorded robot episode against a VLA policy through anyserve.

Plays one episode of a LeRobot dataset in real time. Every ``--execute`` ticks
the robot sends what it sees now (camera frames, joint state and the task) and
runs the returned actions from the tick the answer arrives; until then it keeps
running the chunk it already has. The summary says how much of the episode had
a planned action, what the calls cost, and how far the policy's actions are
from the recorded ones. Needs ``pip install "lerobot[dataset]"``.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import math
import time

from anyserve import AnyserveClient, ExecutionOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode against a VLA policy.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--interface", default="vla.v1")
    parser.add_argument("--dataset", default="lerobot/svla_so100_pickplace")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", action="append", metavar="POLICY=DATASET",
                        help="policy camera fed from a dataset camera (repeatable); default: camera1=top camera2=wrist")
    parser.add_argument("--execute", type=int, default=25, help="ticks between policy calls")
    parser.add_argument("--budget-ms", type=int, default=1500, help="time a policy call may take")
    parser.add_argument("--seconds", type=float, help="replay only the start of the episode")
    parser.add_argument("--prefer-site", help="prefer policy servers with this site attribute")
    parser.add_argument("--spillover-ms", type=int, default=30, help="wait this long for the preferred site before any server may answer")
    parser.add_argument("--json", action="store_true", help="print the summary as one JSON line")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    def jpeg(frame) -> str:
        buffer = io.BytesIO()
        Image.fromarray((frame.permute(1, 2, 0) * 255).byte().numpy()).save(buffer, "JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode()

    dataset = LeRobotDataset(args.dataset, episodes=[args.episode])
    fps = dataset.fps
    cameras = dict(item.split("=", 1) for item in args.camera or ["camera1=top", "camera2=wrist"])
    ticks = min(len(dataset), int(args.seconds * fps)) if args.seconds else len(dataset)
    payloads, states, recorded = [], [], []
    for index in range(ticks):
        item = dataset[index]
        states.append(item["observation.state"].tolist())
        recorded.append(item["action"].tolist())
        payloads.append(json.dumps({
            "instruction": item["task"], "state": states[-1],
            "images": {name: jpeg(item[f"observation.images.{key}"]) for name, key in cameras.items()},
        }).encode())

    client = AnyserveClient(args.endpoint)
    options = ExecutionOptions(
        queue_timeout_ms=args.budget_ms,
        total_timeout_ms=args.budget_ms,
        max_attempts=1,
        preferred_attributes={"site": args.prefer_site} if args.prefer_site else {},
        spillover_after_ms=args.spillover_ms if args.prefer_site else 0,
    )
    planned: dict[int, tuple[list[float], int]] = {}  # tick -> (action, the tick it was planned from)
    latencies, inferences = [], []
    fallbacks = 0
    tick = 0
    started = time.perf_counter()
    while tick < ticks:
        sent = time.perf_counter()
        try:
            reply = json.loads(client.call(args.interface, payloads[tick], options=options))
            elapsed = time.perf_counter() - sent
            latencies.append(elapsed * 1000)
            inferences.append(reply.get("inference_ms"))
            arrives = tick + math.ceil(elapsed * fps)
            for offset, action in enumerate(reply["actions"]):
                if arrives <= tick + offset < ticks:
                    planned[tick + offset] = (action, tick)
        except RuntimeError as error:
            fallbacks += 1  # the robot keeps running the chunk it already has
            if not args.json:
                print(f"tick {tick}: no policy within {args.budget_ms} ms ({str(error)[:60]})")
        tick = max(tick + args.execute, math.ceil((time.perf_counter() - started) * fps))
        time.sleep(max(0.0, started + tick / fps - time.perf_counter()))

    def mean_error(pairs) -> float | None:
        errors = [abs(a - b) for left, right in pairs for a, b in zip(left, right)]
        return round(sum(errors) / len(errors), 2) if errors else None

    latencies.sort()
    timed = [value for value in inferences if value is not None]
    summary = {
        "ticks": ticks, "hz": fps, "calls": len(latencies), "fallbacks": fallbacks,
        "covered": round(len(planned) / ticks, 3),
        "payload_kb": round(sum(map(len, payloads)) / len(payloads) / 1024, 1),
        "latency_p50_ms": round(latencies[len(latencies) // 2], 1) if latencies else None,
        "latency_max_ms": round(latencies[-1], 1) if latencies else None,
        "inference_p50_ms": sorted(timed)[len(timed) // 2] if timed else None,
        # Distance from the recorded action, per joint, against holding the pose seen at the call.
        "policy_error": mean_error((action, recorded[at]) for at, (action, _) in planned.items()),
        "hold_error": mean_error((states[source], recorded[at]) for at, (_, source) in planned.items()),
    }
    if args.json:
        print(json.dumps(summary))
    else:
        print(f"{ticks} ticks at {fps} Hz: {summary['calls']} calls, {fallbacks} fallbacks, {summary['covered']:.0%} of ticks planned, "
              f"latency p50 {summary['latency_p50_ms']} ms (inference {summary['inference_p50_ms']} ms), max {summary['latency_max_ms']} ms, "
              f"action error {summary['policy_error']} against {summary['hold_error']} for holding still")


if __name__ == "__main__":
    main()
