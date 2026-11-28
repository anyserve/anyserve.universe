"""Replay a recorded robot episode (or run a quick synthetic probe) against a VLA policy through anyserve.

Plays one episode of a LeRobot dataset in real time. Every ``--execute`` ticks
the robot sends what it sees now and runs the returned actions from the tick
the answer arrives; until then it keeps to the chunk it already has. A call
that misses ``--budget-ms`` is dropped: a late action is worth nothing to a
control loop.

Pass ``--probe`` to send a single synthetic observation without downloading a
LeRobot dataset (useful for smoke tests and latency checks).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import time
from pathlib import Path

from PIL import Image

from anyserve import AnyserveClient, ExecutionOptions


def parse_attrs(items: list[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid attribute {item!r}: expected KEY=VALUE")
        key, value = item.split("=", 1)
        attrs[key] = value
    return attrs


def run_probe(client: AnyserveClient, interface: str, options: ExecutionOptions, joints: int, instruction: str) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), color=(64, 128, 192)).save(buffer, "JPEG", quality=85)
    frame_b64 = base64.b64encode(buffer.getvalue()).decode()
    payload = json.dumps(
        {
            "instruction": instruction,
            "state": [0.0] * joints,
            "images": {"camera1": frame_b64},
        }
    ).encode()
    started = time.perf_counter()
    reply = json.loads(client.call(interface, payload, options=options))
    elapsed_ms = (time.perf_counter() - started) * 1000
    actions = reply["actions"]
    print(
        f"probe {interface}: {len(actions)} actions x {len(actions[0]) if actions else 0} joints, "
        f"round-trip {elapsed_ms:.1f} ms (inference {reply.get('inference_ms')} ms), first={actions[0] if actions else []}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode or probe a VLA policy through anyserve.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", default="lerobot/smolvla_base", help="the interface is vla/<model>")
    parser.add_argument("--probe", action="store_true", help="send one synthetic observation without downloading a dataset")
    parser.add_argument("--joints", type=int, default=6, help="joint count for --probe mode (default: 6)")
    parser.add_argument("--instruction", default="Pick up the block and place it in the bin.", help="task string for --probe mode")
    parser.add_argument("--dataset", default="lerobot/svla_so100_pickplace")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--camera",
        action="append",
        metavar="POLICY=DATASET",
        help="policy camera fed from a dataset camera (repeatable); default: camera1=top camera2=wrist",
    )
    parser.add_argument("--execute", type=int, default=25, help="ticks between policy calls")
    parser.add_argument("--budget-ms", type=int, default=1500, help="time a policy call may take")
    parser.add_argument("--seconds", type=float, help="replay only the start of the episode")
    parser.add_argument("--out", help="optional path to write a JSON summary report")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="preferred worker attribute (repeatable)")
    parser.add_argument("--require", action="append", default=[], metavar="KEY=VALUE", help="required worker attribute (repeatable)")
    args = parser.parse_args()

    client = AnyserveClient(args.endpoint)
    options = ExecutionOptions(
        queue_timeout_ms=args.budget_ms,
        total_timeout_ms=args.budget_ms,
        max_attempts=1,
        preferred_attributes=parse_attrs(args.attr),
        required_attributes=parse_attrs(args.require),
    )
    interface = f"vla/{args.model}"

    if args.probe:
        run_probe(client, interface, options, args.joints, args.instruction)
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset, episodes=[args.episode])
    fps = dataset.fps
    cameras = dict(item.partition("=")[::2] for item in args.camera or ["camera1=top", "camera2=wrist"])
    ticks = min(len(dataset), int(args.seconds * fps)) if args.seconds else len(dataset)
    payloads, states, recorded = [], [], []
    for index in range(ticks):
        item = dataset[index]
        states.append(item["observation.state"].tolist())
        recorded.append(item["action"].tolist())
        payloads.append(
            json.dumps(
                {
                    "instruction": item["task"],
                    "state": states[-1],
                    "images": {name: jpeg(item[f"observation.images.{key}"]) for name, key in cameras.items()},
                }
            ).encode()
        )

    planned: dict[int, tuple[list[float], int]] = {}  # tick -> (action, the tick it was planned from)
    latencies, inferences, dropped = [], [], 0
    tick = 0
    started = time.perf_counter()
    while tick < ticks:
        sent = time.perf_counter()
        try:
            reply = json.loads(client.call(interface, payloads[tick], options=options))
            elapsed = time.perf_counter() - sent
            latencies.append(elapsed * 1000)
            inferences.append(reply["inference_ms"])
            arrives = tick + math.ceil(elapsed * fps)
            for offset, action in enumerate(reply["actions"]):
                if arrives <= tick + offset < ticks:
                    planned[tick + offset] = (action, tick)
        except RuntimeError as error:
            dropped += 1  # the robot keeps to the chunk it already has
            print(f"tick {tick}: no policy within {args.budget_ms} ms ({str(error)[:60]})")
        tick = max(tick + args.execute, math.ceil((time.perf_counter() - started) * fps))
        time.sleep(max(0.0, started + tick / fps - time.perf_counter()))
    if not latencies:
        raise SystemExit(f"no call was answered; {dropped} dropped")

    def error(pairs) -> float:
        distances = [abs(a - b) for left, right in pairs for a, b in zip(left, right)]
        return sum(distances) / max(1, len(distances))

    policy_error = error((action, recorded[at]) for at, (action, _) in planned.items())
    hold_error = error((states[source], recorded[at]) for at, (_, source) in planned.items())
    latencies.sort()
    inferences.sort()
    summary = {
        "interface": interface,
        "dataset": args.dataset,
        "episode": args.episode,
        "ticks": ticks,
        "fps": fps,
        "calls": len(latencies),
        "dropped": dropped,
        "planned_ratio": round(len(planned) / ticks, 4),
        "latency_p50_ms": round(latencies[len(latencies) // 2], 1),
        "inference_p50_ms": round(inferences[len(inferences) // 2], 1),
        "latency_max_ms": round(latencies[-1], 1),
        "policy_action_error": round(policy_error, 3),
        "hold_action_error": round(hold_error, 3),
    }
    print(
        f"{ticks} ticks at {fps} Hz: {len(latencies)} calls, {dropped} dropped, {len(planned) / ticks:.0%} of ticks planned, "
        f"latency p50 {summary['latency_p50_ms']} ms (inference {summary['inference_p50_ms']} ms), max {summary['latency_max_ms']} ms, "
        f"action error {policy_error:.2f} against {hold_error:.2f} for holding still"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")


def jpeg(frame) -> str:
    buffer = io.BytesIO()
    Image.fromarray((frame.permute(1, 2, 0) * 255).byte().numpy()).save(buffer, "JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":
    main()
