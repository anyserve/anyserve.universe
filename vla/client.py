"""Replay a recorded robot episode against a VLA policy through anyserve.

Plays one episode of a LeRobot dataset in real time. Every ``--execute`` ticks
the robot sends what it sees now and runs the returned actions from the tick
the answer arrives; until then it keeps to the chunk it already has. A call
that misses ``--budget-ms`` is dropped: a late action is worth nothing to a
control loop.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import math
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from anyserve import AnyserveClient, ExecutionOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a LeRobot episode against a VLA policy.")
    parser.add_argument("--endpoint", help="default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", default="lerobot/smolvla_base", help="the interface is vla/<model>")
    parser.add_argument("--dataset", default="lerobot/svla_so100_pickplace")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", action="append", metavar="POLICY=DATASET",
                        help="policy camera fed from a dataset camera (repeatable); default: camera1=top camera2=wrist")
    parser.add_argument("--execute", type=int, default=25, help="ticks between policy calls")
    parser.add_argument("--budget-ms", type=int, default=1500, help="time a policy call may take")
    parser.add_argument("--seconds", type=float, help="replay only the start of the episode")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.dataset, episodes=[args.episode])
    fps = dataset.fps
    cameras = dict(item.partition("=")[::2] for item in args.camera or ["camera1=top", "camera2=wrist"])
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
    options = ExecutionOptions(queue_timeout_ms=args.budget_ms, total_timeout_ms=args.budget_ms, max_attempts=1)
    planned: dict[int, tuple[list[float], int]] = {}  # tick -> (action, the tick it was planned from)
    latencies, inferences, dropped = [], [], 0
    tick = 0
    started = time.perf_counter()
    while tick < ticks:
        sent = time.perf_counter()
        try:
            reply = json.loads(client.call(f"vla/{args.model}", payloads[tick], options=options))
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

    # Distance from the recorded action, per joint, against holding the pose seen at the call.
    policy_error = error((action, recorded[at]) for at, (action, _) in planned.items())
    hold_error = error((states[source], recorded[at]) for at, (_, source) in planned.items())
    latencies.sort()
    inferences.sort()
    print(f"{ticks} ticks at {fps} Hz: {len(latencies)} calls, {dropped} dropped, {len(planned) / ticks:.0%} of ticks planned, "
          f"latency p50 {latencies[len(latencies) // 2]:.1f} ms (inference {inferences[len(inferences) // 2]} ms), max {latencies[-1]:.1f} ms, "
          f"action error {policy_error:.2f} against {hold_error:.2f} for holding still")


def jpeg(frame) -> str:
    buffer = io.BytesIO()
    Image.fromarray((frame.permute(1, 2, 0) * 255).byte().numpy()).save(buffer, "JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":
    main()
