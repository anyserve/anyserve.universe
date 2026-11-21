"""Serve SmolVLA, a vision-language-action policy, through anyserve.

One call is one robot observation and the result is the next chunk of actions,
50 per inference:

    payload  {"instruction": str, "state": [float, ...], "images": {"camera1": base64 JPEG or PNG, ...}}
    result   {"actions": [[float, ...], ...], "inference_ms": float}

A camera the observation leaves out reaches the model as a black frame. A
fine-tuned checkpoint carries the statistics that scale joint values for its
robot. The base checkpoint does not: without ``--stats`` it answers in
normalized values instead of the robot's units.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import time

import numpy
import torch
from huggingface_hub import hf_hub_download
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from PIL import Image

from anyserve import WorkerError, serve, worker

IMAGES = "observation.images."


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SmolVLA through anyserve.")
    parser.add_argument("--endpoint", help="anyserve control plane; default: ANYSERVE_ENDPOINT or http://127.0.0.1:50052")
    parser.add_argument("--model", default="lerobot/smolvla_base", help="checkpoint to load; the interface is vla/<model>")
    parser.add_argument("--stats", help="LeRobot dataset, or the path of its meta/stats.json, to scale joint values with")
    parser.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE", help="worker attribute for placement, such as site=edge (repeatable)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = SmolVLAPolicy.from_pretrained(args.model).to(device).eval()
    stats = {"stats": load_stats(args.stats)} if args.stats else {}
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": device}, "normalizer_processor": stats},
        postprocessor_overrides={"unnormalizer_processor": stats})
    features = policy.config.input_features
    joints = features["observation.state"].shape[0]
    cameras = {key[len(IMAGES):]: tuple(feature.shape) for key, feature in features.items() if key.startswith(IMAGES)}

    # One inference at a time: the model serves one robot step per call.
    @worker(interface=f"vla/{args.model}", attributes=dict(item.partition("=")[::2] for item in args.attr),
            codec="json", max_active_leases=1)
    def act(observation: dict) -> dict:
        started = time.perf_counter()
        state, images = observation.get("state"), observation.get("images") or {}
        if not isinstance(state, list) or len(state) != joints:
            raise WorkerError(f"state must be a list of {joints} numbers")
        if set(images) - set(cameras):
            raise WorkerError(f"unknown camera {sorted(set(images) - set(cameras))[0]!r}; this checkpoint takes {sorted(cameras)}")
        batch = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": str(observation.get("instruction", ""))}
        for camera, shape in cameras.items():
            batch[IMAGES + camera] = decode(images[camera]) if camera in images else torch.zeros(shape)
        with torch.inference_mode():
            actions = post(policy.predict_action_chunk(pre(batch)))[0].float().cpu().tolist()
        return {"actions": [[round(value, 4) for value in action] for action in actions],
                "inference_ms": round((time.perf_counter() - started) * 1000, 1)}

    act({"state": [0.0] * joints})  # the first inference is slow; pay for it before taking calls
    serve(act, args.endpoint)


def decode(image: str) -> torch.Tensor:
    try:
        pixels = numpy.array(Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB"))
    except Exception as error:
        raise WorkerError(f"a camera frame is not a JPEG or PNG image: {error}")
    return torch.from_numpy(pixels).permute(2, 0, 1).float() / 255


def load_stats(source: str) -> dict:
    path = source if os.path.exists(source) else hf_hub_download(source, "meta/stats.json", repo_type="dataset")
    with open(path) as file:
        stats = json.load(file)
    return {key: {name: torch.tensor(value) for name, value in stats[key].items()} for key in ("observation.state", "action")}


if __name__ == "__main__":
    main()
