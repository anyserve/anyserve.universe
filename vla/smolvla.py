"""SmolVLA behind the VLA worker: ``python worker.py --policy smolvla:policy``.

SmolVLA is a 450M-parameter vision-language-action model from LeRobot. One
inference turns camera frames, joint state and an instruction into the next
50 actions. Needs ``pip install "lerobot[smolvla]"``; the checkpoint comes
from ``SMOLVLA_MODEL`` (default ``lerobot/smolvla_base``).

A fine-tuned checkpoint carries the statistics that scale joint values for its
robot. The base checkpoint does not: name them with ``SMOLVLA_STATS``, a LeRobot
dataset id or the path of its ``meta/stats.json``, or the actions come back
normalized instead of in the robot's units.

Observation: {"instruction": str, "state": [float, ...],
              "images": {"camera1": base64 JPEG or PNG, "camera2": ..., "camera3": ...}}
Action chunk: {"actions": [[float, ...], ...], "policy": "smolvla", "inference_ms": float}

A camera the observation leaves out is sent to the model as a black frame.
"""
from __future__ import annotations
import base64
import io
import json
import os
import time

import numpy
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from anyserve import WorkerError
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

MODEL = os.environ.get("SMOLVLA_MODEL", "lerobot/smolvla_base")
STATS = os.environ.get("SMOLVLA_STATS")
IMAGE_PREFIX = "observation.images."


class SmolVLA:
    name = "smolvla"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SmolVLAPolicy.from_pretrained(MODEL).to(self.device).eval()
        stats = {"stats": load_stats(STATS)} if STATS else {}
        self.pre, self.post = make_pre_post_processors(
            self.model.config, pretrained_path=MODEL,
            preprocessor_overrides={"device_processor": {"device": self.device}, "normalizer_processor": stats},
            postprocessor_overrides={"unnormalizer_processor": stats})
        features = self.model.config.input_features
        self.state_size = features["observation.state"].shape[0]
        self.cameras = {key[len(IMAGE_PREFIX):]: tuple(feature.shape)
                        for key, feature in features.items() if key.startswith(IMAGE_PREFIX)}
        # The first inference compiles kernels; pay for it before the worker takes calls.
        self({"instruction": "warm up", "state": [0.0] * self.state_size, "images": {}})

    def __call__(self, observation: dict) -> dict:
        started = time.perf_counter()
        state = observation.get("state")
        if not isinstance(state, list) or len(state) != self.state_size:
            raise WorkerError(f"state must be a list of {self.state_size} numbers")
        images = observation.get("images") or {}
        unknown = sorted(set(images) - set(self.cameras))
        if unknown:
            raise WorkerError(f"unknown camera {unknown[0]!r}; this checkpoint takes {sorted(self.cameras)}")
        batch = {"observation.state": torch.tensor(state, dtype=torch.float32),
                 "task": str(observation.get("instruction", ""))}
        for camera, shape in self.cameras.items():
            batch[IMAGE_PREFIX + camera] = self._frame(images[camera]) if camera in images else torch.zeros(shape)
        with torch.inference_mode():
            actions = self.post(self.model.predict_action_chunk(self.pre(batch)))
        if self.device == "cuda":
            torch.cuda.synchronize()
        return {"actions": [[round(value, 4) for value in action] for action in actions[0].float().cpu().tolist()],
                "policy": self.name, "inference_ms": round((time.perf_counter() - started) * 1000, 1)}

    @staticmethod
    def _frame(encoded: str) -> torch.Tensor:
        try:
            image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except Exception as error:
            raise WorkerError(f"a camera frame is not a JPEG or PNG image: {error}")
        return torch.from_numpy(numpy.array(image)).permute(2, 0, 1).float() / 255


def load_stats(source: str) -> dict:
    path = source if os.path.exists(source) else hf_hub_download(source, "meta/stats.json", repo_type="dataset")
    with open(path) as file:
        stats = json.load(file)
    return {key: {name: torch.tensor(value) for name, value in stats[key].items()} for key in ("observation.state", "action")}


policy = SmolVLA()
