# VLA policy for a robot control loop

An embodied-AI shape: a robot sends an observation every control step and a
vision-language-action policy answers with a chunk of actions. `worker.py`
serves the policy, `robot.py` is a simulated robot that runs the loop, and
everything is standard library plus the anyserve SDK, so it runs on a laptop.

- **Observation** (the payload): `{"instruction": ..., "state": [x, y], "image_png": base64}`,
  the camera frame being a 64×64 image of the scene.
- **Action chunk** (the result): `{"actions": [[dx, dy], ...]}`, executed one
  action per tick before the next inference, the way VLA models such as
  OpenVLA or pi0 are driven.

The built-in policy stands in for a model: it finds the red marker in the
frame and plans a few steps toward it after a configurable inference delay.
Put a real model behind the same contract with `--policy module:callable`,
where the callable takes the observation dict and returns the action dict.

```bash
python worker.py --site edge --latency-ms 20
python robot.py --hz 10 --budget-ms 150 --prefer-site edge
```

What anyserve adds to the loop:

- **A latency budget per step.** `robot.py` calls with `queue_timeout_ms` and
  `total_timeout_ms` equal to the budget and `max_attempts=1`; a late answer is
  worth nothing to a control loop, so the robot holds still and asks again on
  the next tick instead of waiting.
- **Near first, then anywhere.** With `--prefer-site edge` the robot prefers
  the policy server on its site and lets another one answer after
  `--spillover-ms`.
- **Degradation, not failure.** Kill the worker during an episode and the
  robot keeps ticking with fallbacks counted; start the worker again and the
  next step gets a real action. Workers reconnect by themselves.

```text
reached the target after 41 ticks: 6 inferences, 0 fallbacks, policy latency p50 24.1 ms, max 31.5 ms
```

Policy inference runs one observation at a time (`max_active_leases=1`); run
one worker per policy server and several servers per site when more robots
share them.

## A real model: SmolVLA

`smolvla.py` puts [SmolVLA](https://huggingface.co/lerobot/smolvla_base), a
450M-parameter policy from LeRobot, behind the same worker. One inference
takes the camera frames, the joint state and the instruction and returns the
next 50 joint targets. It needs about 1 GB of GPU memory.

- **Observation**: `{"instruction": ..., "state": [6 joint values], "images": {"camera1": base64, "camera2": ...}}`,
  each image a JPEG or PNG. A camera left out reaches the model as a black frame.
- **Action chunk**: `{"actions": [[6 joint targets], ...], "policy": "smolvla", "inference_ms": ...}`.

```bash
pip install "lerobot[smolvla,dataset]"
SMOLVLA_STATS=lerobot/svla_so100_pickplace python worker.py --policy smolvla:policy --site edge
python replay.py --prefer-site edge
```

`SMOLVLA_MODEL` names the checkpoint. A fine-tuned checkpoint carries the
statistics that scale joint values for its robot. The base checkpoint does
not, and answers in normalized values unless `SMOLVLA_STATS` names a LeRobot
dataset, or the path of its `meta/stats.json`, to take them from.

`replay.py` plays a recorded episode of a LeRobot dataset in real time. Every
`--execute` ticks it sends the frame the robot sees now and runs the returned
actions from the tick the answer arrives, keeping to the chunk it already has
until then. On a Tesla T4, with the SO-100 pick-and-place episode at 30 Hz:

```text
454 ticks at 30 Hz: 19 calls, 0 fallbacks, 96% of ticks planned, latency p50 588.5 ms (inference 586.3 ms), max 766.8 ms, action error 12.99 against 8.95 for holding still
```

One inference spans 18 ticks, which is what the chunk is for: the robot asks
every 25 ticks and never waits. The call through anyserve costs about 2 ms on
top of the inference. With a second worker on `--site cloud`, killing the edge
worker during the episode loses the one call in flight and the cloud worker
answers the rest. The action error is in joint degrees: the base checkpoint is
not trained for this arm and its cameras, so it lands no closer to the
recording than holding still. Fine-tune it and set `SMOLVLA_MODEL`.
