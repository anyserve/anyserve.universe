# VLA

`worker.py` serves [SmolVLA](https://huggingface.co/lerobot/smolvla_base), a
450M-parameter vision-language-action policy from LeRobot, on the interface
`vla/<model>`. One call is one robot observation and the result is the next
chunk of actions, 50 per inference. It needs about 1.2 GB of GPU memory.

- **Payload**: `{"instruction": ..., "state": [6 joint values], "images": {"camera1": base64, "camera2": ...}}`,
  each image a JPEG or PNG. A camera left out reaches the model as a black frame.
- **Result**: `{"actions": [[6 joint targets], ...], "inference_ms": ...}`.

`client.py` plays a recorded episode of a LeRobot dataset in real time. Every
`--execute` ticks it sends what the robot sees now and runs the returned
actions from the tick the answer arrives, keeping to the chunk it already has
until then. A call that misses `--budget-ms` is dropped: a late action is worth
nothing to a control loop.

```bash
pip install "lerobot[smolvla,dataset]"
python worker.py --stats lerobot/svla_so100_pickplace
python client.py
```

A fine-tuned checkpoint (`--model`) carries the statistics that scale joint
values for its robot. The base checkpoint does not, and answers in normalized
values unless `--stats` names a LeRobot dataset, or the path of its
`meta/stats.json`, to take them from.

On a Tesla T4, with an SO-100 pick-and-place episode at 30 Hz:

```text
454 ticks at 30 Hz: 19 calls, 0 dropped, 96% of ticks planned, latency p50 587.9 ms (inference 585.3 ms), max 597.3 ms, action error 12.67 against 8.93 for holding still
```

- One inference spans 18 ticks, which is what the chunk is for: the robot asks
  every 25 ticks and never waits. The call costs under 3 ms on top of the inference.
- One worker serves one robot at this rate. Two robots on one worker wait for
  each other (p50 1,140 ms a call), and a second worker on the same T4 shares the
  GPU (950 ms an inference). More robots need more GPUs.
- With two workers, killing one during the episode loses the call in flight
  and the other answers the rest.
- The action error is in joint degrees. The base checkpoint is not trained for
  this arm and its cameras, so it lands no closer to the recording than holding
  still. Fine-tune it and pass `--model`.
