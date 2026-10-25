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
