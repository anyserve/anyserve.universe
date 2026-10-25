# ComfyUI

`worker.py` runs ComfyUI workflows. The payload is a workflow in ComfyUI's API
format (export it with "Save (API Format)"); the worker queues it on
`/prompt`, polls `/history` until the run ends and returns the first image the
workflow saved. ComfyUI keeps models and the graph runtime; anyserve adds
queueing across machines, deadlines and cancellation: cancelling the job
removes the workflow from ComfyUI's queue or interrupts it.

```bash
python worker.py --comfyui http://127.0.0.1:8188
python client.py --workflow workflow_api.json --out output.png
```

Each worker handles one workflow at a time, because ComfyUI runs its queue
serially on one GPU; run one worker per ComfyUI instance and let anyserve
spread workflows across them. Queue and total timeouts on the client bound the
wait. A workflow ComfyUI rejects fails without a retry; an unreachable ComfyUI
is reported as retryable so the job can run on another instance.
