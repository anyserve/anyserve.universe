# vLLM

`worker.py` fronts a vLLM server, or any server that speaks the OpenAI chat
completions API. The payload is the request body; the worker fills in the
model name when the request omits it.

- Without `"stream"` the result is the response body.
- With `"stream": true` every `data:` line of the server-sent event stream
  becomes one output frame, so a client reading Invoke events prints tokens as
  vLLM produces them.

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
python worker.py --vllm http://127.0.0.1:8000 --model Qwen/Qwen2.5-0.5B-Instruct
python client.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Say hello in one sentence."
```

`--concurrency` bounds the requests in flight per worker; vLLM batches them.
Run one worker per vLLM server and prefer the near one with
`preferred_attributes` and `spillover_after_ms` from the client. When the
server is unreachable the worker reports a retryable failure so the control
plane can run the job elsewhere; a rejected request (HTTP 4xx) fails without
a retry.
