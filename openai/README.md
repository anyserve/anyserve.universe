# OpenAI-compatible servers

`worker.py` puts any server that speaks the OpenAI HTTP API behind anyserve:
vLLM, Ollama, LM Studio, llama.cpp's server, SGLang, or an embedding server.
The connector does not care which one it is. Clients call a contract, and
whichever server implements it answers.

## Interfaces

The connector asks the server for its models (`GET <base>/models`) and
registers one interface per model and endpoint:

| Interface | Forwarded to | Payload |
|---|---|---|
| `openai/chat/<model>` | `POST <base>/chat/completions` | chat completions request body |
| `openai/completions/<model>` | `POST <base>/completions` | completions request body |
| `openai/embeddings/<model>` | `POST <base>/embeddings` | embeddings request body |

The payload is the JSON body as the OpenAI API takes it; the connector fills
in `"model"` when the body omits it and changes nothing else.

- Without `"stream"` the result is the response body.
- With `"stream": true` every `data:` line of the server-sent event stream
  becomes one output frame, so a client reading Invoke events sees tokens as
  the server produces them.

The model is part of the interface name, so each model has its own queue,
limits and metrics in the control plane. Attributes describe the deployment,
not the model: use them for placement such as `site=edge`.

## Run

```bash
# vLLM
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
python worker.py --base-url http://127.0.0.1:8000/v1

# Ollama: every pulled model becomes an interface
python worker.py --base-url http://127.0.0.1:11434/v1

# an embedding server
python worker.py --base-url http://127.0.0.1:8080/v1 --serve embeddings

python client.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Say hello in one sentence."
```

| Option | Meaning |
|---|---|
| `--base-url` | the server's OpenAI base URL; default `http://127.0.0.1:8000/v1` |
| `--api-key` | bearer token for the server; default `OPENAI_API_KEY` |
| `--model` | serve only these models (repeatable); default: every model the server lists |
| `--serve` | endpoints to register: `chat` (default), `completions`, `embeddings`, comma separated |
| `--concurrency` | requests in flight on this server, shared by its interfaces; default 16 |
| `--attr KEY=VALUE` | worker attribute for placement (repeatable) |
| `--pool` | scheduling pool the server belongs to |
| `--wait` | seconds to wait for the server to answer at startup; default 600 |

Embeddings from Python:

```python
import json
from anyserve import AnyserveClient

body = {"input": ["a sentence", "another one"]}
reply = json.loads(AnyserveClient().call("openai/embeddings/BAAI/bge-m3", json.dumps(body).encode()))
vectors = [item["embedding"] for item in reply["data"]]
```

## Behaviour

- The connector waits for the server at startup, so it can be started
  together with a server that is still loading its model. The model list is
  read once; restart the connector after adding a model.
- A server that is unreachable, overloaded (HTTP 429) or failing (HTTP 5xx)
  is reported as retryable, so the control plane runs the job on another
  server within the job's attempt budget. A rejected request (other HTTP 4xx)
  fails without a retry.
- Closing a streaming Invoke closes the connection to the server, which stops
  generating.
- The connector forwards requests; it does not translate between APIs. A
  server that offers other endpoints under the same base URL is reached only
  through the three above.

Run one connector per server. To prefer a near server and spill to a far one,
give each an attribute and call with `preferred_attributes` and
`spillover_after_ms`.
