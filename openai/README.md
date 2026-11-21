# OpenAI-compatible servers

`worker.py` puts any server that speaks the OpenAI HTTP API behind anyserve:
vLLM, Ollama, LM Studio, llama.cpp's server, SGLang, or an embedding server.
It asks the server for its models (`GET <base>/models`) and registers one
interface per model and endpoint:

| Interface | Forwarded to |
|---|---|
| `openai/chat/<model>` | `POST <base>/chat/completions` |
| `openai/completions/<model>` | `POST <base>/completions` |
| `openai/embeddings/<model>` | `POST <base>/embeddings` |

The payload is the JSON body as the OpenAI API takes it; the connector fills in
`"model"` when the body omits it and changes nothing else. The result is the
response body. With `"stream": true` every `data:` line of the event stream is
one output frame, so a client reading Invoke events sees tokens as the server
produces them, and closing the Invoke stops the generation.

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
python worker.py --base-url http://127.0.0.1:8000/v1
python client.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Say hello in one sentence."

python worker.py --base-url http://127.0.0.1:11434/v1                 # Ollama: every pulled model
python worker.py --base-url http://127.0.0.1:8080/v1 --serve embeddings
```

| Option | Meaning |
|---|---|
| `--base-url` | the server's OpenAI base URL; default `http://127.0.0.1:8000/v1` |
| `--api-key` | bearer token for the server; default `OPENAI_API_KEY` |
| `--model` | serve only these models (repeatable); default: every model the server lists |
| `--serve` | endpoint to register (repeatable): `chat` (default), `completions`, `embeddings` |
| `--concurrency` | requests in flight on this server, shared by its interfaces; default 16 |
| `--attr KEY=VALUE` | worker attribute for placement (repeatable) |

Embeddings from Python:

```python
import json
from anyserve import AnyserveClient

body = {"input": ["a sentence", "another one"]}
reply = json.loads(AnyserveClient().call("openai/embeddings/BAAI/bge-m3", json.dumps(body).encode()))
vectors = [item["embedding"] for item in reply["data"]]
```

The connector waits for the server at startup, so both can be started together.
A server that is unreachable, overloaded (HTTP 429) or failing (HTTP 5xx) is
retryable; any other HTTP 4xx fails without a retry. Run one connector per
server.

On a Tesla T4 with vLLM serving Qwen2.5-0.5B-Instruct, 32 output tokens a request:

| Callers | Direct | Through anyserve |
|---|---|---|
| 1 | 5.3 req/s, p50 188 ms | 5.3 req/s, p50 189 ms |
| 8 | 36.8 req/s, p50 217 ms | 36.1 req/s, p50 221 ms |
| 32 | 95.4 req/s, p50 338 ms | 93.9 req/s, p50 335 ms |

Streaming, the first token arrives after 19.5 ms direct and 19.7 ms through anyserve.
