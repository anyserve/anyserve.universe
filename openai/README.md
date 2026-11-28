# OpenAI-compatible servers

`worker.py` puts any server that speaks the OpenAI HTTP API behind anyserve:
vLLM, Ollama, LM Studio, llama.cpp's server, SGLang, or an embedding server.
It asks the server for its models (`GET <base>/models`, waiting while the
server boots) and registers one interface per model and endpoint:

| Interface | Forwarded to |
|---|---|
| `openai/chat/<model>` | `POST <base>/chat/completions` |
| `openai/completions/<model>` | `POST <base>/completions` |
| `openai/embeddings/<model>` | `POST <base>/embeddings` |

The payload is the JSON body as the OpenAI API takes it; the connector fills in
`"model"` when the body omits it and changes nothing else. The result is the
response body. With `"stream": true` every `data:` line of the event stream is
one output frame, so a client reading Invoke events sees tokens as the server
produces them, and closing the Invoke cancels generation and closes the
upstream stream.

## Quickstart

```bash
# 1. Start your local backend (vLLM, Ollama, llama.cpp, SGLang, ...)
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000

# 2. Connect the outbound worker (zero open inbound ports)
python worker.py --base-url http://127.0.0.1:8000/v1 --attr gpu=t4

# 3. Stream a chat reply or compute embeddings from anywhere
python client.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Say hello in one sentence." --stats
python client.py --model BAAI/bge-m3 --mode embeddings --prompt "AnyServe routes opaque bytes."

# Ollama: auto-discover every pulled model
python worker.py --base-url http://127.0.0.1:11434/v1 --serve chat --serve embeddings
```

## Local OpenAI HTTP proxy (`proxy.py`)

`proxy.py` exposes a standard OpenAI HTTP API on localhost and forwards every
request through AnyServe — so existing tools (`curl`, the official OpenAI
Python/JS SDKs, Open WebUI, Continue, LangChain) work unchanged against GPUs
behind NAT:

```bash
python proxy.py --listen 127.0.0.1:8080 --attr gpu=t4

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="anyserve")
reply = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Hello from behind NAT!"}],
)
print(reply.choices[0].message.content)
```

## Worker options

| Option | Meaning |
|---|---|
| `--endpoint` | control plane URL (`http(s)://` or `ws(s)://`); default `ANYSERVE_ENDPOINT` or `http://127.0.0.1:50052` |
| `--base-url` | the server's OpenAI base URL; default `http://127.0.0.1:8000/v1` |
| `--api-key` | bearer token for the server; default `OPENAI_API_KEY` |
| `--model` | serve only these models (repeatable); default: every model the server lists |
| `--serve` | endpoint to register (repeatable): `chat` (default), `completions`, `embeddings` |
| `--concurrency` | requests in flight on this server, shared by its interfaces; default 16 |
| `--timeout` | upstream HTTP timeout in seconds; default 600 |
| `--pool` | worker pool name for tiering and spillover |
| `--attr KEY=VALUE` | worker attribute for placement (repeatable) |

The connector waits for the server at startup (retrying connection failures and
HTTP 5xx while loading), so both can start together. A server that is
unreachable, overloaded (HTTP 429), or failing (HTTP 5xx) is retryable; any
other HTTP 4xx fails without a retry. Run one connector per server.

## Measured

On a Tesla T4 with vLLM serving Qwen2.5-0.5B-Instruct, 32 output tokens a request:

| Callers | Direct | Through anyserve |
|---|---|---|
| 1 | 5.3 req/s, p50 188 ms | 5.3 req/s, p50 189 ms |
| 8 | 36.8 req/s, p50 217 ms | 36.1 req/s, p50 221 ms |
| 32 | 95.4 req/s, p50 338 ms | 93.9 req/s, p50 335 ms |

Streaming, the first token arrives after 19.5 ms direct and 19.7 ms through anyserve.
