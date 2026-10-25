import assert from "node:assert/strict";
import { test } from "node:test";

import { DEFAULTS, buildClientSnippet, buildServeCommand, buildServerToml, parseBytes, problems } from "./config.js";

test("defaults produce only the server and console sections", () => {
  assert.equal(
    buildServerToml(DEFAULTS),
    '[server]\ngrpc_host = "127.0.0.1"\ngrpc_port = 50052\n\n[console]\nlisten = "127.0.0.1:3001"\n',
  );
  assert.deepEqual(problems(DEFAULTS), []);
});

test("changed values are written with their types", () => {
  const state = {
    ...DEFAULTS,
    grpc_host: "0.0.0.0",
    tls: true,
    cert_file: "/etc/anyserve/tls.crt",
    key_file: "/etc/anyserve/tls.key",
    journal_path: "/data/journal.sqlite",
    full_sync: false,
    retention_secs: "86400",
    client_tokens: "client-a\n\nclient-b\n",
    worker_tokens: "worker-a",
    allowed_origins: "http://localhost:5173",
    runtime: { ...DEFAULTS.runtime, max_attempts: "5", max_event_bytes: "512 MiB", max_queue: "10000" },
    interfaces: [{ name: "image.v1", max_queue: "3", queue_timeout_ms: "250" }],
  };
  const toml = buildServerToml(state);
  assert.match(toml, /\[server\.tls\]\nenabled = true\ncert_file = "\/etc\/anyserve\/tls\.crt"\nkey_file = "\/etc\/anyserve\/tls\.key"/);
  assert.match(toml, /\[journal\]\npath = "\/data\/journal\.sqlite"\nfull_sync = false\nretention_secs = 86400\n/);
  assert.match(toml, /\[auth\]\nclient_tokens = \["client-a", "client-b"\]\nworker_tokens = \["worker-a"\]\n/);
  assert.match(toml, /\[console\]\nlisten = "127\.0\.0\.1:3001"\nallowed_origins = \["http:\/\/localhost:5173"\]\n/);
  assert.match(toml, /\[runtime\]\nmax_attempts = 5\nmax_event_bytes = 536870912\n/);
  assert.doesNotMatch(toml, /max_queue = 10000/, "a value equal to the default is not written");
  assert.match(toml, /\[runtime\.interfaces\."image\.v1"\]\nmax_queue = 3\nqueue_timeout_ms = 250\n/);
  assert.equal(buildServeCommand(state).flagsOnly, null);
  assert.deepEqual(problems(state), []);
});

test("a deployment the flags cover needs no file", () => {
  const state = { ...DEFAULTS, console: false, journal: false, grpc_port: "6000" };
  assert.equal(buildServerToml(state), '[server]\ngrpc_host = "127.0.0.1"\ngrpc_port = 6000\n\n[journal]\nenabled = false\n');
  assert.equal(buildServeCommand(state).flagsOnly, "anyserve serve --grpc-host 127.0.0.1 --grpc-port 6000 --ephemeral");
  assert.equal(buildServeCommand(state).withConfig, "anyserve serve --config server.toml");
});

test("byte sizes and invalid numbers", () => {
  assert.equal(parseBytes("2 GiB"), 2 * 1024 ** 3);
  assert.equal(parseBytes("512MB"), 512_000_000);
  assert.equal(parseBytes("1048576"), 1_048_576);
  assert.equal(parseBytes("lots"), null);
  const state = { ...DEFAULTS, grpc_port: "fifty", runtime: { ...DEFAULTS.runtime, max_event_bytes: "lots" } };
  assert.deepEqual(problems(state), ["gRPC port must be a number", "Event budget must be a byte size such as 512 MiB"]);
  assert.doesNotMatch(buildServerToml(state), /max_event_bytes/);
});

test("client snippet follows the listener and tokens", () => {
  const state = { ...DEFAULTS, grpc_host: "0.0.0.0", tls: true, client_tokens: "c1", worker_tokens: "w1" };
  const snippet = buildClientSnippet(state);
  assert.match(snippet, /export ANYSERVE_TOKEN=c1\nanyserve --endpoint https:\/\/HOST:50052 job ls/);
  assert.match(snippet, /AnyserveClient\("https:\/\/HOST:50052", token="c1"\)/);
  assert.match(snippet, /serve\(echo, endpoint="https:\/\/HOST:50052", token="w1"\)/);
});
