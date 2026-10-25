// Builds a server.toml, the matching `anyserve serve` command and client
// snippets from the setup form. Only values that differ from the defaults are
// written, so the file stays as short as the deployment needs.

export const RUNTIME_FIELDS = [
  { key: "heartbeat_interval_ms", label: "Heartbeat interval (ms)", default: "5000", hint: "Heartbeat period sent to workers." },
  { key: "heartbeat_timeout_ms", label: "Heartbeat timeout (ms)", default: "15000", hint: "Session silence limit, and the reconnect grace for its leases." },
  { key: "max_attempts", label: "Max attempts", default: "3", hint: "Attempts per job, including the first, when the job sets none." },
  { key: "max_queue", label: "Max queue per interface", default: "10000", hint: "Queued jobs per interface before admission refuses more." },
  { key: "max_pending_jobs", label: "Max pending jobs", default: "100000", hint: "Unfinished jobs across all interfaces." },
  { key: "max_pending_bytes", label: "Input budget", default: "2 GiB", hint: "Inputs and params of unfinished jobs.", bytes: true },
  { key: "max_result_bytes", label: "Result budget", default: "2 GiB", hint: "Result bytes alive in the control plane.", bytes: true },
  { key: "max_event_bytes", label: "Event budget", default: "1 GiB", hint: "Event logs of all jobs.", bytes: true },
  { key: "max_sessions", label: "Max worker sessions", default: "4096", hint: "Connected workers, including those in reconnect grace." },
  { key: "consumer_stall_timeout_ms", label: "Consumer stall timeout (ms)", default: "30000", hint: "A streamed consumer may stop reading this long." },
  { key: "cancel_grace_ms", label: "Cancel grace (ms)", default: "15000", hint: "Wait for a worker to acknowledge a cancellation." },
  { key: "terminal_retention_secs", label: "Terminal retention (s)", default: "600", hint: "Finished jobs stay in memory this long." },
  { key: "max_terminal_jobs", label: "Max terminal jobs", default: "10000", hint: "Finished jobs kept in memory." },
  { key: "event_log_max_events", label: "Events per job", default: "8192", hint: "Event log length per job." },
  { key: "event_log_max_bytes", label: "Bytes per job log", default: "8 MiB", hint: "Event log size per job.", bytes: true },
];

export const DEFAULTS = Object.freeze({
  grpc_host: "127.0.0.1",
  grpc_port: "50052",
  tls: false,
  cert_file: "",
  key_file: "",
  journal: true,
  journal_path: ".anyserve/journal.sqlite",
  full_sync: true,
  retention_secs: "604800",
  queue_capacity: "4096",
  max_batch: "256",
  client_tokens: "",
  worker_tokens: "",
  client_tokens_env: "",
  worker_tokens_env: "",
  console: true,
  console_listen: "127.0.0.1:3001",
  allowed_origins: "",
  runtime: Object.freeze(Object.fromEntries(RUNTIME_FIELDS.map((field) => [field.key, ""]))),
  interfaces: Object.freeze([]),
});

const UNITS = {
  "": 1,
  b: 1,
  k: 1024,
  kib: 1024,
  kb: 1000,
  m: 1024 ** 2,
  mib: 1024 ** 2,
  mb: 1000 ** 2,
  g: 1024 ** 3,
  gib: 1024 ** 3,
  gb: 1000 ** 3,
};

/** "2 GiB", "512MB" or "1048576" as a byte count; null when unreadable. */
export function parseBytes(text) {
  const match = /^\s*(\d+(?:\.\d+)?)\s*([a-z]*)\s*$/i.exec(text);
  if (!match) {
    return null;
  }
  const unit = UNITS[match[2].toLowerCase()];
  if (unit === undefined) {
    return null;
  }
  return Math.round(Number(match[1]) * unit);
}

function parseInteger(text) {
  return /^\s*\d+\s*$/.test(text) ? Number(text) : null;
}

function quote(value) {
  return JSON.stringify(value);
}

function lines(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function section(out, header, body) {
  if (body.length === 0) {
    return;
  }
  if (out.length > 0) {
    out.push("");
  }
  out.push(header, ...body);
}

/** Field values that cannot be written, by label. */
export function problems(state) {
  const found = [];
  if (parseInteger(state.grpc_port) === null) {
    found.push("gRPC port must be a number");
  }
  for (const [label, value] of [
    ["Journal retention", state.retention_secs],
    ["Journal queue capacity", state.queue_capacity],
    ["Journal max batch", state.max_batch],
  ]) {
    if (state.journal && value.trim() !== "" && parseInteger(value) === null) {
      found.push(`${label} must be a number`);
    }
  }
  for (const field of RUNTIME_FIELDS) {
    const value = state.runtime[field.key] ?? "";
    if (value.trim() === "") {
      continue;
    }
    const parsed = field.bytes ? parseBytes(value) : parseInteger(value);
    if (parsed === null) {
      found.push(`${field.label} must be ${field.bytes ? "a byte size such as 512 MiB" : "a number"}`);
    }
  }
  state.interfaces.forEach((item, index) => {
    if (item.name.trim() === "" && (item.max_queue.trim() !== "" || item.queue_timeout_ms.trim() !== "")) {
      found.push(`Interface override ${index + 1} needs a name`);
    }
    for (const value of [item.max_queue, item.queue_timeout_ms]) {
      if (value.trim() !== "" && parseInteger(value) === null) {
        found.push(`Interface override ${index + 1} must use numbers`);
      }
    }
  });
  return found;
}

export function buildServerToml(state) {
  const out = [];
  const host = state.grpc_host.trim() || DEFAULTS.grpc_host;
  const port = parseInteger(state.grpc_port) ?? Number(DEFAULTS.grpc_port);
  section(out, "[server]", [`grpc_host = ${quote(host)}`, `grpc_port = ${port}`]);
  if (state.tls) {
    const tls = ["enabled = true"];
    if (state.cert_file.trim()) {
      tls.push(`cert_file = ${quote(state.cert_file.trim())}`);
    }
    if (state.key_file.trim()) {
      tls.push(`key_file = ${quote(state.key_file.trim())}`);
    }
    section(out, "[server.tls]", tls);
  }

  const journal = [];
  if (!state.journal) {
    journal.push("enabled = false");
  } else {
    if (state.journal_path.trim() && state.journal_path.trim() !== DEFAULTS.journal_path) {
      journal.push(`path = ${quote(state.journal_path.trim())}`);
    }
    if (!state.full_sync) {
      journal.push("full_sync = false");
    }
    for (const [key, value, fallback] of [
      ["retention_secs", state.retention_secs, DEFAULTS.retention_secs],
      ["queue_capacity", state.queue_capacity, DEFAULTS.queue_capacity],
      ["max_batch", state.max_batch, DEFAULTS.max_batch],
    ]) {
      const parsed = parseInteger(value);
      if (parsed !== null && parsed !== Number(fallback)) {
        journal.push(`${key} = ${parsed}`);
      }
    }
  }
  section(out, "[journal]", journal);

  const auth = [];
  const clientTokens = lines(state.client_tokens);
  const workerTokens = lines(state.worker_tokens);
  if (clientTokens.length > 0) {
    auth.push(`client_tokens = [${clientTokens.map(quote).join(", ")}]`);
  }
  if (state.client_tokens_env.trim()) {
    auth.push(`client_tokens_env = ${quote(state.client_tokens_env.trim())}`);
  }
  if (workerTokens.length > 0) {
    auth.push(`worker_tokens = [${workerTokens.map(quote).join(", ")}]`);
  }
  if (state.worker_tokens_env.trim()) {
    auth.push(`worker_tokens_env = ${quote(state.worker_tokens_env.trim())}`);
  }
  section(out, "[auth]", auth);

  if (state.console) {
    const console = [`listen = ${quote(state.console_listen.trim() || DEFAULTS.console_listen)}`];
    const origins = lines(state.allowed_origins);
    if (origins.length > 0) {
      console.push(`allowed_origins = [${origins.map(quote).join(", ")}]`);
    }
    section(out, "[console]", console);
  }

  const runtime = [];
  for (const field of RUNTIME_FIELDS) {
    const value = state.runtime[field.key] ?? "";
    if (value.trim() === "") {
      continue;
    }
    const parsed = field.bytes ? parseBytes(value) : parseInteger(value);
    const fallback = field.bytes ? parseBytes(field.default) : parseInteger(field.default);
    if (parsed !== null && parsed !== fallback) {
      runtime.push(`${field.key} = ${parsed}`);
    }
  }
  section(out, "[runtime]", runtime);
  for (const item of state.interfaces) {
    const name = item.name.trim();
    if (!name) {
      continue;
    }
    const body = [];
    const maxQueue = parseInteger(item.max_queue);
    if (maxQueue !== null) {
      body.push(`max_queue = ${maxQueue}`);
    }
    const timeout = parseInteger(item.queue_timeout_ms);
    if (timeout !== null) {
      body.push(`queue_timeout_ms = ${timeout}`);
    }
    section(out, `[runtime.interfaces.${quote(name)}]`, body);
  }
  return out.join("\n") + "\n";
}

/** The serve command with the file, and as flags alone when nothing needs the file. */
export function buildServeCommand(state) {
  const host = state.grpc_host.trim() || DEFAULTS.grpc_host;
  const port = parseInteger(state.grpc_port) ?? Number(DEFAULTS.grpc_port);
  const toml = buildServerToml(state);
  const needsFile =
    toml.includes("[auth]") ||
    toml.includes("[console]") ||
    toml.includes("[runtime") ||
    toml.includes("full_sync") ||
    toml.includes("retention_secs") ||
    toml.includes("queue_capacity") ||
    toml.includes("max_batch");
  let flags = `anyserve serve --grpc-host ${host} --grpc-port ${port}`;
  if (state.tls) {
    flags += " --grpc-tls-enabled";
    if (state.cert_file.trim()) {
      flags += ` --grpc-cert-file ${state.cert_file.trim()}`;
    }
    if (state.key_file.trim()) {
      flags += ` --grpc-key-file ${state.key_file.trim()}`;
    }
  }
  if (!state.journal) {
    flags += " --ephemeral";
  } else if (state.journal_path.trim() && state.journal_path.trim() !== DEFAULTS.journal_path) {
    flags += ` --journal ${state.journal_path.trim()}`;
  }
  return {
    withConfig: "anyserve serve --config server.toml",
    flagsOnly: needsFile ? null : flags,
  };
}

export function endpointFor(state) {
  const host = state.grpc_host.trim() || DEFAULTS.grpc_host;
  const port = parseInteger(state.grpc_port) ?? Number(DEFAULTS.grpc_port);
  const scheme = state.tls ? "https" : "http";
  const reachable = host === "0.0.0.0" || host === "::" ? "HOST" : host;
  return `${scheme}://${reachable}:${port}`;
}

export function buildClientSnippet(state) {
  const endpoint = endpointFor(state);
  const clientToken = lines(state.client_tokens)[0] ?? (state.client_tokens_env.trim() ? "CLIENT_TOKEN" : "");
  const workerToken = lines(state.worker_tokens)[0] ?? (state.worker_tokens_env.trim() ? "WORKER_TOKEN" : "");
  const shell = [
    clientToken ? `export ANYSERVE_TOKEN=${clientToken}` : "# no client token configured",
    `anyserve --endpoint ${endpoint} job ls`,
  ];
  const python = [
    "from anyserve import AnyserveClient, serve, worker",
    "",
    `client = AnyserveClient(${quote(endpoint)}${clientToken ? `, token=${quote(clientToken)}` : ""})`,
    'print(client.call("demo.echo.v1", b"hello"))',
    "",
    '@worker(interface="demo.echo.v1")',
    "def echo(payload: bytes) -> bytes:",
    "    return payload",
    "",
    `serve(echo, endpoint=${quote(endpoint)}${workerToken ? `, token=${quote(workerToken)}` : ""})`,
  ];
  return [...shell, "", ...python].join("\n") + "\n";
}
