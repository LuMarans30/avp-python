# Latent Server (MCP + Pi)

The latent server wraps `avp-python` in a **persistent, model-resident daemon**.
Agents call two operations — `latent_think` and `latent_generate` — and pass a
short `context_id` instead of long text. The heavy KV-cache / hidden state stays
in the daemon's VRAM registry.

It exposes two surfaces:

| Surface | Endpoint | Consumer |
|---------|----------|----------|
| MCP (FastMCP) | `POST /mcp` | MCP clients |
| Plain JSON/HTTP | `/api/*` | Pi extension, `curl`, any HTTP caller |

Pi does **not** speak MCP, which is why the plain-HTTP mirror exists. Both
surfaces call the same `LatentService`.

## Install

```bash
pip install "avp[mcp]"          # avp[hf] + fastmcp
```

## Run

```bash
avp-server \
  --host 127.0.0.1 --port 8765 \
  --default-model Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --preload Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.2-3B-Instruct \
  --max-contexts 8 --ttl 300
```

Equivalent: `python -m avp.server.daemon …`. Use `--transport stdio` for MCP
clients that launch a stdio server instead of connecting to HTTP. `--token` (or
`AVP_DAEMON_TOKEN`) requires `Authorization: Bearer <token>` on every route
except `/health` — the MCP endpoint at `/mcp` included, since it can run
inference and load models. `/health` stays open so container probes need no
credentials, and binding off-loopback without a token logs a warning.

### llama.cpp / GGUF backend

```bash
avp-server --backend llamacpp \
  --default-model /models/Ling-3.0-tiny-Q4_K_M.gguf \
  --n-gpu-layers 0 --n-ctx 2048
```

`--n-gpu-layers` maps to llama.cpp layer offload (`0` = CPU only, `-1` = all
layers, the connector default), and `--n-ctx` sets the context window. This
matters for models larger than VRAM: a GGUF file bigger than the GPU will OOM
if loaded with the default `-1`.

The server uses the `llama-cpp-python` binding, which loads a shared
`libllama` via ctypes. If you already have a llama.cpp build, point the binding
at it instead of recompiling:

```bash
export LLAMA_CPP_LIB_PATH=/path/to/llama.cpp/build/bin
```

The binding's ctypes definitions must match the library version. If the
library predates the binding by a few commits it may be missing new symbols;
build `llama-cpp-python` from the matching commit if that happens.

## How routing works

`latent_think` runs `connector.think()` and stores the resulting `AVPContext`
in a thread-safe registry. `latent_generate` resolves the handle and picks a
mode:

| Context vs target model | Mode | What runs |
|-------------------------|------|-----------|
| same `model_hash` | `same_model` | `target.generate(prompt, context=ctx)` — full KV-cache |
| different `model_hash` | `cross_model` | `target.generate(prompt, context=ctx, source=src, cross_model=True)` — in-process Rosetta Stone projection |
| no context | `text` | `target.generate(prompt)` |

`output="hidden_state"` in `latent_think` drops the KV-cache and keeps only the
~6 KB last hidden state — useful when the next model differs and you want to
save VRAM.

On the **llama.cpp/GGUF** backend the cross-model projection is chosen from the
two vocabularies: vocabulary-mediated when the token maps are identical (e.g.
two Qwen sizes), and a shared-token projection when they differ (e.g. BPE vs
SentencePiece). The shared-token projection map is built once per model pair and
cached under `$AVP_CACHE_DIR/gguf_maps`, so the embedding dequantization is a
one-time cost. Pick the method with `--projection-method`:

* `vocab_overlap` (default) — softmax over shared-token logits.
* `linear` — a ridge-regression map fitted on paired shared-token embeddings;
  measurably better alignment on held-out tokens.

Cross-model transfer carries a **projected hidden trajectory** (the per-step
latent states, or a single vector when unavailable), so it suits structured
reasoning (math, code) better than verbatim recall or comprehension, where a
full KV-cache would be needed.

### Why there is no hidden-state serialization

`AVPContext.to_bytes()` only serializes KV-cache contexts; it raises for
hidden-state-only contexts, and `from_bytes()` does not restore
`last_hidden_state`. Cross-model projection additionally needs the **source
connector** in the same process. Because the daemon holds every connector, the
context never leaves the process: the `context_id` is a registry key, not a
base64 blob. There is nothing useful to hand to the agent as text.

## HTTP API

```bash
# Think (returns {"context_id": "...", ...})
curl -s localhost:8765/api/latent_think \
  -H 'content-type: application/json' \
  -d '{"prompt":"Analyze 24*17+3","model":"Qwen/Qwen2.5-7B-Instruct","steps":20}'

# Generate on the same model
curl -s localhost:8765/api/latent_generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Solve it","model":"Qwen/Qwen2.5-7B-Instruct","context_id":"<id>"}'

# Cross-model (context from a different model is projected automatically)
curl -s localhost:8765/api/latent_generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Solve it","model":"meta-llama/Llama-3.2-3B-Instruct","context_id":"<id>"}'

curl -s localhost:8765/api/status
curl -s localhost:8765/api/latent_release -H 'content-type: application/json' -d '{"context_id":"<id>"}'
curl -s localhost:8765/health
```

`latent_think` body: `prompt` (required), `model`, `steps`, `output`
(`auto` | `kv_cache` | `hidden_state`), `ttl`, `context_id` (continue a prior
same-model context).

`latent_generate` body: `prompt` (required), `model`, `context_id`, `steps`
(when no context), `max_new_tokens`, `temperature`, `top_p`, `do_sample`,
`store_context`, `ttl`.

Errors are `{"ok": false, "error": "...", "code": "..."}` with codes
`context_not_found` (404), `source_model_unavailable` / `model_in_use` (409),
`invalid_request` / `cross_model_disabled` / `context_model_mismatch` (400).

## MCP tools

`latent_think`, `latent_generate`, `latent_status`, `latent_release` — same
arguments as the HTTP API.

## Pi extension

The extension lives in [`pi-extension/`](../pi-extension/). It registers
`latent_think` and `latent_generate` as sequential tools (they share daemon
state) plus an `/avp-status` command.

```bash
# try it for one invocation
pi -e ./pi-extension

# or install it
pi install ./pi-extension
```

Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `AVP_DAEMON_URL` | `http://127.0.0.1:8765` | Daemon base URL |
| `AVP_DAEMON_TOKEN` | – | Bearer token, matching `--token` |
| `AVP_DAEMON_AUTOSTART` | `0` | `1` to spawn the daemon on session start |
| `AVP_DAEMON_STOP_ON_EXIT` | `0` | `1` to kill an autostarted daemon on shutdown |
| `AVP_DAEMON_CMD` | `avp-server` | Autostart command |

## Resource management

- Contexts expire after `--ttl` seconds (default 300) and are LRU-evicted past
  `--max-contexts` (default 8). A 7B KV-cache is ~390 MB, so keep this modest.
- Release explicitly with `latent_release` when a sub-agent is done.
- Cross-model generation requires the **source** model to still be loaded. If
  it was evicted or unloaded, `latent_generate` returns
  `source_model_unavailable`; re-run `latent_think` or reload the model.
- `latent_think` and `latent_generate` are serialized per model, so concurrent
  tool calls cannot corrupt a shared KV-cache.

## Deployment

Packaging for running the daemon as a background service lives in
[`deploy/`](../deploy/):

- **Docker / Compose** — `deploy/Dockerfile`, `deploy/docker-compose.yml`, and
  a `docker-compose.gpu.yml` override for CUDA hosts.
- **systemd** — `deploy/avp-server.service` for a bare-metal / VM install.

See [`deploy/README.md`](../deploy/README.md) for install steps. In short:

```bash
cd deploy && cp env.example .env && docker compose up -d --build      # Docker
sudo systemctl enable --now avp-server                               # systemd
```

## Security

The daemon loads whatever model ids it is asked for, and model downloads can be
large. Bind it to `127.0.0.1`, set `--token`, and pass `--allowed-models` to
restrict the set of loadable models in shared environments.
