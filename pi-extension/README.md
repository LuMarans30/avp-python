# pi-avp-latent

A [Pi](https://pi.dev) extension that exposes two tools backed by a persistent
[AVP](../../README.md) latent daemon:

- **`latent_think`** — think about a prompt on a self-hosted model, keep the
  latent state resident, return a short `context_id`.
- **`latent_generate`** — generate text, optionally continuing from a
  `context_id` (same-model KV-cache or cross-model Rosetta Stone projection).

Pi has no MCP client, so the extension talks to the daemon's plain-HTTP mirror
(`/api/*`) rather than the MCP endpoint. Agents hand each other `context_id`s
instead of long text; tensor state never enters the model context.

## 1. Start the daemon

```bash
pip install "avp[mcp]"

avp-server \
  --default-model Qwen/Qwen2.5-7B-Instruct \
  --preload Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.2-3B-Instruct \
  --port 8765
```

The Pi extension can also autostart it (see below).

## 2. Load the extension

```bash
# one-off
pi -e ./pi-extension

# install into settings
pi install ./pi-extension
```

## 3. Configure (optional)

| Variable | Default | Meaning |
|----------|---------|---------|
| `AVP_DAEMON_URL` | `http://127.0.0.1:8765` | Daemon base URL |
| `AVP_DAEMON_TOKEN` | – | Bearer token, matching `--token` |
| `AVP_DAEMON_AUTOSTART` | `0` | `1` to spawn `avp-server` on session start |
| `AVP_DAEMON_STOP_ON_EXIT` | `0` | `1` to kill an autostarted daemon on shutdown |
| `AVP_DAEMON_CMD` | `avp-server` | Autostart command line |

Example:

```bash
export AVP_DAEMON_AUTOSTART=1
export AVP_DAEMON_URL=http://127.0.0.1:8765
pi -e ./pi-extension
```

## Tools

```
latent_think(prompt, model?, steps?, output?, ttl?, context_id?) -> context_id
latent_generate(prompt, model?, context_id?, steps?, max_new_tokens?,
                temperature?, top_p?, do_sample?, store_context?) -> text
```

`/avp-status` shows loaded models and resident context count.

## Develop

```bash
npm install
npx tsc --noEmit
```

Pi supplies the `@earendil-works/*` packages at runtime; they are declared as
`peerDependencies`.
