# Deploying the AVP latent server

Two supported deployment paths for `avp-server`:

- **Docker / Compose** — `Dockerfile`, `docker-compose.yml`, GPU override.
- **systemd** — `avp-server.service` for a bare-metal / VM install.

Both expose the same MCP (`/mcp`) and plain-HTTP (`/api/*`) surfaces. See
[`docs/LATENT_SERVER.md`](../docs/LATENT_SERVER.md) for the API.

## Docker

The image builds this checkout and installs `avp[mcp]`. The default CPU build
pulls a CPU-only torch wheel (`AVP_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`)
to keep the image small; the GPU override clears that so the CUDA base's torch
is used.

```bash
cd deploy
cp .env.example .env
$EDITOR .env                 # at minimum set AVP_DAEMON_TOKEN
docker compose up -d --build
docker compose logs -f avp-server
```

The default bind is `127.0.0.1:8765`. Model weights and AVP maps persist in the
`avp-data` named volume (`/data`).

### GPU

Use a base image that already ships CUDA-enabled torch and layer the GPU
override so the CPU path stays unchanged:

```bash
cd deploy
AVP_BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Requires the NVIDIA Container Toolkit on the host.

### Verify

```bash
curl -s localhost:8765/health
curl -s -H "Authorization: Bearer $AVP_DAEMON_TOKEN" localhost:8765/api/status
```

## systemd

Assumes a venv at `/opt/avp/venv` and a system user `avp`.

```bash
# 1. Service user
sudo useradd --system --home /var/lib/avp --shell /usr/sbin/nologin avp

# 2. Virtualenv with the daemon
sudo python3 -m venv /opt/avp/venv
sudo /opt/avp/venv/bin/pip install --upgrade pip
sudo /opt/avp/venv/bin/pip install "avp[mcp]"     # or: pip install /path/to/avp-python[mcp]

# 3. Unit + config
sudo install -m 644 deploy/avp-server.service /etc/systemd/system/avp-server.service
sudo mkdir -p /etc/avp
sudo install -m 600 deploy/avp-server.env.example /etc/avp/avp-server.env
sudo $EDITOR /etc/avp/avp-server.env              # token + AVP_SERVER_ARGS

# 4. Start
sudo systemctl daemon-reload
sudo systemctl enable --now avp-server
systemctl status avp-server
journalctl -u avp-server -f
```

`AVP_SERVER_ARGS` in `/etc/avp/avp-server.env` carries the CLI arguments, so the
model set can change without editing the unit. Model and map caches live under
`/var/lib/avp` (systemd `StateDirectory=avp`), which is the only writable path
under the unit's hardening profile.

For GPUs, uncomment `SupplementaryGroups=` in the unit and set your distro's
device group(s), typically `video render`. The line is commented by default
because systemd fails if a named group does not exist. Device is auto-detected;
add `--device cuda` to `AVP_SERVER_ARGS` to force it.

## Security notes

- Bind to `127.0.0.1` unless a reverse proxy provides TLS and auth.
- Always set `AVP_DAEMON_TOKEN`. It guards every route except `/health`, so
  `/mcp` is covered too; keep the port private regardless.
- Use `--allowed-models` to restrict which model ids the daemon may load.
- A 7B KV-cache is ~390 MB; size `--max-contexts` and `--ttl` to your VRAM.
