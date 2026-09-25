# syntax=docker/dockerfile:1
#
# AVP latent server image (FastMCP + plain HTTP).
#
# CPU (default):
#   docker build -f deploy/Dockerfile -t avp-server .
#
# GPU: use a base image that already ships a CUDA-enabled torch, e.g.
#   docker build -f deploy/Dockerfile \
#     --build-arg BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
#     -t avp-server:gpu .
# then run with `--gpus all` (or the compose GPU override).
#
# The image installs the current checkout, so it always matches this tree.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Optional: install torch from a specific wheel index before the package, e.g.
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# keeps the CPU image small. Leave empty when the base image already ships the
# torch build you want (e.g. a CUDA base).
ARG TORCH_INDEX_URL=""

LABEL org.opencontainers.image.title="avp-server" \
      org.opencontainers.image.description="Persistent AVP latent server (MCP + HTTP)" \
      org.opencontainers.image.source="https://github.com/VectorArc/avp-python" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/huggingface \
    AVP_CACHE_DIR=/data/avp

WORKDIR /opt/avp

# Copy package metadata + source, then install. (src layout: hatchling builds
# the wheel from src/avp.)
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && \
    if [ -n "${TORCH_INDEX_URL}" ]; then \
        python -m pip install torch --index-url "${TORCH_INDEX_URL}"; \
    fi && \
    python -m pip install ".[mcp]"

# Non-root runtime user and a writable model / context cache.
RUN useradd --uid 10001 --create-home --home-dir /home/avp --shell /usr/sbin/nologin avp && \
    mkdir -p /data/huggingface /data/avp && \
    chown -R avp:avp /data /opt/avp
USER avp

EXPOSE 8765
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["python", "-m", "avp.server.daemon"]
CMD ["--host", "0.0.0.0", "--port", "8765"]
