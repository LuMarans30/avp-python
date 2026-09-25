"""FastMCP daemon exposing the latent service over MCP and plain HTTP.

The daemon is the process that owns the models.  It offers two surfaces:

* **MCP** at ``/mcp`` — for MCP clients.  Tools: ``latent_think``,
  ``latent_generate``, ``latent_status``, ``latent_release``.
* **Plain JSON/HTTP** — for the Pi extension (Pi has no MCP client) and any
  simple caller.  Routes: ``POST /api/latent_think``,
  ``POST /api/latent_generate``, ``GET /api/status``, ``POST /api/latent_release``,
  and an unauthenticated ``GET /health`` for lifecycle probes.  When a bearer
  token is configured it guards every route except ``/health`` — the MCP surface
  at ``/mcp`` included.

Requires the optional dependency: ``pip install "avp[mcp]"``.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
from typing import Any

from ..errors import AVPError, ConfigurationError
from .service import LatentService, ServerConfig, ServerError

logger = logging.getLogger("avp.server")

# Maps service error codes to HTTP status codes.
_STATUS_BY_CODE = {
    "context_not_found": 404,
    "source_model_unavailable": 409,
    "model_in_use": 409,
    "cross_model_disabled": 400,
    "context_model_mismatch": 400,
    "context_not_continuable": 400,
    "invalid_request": 400,
    "server_error": 500,
}


def _error_response(exc: Exception) -> Any:
    """Translate a service/AVP error into a JSON HTTP response."""
    from starlette.responses import JSONResponse

    if isinstance(exc, ServerError):
        status = _STATUS_BY_CODE.get(exc.code, 500)
        return JSONResponse(
            {"ok": False, **exc.to_dict()}, status_code=status
        )
    if isinstance(exc, ConfigurationError):
        return JSONResponse(
            {"ok": False, "error": str(exc), "code": "invalid_request"},
            status_code=400,
        )
    if isinstance(exc, AVPError):
        return JSONResponse(
            {"ok": False, "error": str(exc), "code": "server_error"},
            status_code=500,
        )
    logger.exception("unhandled error in latent server")
    return JSONResponse(
        {"ok": False, "error": str(exc), "code": "server_error"},
        status_code=500,
    )


# Paths that stay reachable without a token so probes need no credentials.
_PUBLIC_PATHS = frozenset({"/health"})


def _unauthorized() -> Any:
    from starlette.responses import JSONResponse

    return JSONResponse(
        {"ok": False, "error": "unauthorized", "code": "unauthorized"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


class BearerAuthMiddleware:
    """Require ``Authorization: Bearer <token>`` on every route but ``/health``.

    Wrapping the whole ASGI app (rather than checking inside each route) keeps
    the MCP surface and the JSON mirror behind one perimeter.  Without it, a
    configured token would protect only ``/api/*`` and leave ``/mcp`` — the more
    capable surface, since it can run inference and load models — open.
    ``/health`` stays public so container/orchestrator probes need no secrets.
    """

    def __init__(
        self,
        app: Any,
        token: str | None = None,
        public_paths: frozenset[str] = _PUBLIC_PATHS,
    ) -> None:
        self.app = app
        self.token = token or ""
        self.public_paths = public_paths

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or not self.token
            or scope.get("path") in self.public_paths
        ):
            await self.app(scope, receive, send)
            return
        header = dict(scope.get("headers") or []).get(b"authorization", b"")
        scheme, _, value = header.decode("latin-1").partition(" ")
        if (
            scheme.lower() != "bearer"
            or not value
            or not secrets.compare_digest(value, self.token)
        ):
            await _unauthorized()(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_mcp(service: LatentService, name: str = "avp-latent") -> Any:
    """Build a :class:`fastmcp.FastMCP` app wrapping ``service``.

    Raises:
        ImportError: If ``fastmcp`` is not installed.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised via CLI
        raise ImportError(
            "fastmcp is required for the latent server. "
            'Install with: pip install "avp[mcp]"'
        ) from exc

    from starlette.concurrency import run_in_threadpool
    from starlette.responses import JSONResponse

    mcp = FastMCP(name)

    # ------------------------------------------------------------------
    # MCP tool surface (sync functions run in FastMCP's thread pool)
    # ------------------------------------------------------------------

    @mcp.tool
    def latent_think(
        prompt: str,
        model: str | None = None,
        steps: int | None = None,
        output: str = "auto",
        ttl: float | None = None,
    ) -> dict:
        """Think about a prompt on a model and return a reusable context_id.

        The heavy KV-cache/hidden-state stays resident in the daemon under the
        returned ``context_id``. Pass that id to latent_generate to continue in
        latent space instead of text. Use output="hidden_state" to skip the
        KV-cache and save VRAM when the next model will differ.
        """
        return service.latent_think(
            prompt, model=model, steps=steps, output=output, ttl=ttl
        )

    @mcp.tool
    def latent_generate(
        prompt: str,
        model: str | None = None,
        context_id: str | None = None,
        steps: int = 0,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        store_context: bool = False,
        ttl: float | None = None,
    ) -> dict:
        """Generate text, optionally conditioned on a context_id from latent_think.

        Same-model contexts reuse the full KV-cache; contexts from a different
        model are projected through the Rosetta Stone path in-process. Omit
        context_id and pass steps>0 to think-and-generate on the target model,
        or leave both unset for plain text generation.
        """
        return service.latent_generate(
            prompt,
            model=model,
            context_id=context_id,
            steps=steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            store_context=store_context,
            ttl=ttl,
        )

    @mcp.tool
    def latent_status() -> dict:
        """Report loaded models and resident latent contexts."""
        return service.status()

    @mcp.tool
    def latent_release(context_id: str) -> dict:
        """Release a resident context, freeing its VRAM."""
        return service.release(context_id)

    # ------------------------------------------------------------------
    # Plain HTTP surface (used by the Pi extension)
    # ------------------------------------------------------------------

    async def _read_json(request: Any) -> dict:
        try:
            body = await request.json()
        except Exception as exc:
            raise ConfigurationError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise ConfigurationError("Request body must be a JSON object")
        return body

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Any) -> Any:
        # Liveness only: no model/registry detail for unauthenticated callers.
        # Rich status lives behind the token at /api/status.
        return JSONResponse({"status": "ok"})

    @mcp.custom_route("/api/status", methods=["GET"])
    async def api_status(request: Any) -> Any:
        return JSONResponse({"ok": True, **service.status()})

    @mcp.custom_route("/api/latent_think", methods=["POST"])
    async def api_latent_think(request: Any) -> Any:
        try:
            body = await _read_json(request)
            result = await run_in_threadpool(
                service.latent_think,
                body.get("prompt", ""),
                model=body.get("model"),
                steps=body.get("steps"),
                output=body.get("output", "auto"),
                ttl=body.get("ttl"),
                context_id=body.get("context_id"),
            )
        except (ServerError, AVPError) as exc:
            return _error_response(exc)
        return JSONResponse({"ok": True, **result})

    @mcp.custom_route("/api/latent_generate", methods=["POST"])
    async def api_latent_generate(request: Any) -> Any:
        try:
            body = await _read_json(request)
            result = await run_in_threadpool(
                service.latent_generate,
                body.get("prompt", ""),
                model=body.get("model"),
                context_id=body.get("context_id"),
                steps=body.get("steps", 0),
                max_new_tokens=body.get("max_new_tokens"),
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.95),
                do_sample=body.get("do_sample", True),
                store_context=body.get("store_context", False),
                ttl=body.get("ttl"),
            )
        except (ServerError, AVPError) as exc:
            return _error_response(exc)
        return JSONResponse({"ok": True, **result})

    @mcp.custom_route("/api/latent_release", methods=["POST"])
    async def api_latent_release(request: Any) -> Any:
        try:
            body = await _read_json(request)
            context_id = body.get("context_id", "")
            if not context_id:
                raise ConfigurationError("context_id is required")
            result = service.release(context_id)
        except (ServerError, AVPError) as exc:
            return _error_response(exc)
        return JSONResponse({"ok": True, **result})

    return mcp


def _asgi_auth_middleware(service: LatentService) -> list[Any]:
    """ASGI middleware enforcing the bearer token on every route but ``/health``.

    Must be passed to :meth:`~fastmcp.FastMCP.http_app` / ``run()``: FastMCP's
    *constructor* ``middleware`` argument takes MCP protocol middleware
    (``__call__(context, call_next)``), which never sees HTTP routes, so it
    cannot guard ``/mcp``.
    """
    from starlette.middleware import Middleware

    return [
        Middleware(
            BearerAuthMiddleware,
            token=service.config.token,
            public_paths=_PUBLIC_PATHS,
        )
    ]


def create_app(service: LatentService, path: str = "/mcp") -> Any:
    """Return a Starlette ASGI app (MCP + plain HTTP) for embedding/tests."""
    return create_mcp(service).http_app(
        path=path, middleware=_asgi_auth_middleware(service)
    )


def _config_from_args(args: argparse.Namespace) -> ServerConfig:
    allowed = tuple(
        m.strip() for m in (args.allowed_models or "").split(",") if m.strip()
    )
    return ServerConfig(
        default_model=args.default_model or "",
        device=args.device,
        backend=args.backend,
        default_ttl=args.ttl,
        max_contexts=args.max_contexts,
        default_steps=args.default_steps,
        max_new_tokens=args.max_new_tokens,
        cross_model=not args.no_cross_model,
        allowed_models=allowed,
        token=args.token or os.environ.get("AVP_DAEMON_TOKEN"),
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        projection_method=args.projection_method,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avp-server",
        description="Persistent AVP latent server (MCP + HTTP).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--transport",
        default="http",
        choices=["http", "streamable-http", "sse", "stdio"],
    )
    parser.add_argument("--default-model", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", default="hf", choices=["hf", "ollama", "llamacpp"])
    parser.add_argument("--ttl", type=float, default=300.0)
    parser.add_argument("--max-contexts", type=int, default=8)
    parser.add_argument("--default-steps", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="llama.cpp layers to offload to GPU (0 = CPU only).",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help="llama.cpp context window size.",
    )
    parser.add_argument(
        "--projection-method",
        default="vocab_overlap",
        choices=["vocab_overlap", "linear"],
        help="GGUF cross-model projection method (default: vocab_overlap).",
    )
    parser.add_argument("--allowed-models", default="")
    parser.add_argument("--token", default="")
    parser.add_argument(
        "--preload",
        default="",
        help="Comma-separated model ids to load at startup.",
    )
    parser.add_argument("--no-cross-model", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _is_loopback(host: str) -> bool:
    """Whether ``host`` only accepts local connections."""
    return host in {"localhost", "::1"} or host.startswith("127.")


def warn_if_publicly_bound(host: str, token: str | None) -> None:
    """Warn when binding off-loopback with no token: both surfaces are open."""
    if token or _is_loopback(host):
        return
    logger.warning(
        "Binding %s without a token: /mcp and /api/* accept unauthenticated "
        "requests. Set --token/AVP_DAEMON_TOKEN, or bind 127.0.0.1.",
        host,
    )


def serve(
    config: ServerConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    transport: str = "http",
    preload: str = "",
    log_level: str = "INFO",
) -> None:
    """Build and run the latent server (blocking)."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = LatentService(config or ServerConfig())
    for model_id in (m.strip() for m in preload.split(",") if m.strip()):
        logger.info("preloading model %s", model_id)
        service.manager.get(model_id)

    warn_if_publicly_bound(host, service.config.token)
    mcp = create_mcp(service)
    logger.info(
        "starting AVP latent server (transport=%s host=%s port=%d, models=%d)",
        transport, host, port, len(service.manager.loaded_models()),
    )
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Pass the selected transport through: FastMCP accepts "http"
        # (streamable HTTP), "streamable-http" and "sse".  Every non-stdio
        # value used to be silently downgraded to "http".
        mcp.run(
            transport=transport,
            host=host,
            port=port,
            middleware=_asgi_auth_middleware(service),
        )


def main(argv: list | None = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    serve(
        config=_config_from_args(args),
        host=args.host,
        port=args.port,
        transport=args.transport,
        preload=args.preload,
        log_level=args.log_level,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
