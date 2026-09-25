"""AVP latent server — expose ``think()``/``generate()`` over MCP and HTTP.

This package wraps :mod:`avp` in a persistent, model-resident process so that
agents (Pi, MCP clients, or any HTTP caller) can share latent context without
reloading models on every call.

The design follows the two transfer modes described in the AVP README, but
implemented entirely **in-process**:

``same_model``
    The context produced by ``think()`` is kept alive in a thread-safe
    :class:`~avp.server.registry.ContextRegistry` (the "VRAM registry") and
    referenced by a short ``context_id``.

``cross_model``
    Because the daemon holds *both* connectors, the Rosetta Stone projection
    runs in-process: the target connector is called with ``source=<source
    connector>, cross_model=True``.  No hidden-state serialization is needed
    (and ``AVPContext.to_bytes()`` does not support hidden-state-only contexts
    anyway).

The package is import-safe without the optional ``fastmcp`` dependency; only
:mod:`avp.server.daemon` imports it.
"""

from .manager import ConnectorManager
from .registry import ContextRegistry, StoredContext
from .service import LatentService, ServerConfig, ServerError

__all__ = [
    "ConnectorManager",
    "ContextRegistry",
    "LatentService",
    "ServerConfig",
    "ServerError",
    "StoredContext",
]
