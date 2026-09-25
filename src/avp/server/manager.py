"""Connector lifecycle management for the latent server.

Loading a transformer is slow and VRAM-hungry, so connectors are created once
and reused for the lifetime of the daemon.  This manager also owns the
per-model inference locks that serialize GPU work: a single CUDA device cannot
safely run two overlapping ``generate()`` calls that mutate a shared KV-cache.

Model identity (including the config hash used to route cross-model contexts)
is read through the public :meth:`avp.connectors.base.EngineConnector.get_model_identity`
API — never through private attributes.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from ..errors import ConfigurationError

logger = logging.getLogger(__name__)

ConnectorFactory = Callable[[str], Any]


def _default_factory(
    backend: str,
    device: str | None,
    backend_kwargs: dict[str, Any] | None = None,
) -> ConnectorFactory:
    """Build a ``model_id -> connector`` factory for the requested backend.

    Args:
        backend: ``"hf"``, ``"ollama"``, or ``"llamacpp"``.
        device: Device passed to the HuggingFace factory.
        backend_kwargs: Extra keyword arguments for the engine factory (e.g.
            ``n_gpu_layers`` / ``n_ctx`` for llama.cpp).  ``None`` values are
            dropped.
    """
    extra = {k: v for k, v in (backend_kwargs or {}).items() if v is not None}

    def _make(model_id: str) -> Any:
        normalized = backend.lower()
        if normalized in ("hf", "huggingface"):
            from ..connectors.huggingface import HuggingFaceConnector

            return HuggingFaceConnector.from_pretrained(model_id, device=device)
        if normalized == "ollama":
            from ..connectors.ollama import OllamaConnector

            return OllamaConnector.from_ollama(model_id, **extra)
        if normalized in ("llamacpp", "llama.cpp"):
            from ..connectors.llamacpp import LlamaCppConnector

            return LlamaCppConnector.from_pretrained(model_id, **extra)
        raise ConfigurationError(
            f"Unknown backend {backend!r}. Supported: hf, ollama, llamacpp."
        )

    return _make


class ConnectorManager:
    """Cache connectors by model id and serialize per-model inference.

    Args:
        factory: ``model_id -> EngineConnector`` callable.  Defaults to a
            :class:`avp.HuggingFaceConnector` factory.  Inject a stub in tests
            to avoid loading real weights.
        device: Device passed to the default HuggingFace factory.
        backend: ``"hf"`` (default), ``"ollama"``, or ``"llamacpp"``.  Ignored
            when ``factory`` is supplied.
        backend_kwargs: Extra keyword arguments forwarded to the backend's
            factory (e.g. ``{"n_gpu_layers": 0, "n_ctx": 2048}``).
    """

    def __init__(
        self,
        factory: ConnectorFactory | None = None,
        device: str | None = None,
        backend: str = "hf",
        backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._factory: ConnectorFactory = factory or _default_factory(
            backend, device, backend_kwargs
        )
        self._device = device
        self._backend = backend
        self._connectors: dict[str, Any] = {}
        self._hash_to_model: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}
        # Only one model may load at a time; loading is the VRAM spike.
        self._load_lock = threading.Lock()
        self._meta_lock = threading.Lock()

    # --- Lookup / loading ---

    def get(self, model_id: str) -> Any:
        """Return the connector for ``model_id``, loading it on first use.

        Raises:
            ConfigurationError: If the factory returns an unusable connector
                or loading fails.
        """
        if not model_id:
            raise ConfigurationError("model_id must be a non-empty string")

        with self._meta_lock:
            existing = self._connectors.get(model_id)
        if existing is not None:
            return existing

        with self._load_lock:
            # Re-check under the load lock: another thread may have loaded it.
            with self._meta_lock:
                existing = self._connectors.get(model_id)
            if existing is not None:
                return existing

            logger.info("manager: loading model %s (backend=%s)", model_id, self._backend)
            try:
                connector = self._factory(model_id)
            except Exception as exc:
                raise ConfigurationError(
                    f"Failed to load model {model_id!r}: {exc}"
                ) from exc

            identity = connector.get_model_identity()
            model_hash = getattr(identity, "model_hash", "") or ""

            with self._meta_lock:
                self._connectors[model_id] = connector
                if model_hash:
                    self._hash_to_model[model_hash] = model_id
                self._locks.setdefault(model_id, threading.Lock())
            logger.info(
                "manager: loaded %s (family=%s, hash=%s)",
                model_id,
                getattr(identity, "model_family", "?"),
                model_hash[:16] or "?",
            )
            return connector

    def get_by_hash(self, model_hash: str) -> Any | None:
        """Return the loaded connector whose config hash matches, if any."""
        if not model_hash:
            return None
        with self._meta_lock:
            model_id = self._hash_to_model.get(model_hash)
        return self._connectors.get(model_id) if model_id else None

    def model_id_for_hash(self, model_hash: str) -> str | None:
        """Return the model id registered for ``model_hash``, if any."""
        with self._meta_lock:
            return self._hash_to_model.get(model_hash)

    def lock_for(self, model_id: str) -> threading.Lock:
        """Return the inference lock for ``model_id`` (created on demand)."""
        with self._meta_lock:
            return self._locks.setdefault(model_id, threading.Lock())

    @contextmanager
    def locked(self, *model_ids: str) -> Iterator[None]:
        """Acquire inference locks for all given models in a stable order.

        Acquiring in sorted order prevents deadlock when two cross-model
        requests reference the same pair in opposite directions.
        """
        unique = sorted({m for m in model_ids if m})
        with ExitStack() as stack:
            for model_id in unique:
                stack.enter_context(self.lock_for(model_id))
            yield

    # --- Introspection / lifecycle ---

    def loaded_models(self) -> list[dict[str, Any]]:
        """Return JSON-serializable metadata for every loaded connector."""
        with self._meta_lock:
            items = list(self._connectors.items())
        result = []
        for model_id, connector in items:
            identity = connector.get_model_identity()
            result.append(
                {
                    "model_id": model_id,
                    "model_hash": getattr(identity, "model_hash", ""),
                    "model_family": getattr(identity, "model_family", ""),
                    "hidden_dim": getattr(identity, "hidden_dim", 0),
                    "num_layers": getattr(identity, "num_layers", 0),
                    "device": getattr(connector, "device", "unknown"),
                    "dtype": getattr(connector, "dtype", "unknown"),
                    "can_think": bool(getattr(connector, "can_think", False)),
                }
            )
        return result

    def unload(self, model_id: str) -> bool:
        """Drop a connector and its hash mapping.  Returns True if present.

        The caller is responsible for ensuring no contexts still reference the
        model; the service performs that check before calling this.
        """
        with self._meta_lock:
            connector = self._connectors.pop(model_id, None)
            if connector is None:
                return False
            identity = connector.get_model_identity()
            model_hash = getattr(identity, "model_hash", "")
            if model_hash and self._hash_to_model.get(model_hash) == model_id:
                del self._hash_to_model[model_hash]
            self._locks.pop(model_id, None)
        logger.info("manager: unloaded model %s", model_id)
        return True

    def clear(self) -> None:
        """Drop all connectors."""
        with self._meta_lock:
            self._connectors.clear()
            self._hash_to_model.clear()
            self._locks.clear()
