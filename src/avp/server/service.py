"""Latent context service: the logic behind ``latent_think``/``latent_generate``.

This module is deliberately free of any web/MCP dependency so it can be unit
tested with fake connectors and reused by other transports.

Routing
-------
:meth:`LatentService.latent_think` runs ``connector.think()`` and stores the
resulting :class:`avp.context.AVPContext` in a :class:`ContextRegistry`,
returning a short ``context_id``.  :meth:`LatentService.latent_generate`
resolves that handle and decides:

* **same model** → ``target.generate(prompt, context=ctx)`` (full KV-cache),
* **cross model** → ``target.generate(prompt, context=ctx, source=src,
  cross_model=True)`` (in-process Rosetta Stone projection),
* **no context** → plain ``target.generate(prompt)``.

No hidden state is ever serialized to the caller.  Cross-model projection is
only possible because the daemon holds both connectors in one process.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..errors import AVPError, ConfigurationError
from ..types import OutputType
from .manager import ConnectorManager
from .registry import ContextRegistry, StoredContext

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class ServerError(AVPError):
    """Structured error raised by the latent server.

    Carries a machine-readable ``code`` so transports can map failures to HTTP
    status codes or MCP error payloads.
    """

    def __init__(self, message: str, code: str = "server_error") -> None:
        self.code = code
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {"error": str(self), "code": self.code}


@dataclass
class ServerConfig:
    """Runtime configuration for :class:`LatentService`.

    Attributes:
        default_model: Model used when a tool call omits ``model``.
        device: Device for the default HuggingFace factory.
        backend: ``"hf"``, ``"ollama"``, or ``"llamacpp"``.
        default_ttl: Default context TTL in seconds.
        max_contexts: Max live contexts retained in VRAM.
        default_steps: Default latent thinking steps.
        max_new_tokens: Hard cap on generated tokens (protects VRAM/latency).
        cross_model: Default for cross-model projection when a context comes
            from a different model.
        allowed_models: Optional allowlist of model ids.  When non-empty, any
            other model is rejected.  Empty means "allow any model".
        token: Optional bearer token required by the HTTP surface.
        n_gpu_layers: llama.cpp layers to offload (``0`` = CPU only).  ``None``
            leaves the connector default (``-1``, all layers).
        n_ctx: llama.cpp context window.  ``None`` leaves the connector default.
        projection_method: GGUF cross-model projection, ``"vocab_overlap"`` or
            ``"linear"``.
    """

    default_model: str = ""
    device: str | None = None
    backend: str = "hf"
    default_ttl: float = 300.0
    max_contexts: int = 8
    default_steps: int = 20
    max_new_tokens: int = 2048
    cross_model: bool = True
    allowed_models: Sequence[str] = field(default_factory=tuple)
    token: str | None = None
    n_gpu_layers: int | None = None
    n_ctx: int | None = None
    projection_method: str = "vocab_overlap"


class LatentService:
    """Coordinates connector loading, latent thinking, and generation."""

    def __init__(
        self,
        config: ServerConfig | None = None,
        manager: ConnectorManager | None = None,
        registry: ContextRegistry | None = None,
    ) -> None:
        self.config = config or ServerConfig()
        self.manager = manager or ConnectorManager(
            device=self.config.device,
            backend=self.config.backend,
            backend_kwargs={
                "n_gpu_layers": self.config.n_gpu_layers,
                "n_ctx": self.config.n_ctx,
                "projection_method": self.config.projection_method,
            },
        )
        self.registry = registry or ContextRegistry(
            default_ttl=self.config.default_ttl,
            max_entries=self.config.max_contexts,
        )

    # --- Public tool operations ---

    def latent_think(
        self,
        prompt: str,
        model: str | None = None,
        steps: int | None = None,
        output: str = "auto",
        ttl: float | None = None,
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """Run ``think()`` and register the resulting latent context.

        Args:
            prompt: Prompt to think about.
            model: Model identifier (defaults to ``config.default_model``).
            steps: Latent thinking steps (defaults to ``config.default_steps``).
            output: ``"auto"``, ``"kv_cache"``, or ``"hidden_state"``.
            ttl: Override context TTL in seconds.
            context_id: Optional prior context to continue from (must be the
                same model).  The prior entry is replaced by the new one.

        Returns:
            Summary dict including ``context_id``.
        """
        self._require_prompt(prompt, "latent_think")
        model_id = self._resolve_model_id(model)
        steps = self._resolve_steps(steps)
        output_type = self._resolve_output(output)
        ttl = self._resolve_ttl(ttl)

        prior = self._load_prior_context(context_id, model_id)

        connector = self.manager.get(model_id)
        lock_ids = [model_id]
        if prior is not None:
            lock_ids.append(prior.source_model_id)
        with self.manager.locked(*lock_ids):
            context = connector.think(
                prompt, steps=steps, context=prior.context if prior else None,
                output=output_type,
            )

        entry = self.registry.put(
            context,
            source_model_id=model_id,
            source_model_hash=self._model_hash(connector),
            ttl=ttl,
        )
        if prior is not None:
            self.registry.remove(prior.context_id)

        logger.info(
            "latent_think: model=%s steps=%d context=%s seq_len=%d",
            model_id, steps, entry.context_id, entry.seq_len,
        )
        return entry.to_public()

    def latent_generate(
        self,
        prompt: str,
        model: str | None = None,
        context_id: str | None = None,
        steps: int = 0,
        store_context: bool = False,
        ttl: float | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> dict[str, Any]:
        """Generate text, optionally conditioned on a registered context.

        If ``context_id`` is given it takes precedence over ``steps``.  When no
        context is supplied and ``steps > 0``, the target model thinks first
        (in-process same-model transfer).  With neither, this is plain text
        generation.

        Args:
            prompt: Prompt for generation.
            model: Target model identifier.
            context_id: Handle from :meth:`latent_think`.
            steps: Think steps when no ``context_id`` is supplied.
            store_context: When ``steps > 0``, also store the context produced
                by the think step and return its ``context_id``.
            ttl: TTL for a context stored via ``store_context``.
            max_new_tokens: Generation cap; clamped to ``config.max_new_tokens``.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            do_sample: Sample (True) or greedy decode (False).

        Returns:
            Dict with ``text`` plus routing metadata.
        """
        self._require_prompt(prompt, "latent_generate")
        target_id = self._resolve_model_id(model)
        target = self.manager.get(target_id)
        target_hash = self._model_hash(target)

        max_tokens = self._resolve_max_tokens(max_new_tokens)
        ttl = self._resolve_ttl(ttl)
        temperature, top_p = self._resolve_sampling(temperature, top_p)

        context: Any = None
        source_connector: Any = None
        source_id: str | None = None
        mode = "text"
        stored: StoredContext | None = None
        new_entry: StoredContext | None = None

        if context_id:
            stored = self.registry.get(context_id)
            if stored is None:
                raise ServerError(
                    f"Unknown or expired context_id {context_id!r}. "
                    "It may have reached its TTL or been evicted.",
                    code="context_not_found",
                )
            context = stored.context
            same_model = (
                stored.source_model_hash == target_hash
                if stored.source_model_hash and target_hash
                else stored.source_model_id == target_id
            )
            if same_model:
                mode = "same_model"
            else:
                mode = "cross_model"
                if not self.config.cross_model:
                    raise ServerError(
                        "Context belongs to a different model and cross-model "
                        "transfer is disabled.",
                        code="cross_model_disabled",
                    )
                source_connector = self.manager.get_by_hash(stored.source_model_hash)
                if source_connector is None:
                    raise ServerError(
                        f"Context was produced by model {stored.source_model_id!r} "
                        "(hash "
                        f"{stored.source_model_hash[:16]}), which is not loaded. "
                        "Reload that model or re-run latent_think on it.",
                        code="source_model_unavailable",
                    )
                source_id = stored.source_model_id
        elif steps and steps > 0:
            steps = self._resolve_steps(steps)
            with self.manager.locked(target_id):
                context = target.think(prompt, steps=steps, output=OutputType.AUTO)
            mode = "same_model"
            if store_context:
                new_entry = self.registry.put(
                    context,
                    source_model_id=target_id,
                    source_model_hash=target_hash,
                    ttl=ttl,
                )

        lock_ids = [target_id]
        if source_id:
            lock_ids.append(source_id)
        with self.manager.locked(*lock_ids):
            if mode == "cross_model":
                text = target.generate(
                    prompt,
                    context=context,
                    source=source_connector,
                    cross_model=True,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
            elif mode == "same_model":
                text = target.generate(
                    prompt,
                    context=context,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
            else:
                text = target.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )

        logger.info(
            "latent_generate: model=%s mode=%s context=%s chars=%d",
            target_id, mode, context_id or "-", len(text),
        )
        return {
            "text": text,
            "model": target_id,
            "model_hash": target_hash,
            "mode": mode,
            "used_context": context is not None,
            "context_id": context_id,
            "source_model": source_id,
            "stored_context_id": new_entry.context_id if new_entry else None,
            "max_new_tokens": max_tokens,
        }

    def release(self, context_id: str) -> dict[str, Any]:
        """Drop a stored context, freeing its VRAM.  Returns whether it existed."""
        removed = self.registry.remove(context_id)
        return {"context_id": context_id, "released": bool(removed)}

    def status(self) -> dict[str, Any]:
        """Return daemon status: models, registry, and config summary."""
        return {
            "ready": True,
            "default_model": self.config.default_model,
            "backend": self.config.backend,
            "device": self.config.device,
            "cross_model": self.config.cross_model,
            "max_new_tokens": self.config.max_new_tokens,
            "models": self.manager.loaded_models(),
            "registry": self.registry.stats(),
        }

    def unload_model(self, model_id: str) -> dict[str, Any]:
        """Unload a model, refusing if live contexts still reference it."""
        referenced = [
            entry.context_id
            for entry in (self.registry.get(k) for k in self.registry)
            if entry is not None and entry.source_model_id == model_id
        ]
        if referenced:
            raise ServerError(
                f"Cannot unload {model_id!r}: {len(referenced)} live context(s) "
                f"still reference it. Release them first.",
                code="model_in_use",
            )
        return {"model_id": model_id, "unloaded": self.manager.unload(model_id)}

    # --- Internals ---

    @staticmethod
    def _require_prompt(prompt: Any, tool: str) -> None:
        if not isinstance(prompt, str) or not prompt:
            raise ConfigurationError(f"{tool} requires a non-empty prompt string")

    def _resolve_model_id(self, model: str | None) -> str:
        model_id = model or self.config.default_model
        if not model_id:
            raise ConfigurationError(
                "No model specified and no default_model configured."
            )
        allowed = self.config.allowed_models
        if allowed and model_id not in allowed:
            raise ConfigurationError(
                f"Model {model_id!r} is not in the allowlist. "
                f"Allowed: {list(allowed)}"
            )
        return model_id

    def _resolve_steps(self, steps: int | None) -> int:
        if steps is None:
            return self.config.default_steps
        try:
            resolved = int(steps)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"steps must be an integer, got {steps!r}"
            ) from exc
        if resolved < 0:
            raise ConfigurationError(f"steps must be >= 0, got {resolved}")
        return resolved

    def _resolve_output(self, output: Any) -> OutputType:
        if isinstance(output, OutputType):
            return output
        if not isinstance(output, str):
            raise ConfigurationError(
                f"output must be a string, got {type(output).__name__}"
            )
        try:
            return OutputType(output.lower())
        except ValueError as exc:
            raise ConfigurationError(
                f"Unknown output {output!r}. Use one of: "
                f"{[o.value for o in OutputType]}"
            ) from exc

    def _resolve_max_tokens(self, max_new_tokens: int | None) -> int:
        cap = self.config.max_new_tokens
        if max_new_tokens is None:
            return cap
        try:
            value = int(max_new_tokens)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"max_new_tokens must be an integer, got {max_new_tokens!r}"
            ) from exc
        if value < 1:
            raise ConfigurationError(f"max_new_tokens must be >= 1, got {value}")
        return min(value, cap)

    def _resolve_ttl(self, ttl: float | None) -> float | None:
        """Validate a per-request TTL so bad input becomes a 400, not a 500."""
        if ttl is None:
            return None
        try:
            value = float(ttl)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"ttl must be a number, got {ttl!r}") from exc
        if value <= 0:
            raise ConfigurationError(f"ttl must be > 0, got {value}")
        return value

    def _resolve_sampling(
        self, temperature: float, top_p: float
    ) -> tuple[float, float]:
        """Validate sampling parameters so bad input becomes a 400, not a 500."""
        try:
            temp = float(temperature)
            nucleus = float(top_p)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "temperature and top_p must be numbers, got "
                f"{temperature!r} and {top_p!r}"
            ) from exc
        if temp < 0:
            raise ConfigurationError(f"temperature must be >= 0, got {temp}")
        if not 0 < nucleus <= 1:
            raise ConfigurationError(f"top_p must be in (0, 1], got {nucleus}")
        return temp, nucleus

    def _load_prior_context(
        self, context_id: str | None, model_id: str
    ) -> StoredContext | None:
        if not context_id:
            return None
        entry = self.registry.get(context_id)
        if entry is None:
            raise ServerError(
                f"Unknown or expired context_id {context_id!r}.",
                code="context_not_found",
            )
        if entry.payload_type == "HIDDEN_STATE":
            raise ServerError(
                "Cannot continue from a hidden_state-only context: it has no "
                "KV-cache to extend. Re-run latent_think with "
                "output='kv_cache'.",
                code="context_not_continuable",
            )
        connector = self.manager.get(model_id)
        if entry.source_model_hash != self._model_hash(connector):
            raise ServerError(
                "Continuation context belongs to a different model; cross-model "
                "continuation is not supported. Start a fresh latent_think.",
                code="context_model_mismatch",
            )
        return entry

    @staticmethod
    def _model_hash(connector: Any) -> str:
        """Read the public model hash from a connector."""
        identity = connector.get_model_identity()
        return getattr(identity, "model_hash", "") or ""

    def close(self) -> None:
        """Release all contexts and connectors."""
        self.registry.clear()
        self.manager.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
