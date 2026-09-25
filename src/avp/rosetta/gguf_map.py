"""Persistent cross-model projection maps for GGUF models.

Numpy-only.  Builds a projection between two GGUF models from their token
embeddings and caches it per model-hash pair, so the expensive embedding
dequantization runs **once** instead of on every cross-model call.

Two projection methods are supported:

``vocab_overlap``
    Softmax over the shared-token logits, i.e. a convex combination of the
    target embeddings of the shared tokens (the existing behaviour).
``linear``
    A ridge-regression map ``W: D_src -> D_tgt`` fit on the paired source and
    target embeddings of the shared tokens, applied with
    :func:`avp.rosetta.project.apply_cross_model_projection`.

Maps are held in a process-wide cache and persisted as ``.npz`` under
``$AVP_CACHE_DIR/gguf_maps`` (default ``~/.avp/gguf_maps``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibrate import compute_vocab_overlap_from_dicts

logger = logging.getLogger(__name__)

_MAP_DIR = Path(os.environ.get("AVP_CACHE_DIR", str(Path.home() / ".avp"))) / "gguf_maps"


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _normalize(x: np.ndarray, target_norm: float) -> np.ndarray:
    """L2-normalize rows of ``x`` to ``target_norm``."""
    norm = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
    return (x * (target_norm / norm)).astype(np.float32)


def fit_linear_map(
    src_shared: Any,
    tgt_shared: Any,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``W, bias`` so that ``src_shared @ W + bias ~= tgt_shared``.

    Uses ridge regression via the normal equations, which is cheap for the
    shared-token counts involved (a few thousand rows, hidden dims <= a few
    thousand).  The bias column is not regularized.

    Args:
        src_shared: Source embeddings for shared tokens ``[N, D_src]``.
        tgt_shared: Target embeddings for shared tokens ``[N, D_tgt]``.
        ridge: L2 regularization strength.

    Returns:
        ``(w_map [D_src, D_tgt], bias [D_tgt])`` as float32.
    """
    a = np.asarray(src_shared, dtype=np.float64)
    b = np.asarray(tgt_shared, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError(
            f"shapes must be [N, D_src] and [N, D_tgt]; got {a.shape} and {b.shape}"
        )
    n, d_src = a.shape
    a1 = np.concatenate([a, np.ones((n, 1), dtype=np.float64)], axis=1)
    reg = ridge * np.eye(d_src + 1, dtype=np.float64)
    reg[-1, -1] = 0.0  # do not regularize the bias term
    solution = np.linalg.solve(a1.T @ a1 + reg, a1.T @ b)
    return (
        np.ascontiguousarray(solution[:d_src], dtype=np.float32),
        np.ascontiguousarray(solution[d_src], dtype=np.float32),
    )


@dataclass
class GGUFProjectionMap:
    """A cached projection between two GGUF models.

    Stores both the shared-token embedding rows (for ``vocab_overlap``) and a
    fitted linear map (for ``linear``), so the method can be chosen at
    projection time without rebuilding.
    """

    source_hash: str
    target_hash: str
    source_dim: int
    target_dim: int
    target_norm: float
    overlap_count: int
    overlap_ratio: float
    src_shared: np.ndarray | None = None  # [N, D_src]
    tgt_shared: np.ndarray | None = None  # [N, D_tgt]
    w_map: np.ndarray | None = None       # [D_src, D_tgt]
    bias: np.ndarray | None = None        # [D_tgt]

    def project(
        self,
        hidden: Any,
        method: str = "vocab_overlap",
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Project a source hidden state (or states) into target space.

        Args:
            hidden: Array-like ``[..., D_src]``.
            method: ``"vocab_overlap"`` or ``"linear"``.
            temperature: Softmax temperature for ``vocab_overlap``.

        Returns:
            numpy array ``[..., D_tgt]`` normalized to ``target_norm``.
        """
        h = np.asarray(hidden, dtype=np.float32)
        if method == "linear" and self.w_map is not None:
            projected = h @ self.w_map
            if self.bias is not None:
                projected = projected + self.bias
            return _normalize(projected, self.target_norm)

        if self.src_shared is None or self.tgt_shared is None:
            raise ValueError("vocab_overlap map is missing shared embeddings")
        logits = h @ self.src_shared.T
        probs = _softmax(logits / temperature, axis=-1)
        projected = probs @ self.tgt_shared
        return _normalize(projected, self.target_norm)

    # --- Persistence ---

    def save(self, map_dir: Path | None = None) -> Path:
        """Persist the map to ``map_dir`` (default ``$AVP_CACHE_DIR/gguf_maps``)."""
        directory = map_dir or _MAP_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.source_hash[:16]}_{self.target_hash[:16]}.npz"

        meta = {
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "target_norm": self.target_norm,
            "overlap_count": self.overlap_count,
            "overlap_ratio": self.overlap_ratio,
        }
        arrays: dict[str, Any] = {"meta": np.array(json.dumps(meta))}
        for name in ("src_shared", "tgt_shared", "w_map", "bias"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        np.savez(path, **arrays)
        logger.info("gguf_map: saved %s", path)
        return path

    @classmethod
    def load(
        cls,
        source_hash: str,
        target_hash: str,
        map_dir: Path | None = None,
    ) -> GGUFProjectionMap | None:
        """Load a persisted map, or return ``None`` if absent/corrupt."""
        directory = map_dir or _MAP_DIR
        path = directory / f"{source_hash[:16]}_{target_hash[:16]}.npz"
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                kwargs: dict[str, Any] = {}
                for name in ("src_shared", "tgt_shared", "w_map", "bias"):
                    if name in data:
                        kwargs[name] = data[name]
            return cls(**meta, **kwargs)
        except Exception as exc:  # noqa: BLE001 - corrupt cache is non-fatal
            logger.warning("gguf_map: failed to load %s: %s", path, exc)
            return None


# --- Process-wide cache ---

_CACHE: dict[tuple[str, str], GGUFProjectionMap] = {}
_CACHE_LOCK = threading.Lock()


def _cache_key(source_hash: str, target_hash: str) -> tuple[str, str]:
    return (source_hash or "?", target_hash or "?")


def clear_cache() -> None:
    """Drop the process-wide projection-map cache."""
    with _CACHE_LOCK:
        _CACHE.clear()


def build_map(
    source_path: str,
    target_path: str,
    source_hash: str,
    target_hash: str,
    source_vocab: dict,
    target_vocab: dict,
    min_overlap: int = 100,
    ridge: float = 1e-3,
) -> GGUFProjectionMap | None:
    """Build a projection map from two GGUF files.

    Dequantizes each embedding matrix once, slices the shared rows, then frees
    the full matrices to keep peak RAM bounded.

    Returns:
        A :class:`GGUFProjectionMap`, or ``None`` when the vocabularies share
        fewer than ``min_overlap`` tokens.
    """
    from ..connectors._llamacpp_compat import extract_gguf_embedding_weights

    overlap = compute_vocab_overlap_from_dicts(source_vocab, target_vocab, min_overlap)
    if overlap is None:
        return None
    src_indices, tgt_indices, shared = overlap

    src_full = extract_gguf_embedding_weights(source_path)
    src_shared = np.ascontiguousarray(src_full[src_indices], dtype=np.float32)
    del src_full

    tgt_full = extract_gguf_embedding_weights(target_path)
    target_norm = float(np.linalg.norm(tgt_full, axis=1).mean())
    tgt_shared = np.ascontiguousarray(tgt_full[tgt_indices], dtype=np.float32)
    del tgt_full

    w_map, bias = fit_linear_map(src_shared, tgt_shared, ridge=ridge)

    overlap_ratio = float(len(shared) / min(len(source_vocab), len(target_vocab)))
    result = GGUFProjectionMap(
        source_hash=source_hash,
        target_hash=target_hash,
        source_dim=int(src_shared.shape[1]),
        target_dim=int(tgt_shared.shape[1]),
        target_norm=target_norm,
        overlap_count=len(shared),
        overlap_ratio=overlap_ratio,
        src_shared=src_shared,
        tgt_shared=tgt_shared,
        w_map=w_map,
        bias=bias,
    )
    logger.info(
        "gguf_map: built %s->%s, %d shared tokens (%.1f%%)",
        source_hash[:12], target_hash[:12], result.overlap_count, overlap_ratio * 100,
    )
    return result


def get_or_build_map(
    source_path: str,
    target_path: str,
    source_hash: str,
    target_hash: str,
    source_vocab: dict,
    target_vocab: dict,
    min_overlap: int = 100,
    ridge: float = 1e-3,
) -> GGUFProjectionMap | None:
    """Return a cached map (memory → disk → build) for a model pair."""
    key = _cache_key(source_hash, target_hash)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    loaded = GGUFProjectionMap.load(source_hash, target_hash)
    if loaded is not None:
        with _CACHE_LOCK:
            _CACHE[key] = loaded
        logger.info("gguf_map: loaded %s->%s from disk", source_hash[:12], target_hash[:12])
        return loaded

    built = build_map(
        source_path, target_path, source_hash, target_hash,
        source_vocab, target_vocab, min_overlap=min_overlap, ridge=ridge,
    )
    if built is not None:
        with _CACHE_LOCK:
            _CACHE[key] = built
        try:
            built.save()
        except Exception as exc:  # noqa: BLE001 - caching is best effort
            logger.warning("gguf_map: could not persist map: %s", exc)
    return built
