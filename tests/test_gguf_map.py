"""Tests for cached GGUF cross-model projection maps (numpy only)."""

import numpy as np
import pytest

from avp.rosetta.gguf_map import GGUFProjectionMap, fit_linear_map


def test_fit_linear_map_recovers_known_transform():
    rng = np.random.default_rng(0)
    n, d_src, d_tgt = 400, 16, 10
    a = rng.standard_normal((n, d_src)).astype(np.float32)
    w_true = rng.standard_normal((d_src, d_tgt)).astype(np.float32)
    b_true = rng.standard_normal(d_tgt).astype(np.float32)
    b = a @ w_true + b_true

    w, bias = fit_linear_map(a, b, ridge=1e-6)
    assert np.allclose(w, w_true, atol=1e-3)
    assert np.allclose(bias, b_true, atol=1e-3)


def test_fit_linear_map_rejects_bad_shapes():
    with pytest.raises(ValueError):
        fit_linear_map(np.zeros((3, 4)), np.zeros((2, 5)))


def test_linear_projection_normalizes_to_target_norm():
    src = np.eye(4, dtype=np.float32)
    tgt = np.eye(4, dtype=np.float32) * 2.0
    w, bias = fit_linear_map(src, tgt, ridge=1e-6)
    proj = GGUFProjectionMap(
        source_hash="s",
        target_hash="t",
        source_dim=4,
        target_dim=4,
        target_norm=2.0,
        overlap_count=4,
        overlap_ratio=1.0,
        src_shared=src,
        tgt_shared=tgt,
        w_map=w,
        bias=bias,
    )
    out = proj.project(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), method="linear")
    assert out.shape == (1, 4)
    assert np.isclose(np.linalg.norm(out), 2.0, atol=1e-4)


def test_vocab_overlap_projection_shape_and_norm():
    n, d_src, d_tgt = 5, 3, 4
    src = np.eye(n, d_src, dtype=np.float32)
    tgt = np.arange(n * d_tgt, dtype=np.float32).reshape(n, d_tgt)
    proj = GGUFProjectionMap(
        source_hash="s",
        target_hash="t",
        source_dim=d_src,
        target_dim=d_tgt,
        target_norm=1.0,
        overlap_count=n,
        overlap_ratio=1.0,
        src_shared=src,
        tgt_shared=tgt,
    )
    out = proj.project(np.zeros((1, d_src), dtype=np.float32), method="vocab_overlap")
    assert out.shape == (1, d_tgt)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-4)


def test_multi_state_projection_preserves_leading_dims():
    src = np.eye(3, dtype=np.float32)
    tgt = np.eye(3, dtype=np.float32)
    proj = GGUFProjectionMap(
        source_hash="s",
        target_hash="t",
        source_dim=3,
        target_dim=3,
        target_norm=1.0,
        overlap_count=3,
        overlap_ratio=1.0,
        src_shared=src,
        tgt_shared=tgt,
    )
    states = np.eye(3, dtype=np.float32)  # [S=3, D=3]
    out = proj.project(states, method="vocab_overlap")
    assert out.shape == (3, 3)
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-4)


def test_persistence_roundtrip(tmp_path):
    proj = GGUFProjectionMap(
        source_hash="abcdef0123456789",
        target_hash="fedcba9876543210",
        source_dim=4,
        target_dim=3,
        target_norm=1.5,
        overlap_count=7,
        overlap_ratio=0.5,
        src_shared=np.ones((7, 4), dtype=np.float32),
        tgt_shared=np.ones((7, 3), dtype=np.float32),
        w_map=np.ones((4, 3), dtype=np.float32),
        bias=np.zeros(3, dtype=np.float32),
    )
    path = proj.save(map_dir=tmp_path)
    assert path.exists()

    loaded = GGUFProjectionMap.load(
        "abcdef0123456789", "fedcba9876543210", map_dir=tmp_path
    )
    assert loaded is not None
    assert loaded.overlap_count == 7
    assert loaded.target_norm == pytest.approx(1.5)
    assert np.allclose(loaded.src_shared, proj.src_shared)
    assert np.allclose(loaded.w_map, proj.w_map)
    assert np.allclose(loaded.bias, proj.bias)


def test_load_missing_returns_none(tmp_path):
    assert GGUFProjectionMap.load("nope", "nada", map_dir=tmp_path) is None
