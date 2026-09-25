"""Dependency-free tests for GGUF cross-family projection helpers.

These deliberately live outside ``tests/test_llamacpp_connector.py``: that
module carries a module-level ``pytestmark = skipif(HAS_LLAMACPP)`` for the
import-guard test, which skips *every* test in the file whenever
llama-cpp-python is installed — including tests that never need it.
"""

import pytest


class TestEmbeddingScale:
    """Input-embedding scale factors applied to injected soft prompts."""

    def test_gemma_family_scales_by_sqrt_n_embd(self):
        from avp.connectors.llamacpp import LlamaCppConnector

        assert LlamaCppConnector._embedding_scale_for("gemma4", 2560) == pytest.approx(2560 ** 0.5)
        assert LlamaCppConnector._embedding_scale_for("gemma", 2048) == pytest.approx(2048 ** 0.5)

    def test_other_families_are_unscaled(self):
        from avp.connectors.llamacpp import LlamaCppConnector

        assert LlamaCppConnector._embedding_scale_for("llama", 4096) == 1.0
        assert LlamaCppConnector._embedding_scale_for("bailingmoe3", 1536) == 1.0

    def test_missing_family_or_dim_is_unscaled(self):
        from avp.connectors.llamacpp import LlamaCppConnector

        assert LlamaCppConnector._embedding_scale_for("", 2560) == 1.0
        assert LlamaCppConnector._embedding_scale_for("gemma", 0) == 1.0


class TestVocabOverlapComputation:
    """Shared-token index computation for cross-family projection."""

    def test_shared_indices_are_aligned(self):
        from avp.rosetta.calibrate import compute_vocab_overlap_from_dicts

        src = {"a": 0, "b": 1, "c": 2}
        tgt = {"b": 5, "c": 7, "d": 9}
        result = compute_vocab_overlap_from_dicts(src, tgt, min_overlap=2)
        assert result is not None
        src_idx, tgt_idx, shared = result
        assert shared == ["b", "c"]
        assert list(src_idx) == [1, 2]
        assert list(tgt_idx) == [5, 7]

    def test_below_min_overlap_returns_none(self):
        from avp.rosetta.calibrate import compute_vocab_overlap_from_dicts

        assert compute_vocab_overlap_from_dicts({"a": 0}, {"a": 0}, min_overlap=100) is None

    def test_empty_vocab_returns_none(self):
        from avp.rosetta.calibrate import compute_vocab_overlap_from_dicts

        assert compute_vocab_overlap_from_dicts({}, {"a": 0}) is None
