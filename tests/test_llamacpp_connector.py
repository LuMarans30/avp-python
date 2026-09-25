"""Tests for LlamaCppConnector (mock-based, no model required)."""

import pytest

try:
    import llama_cpp  # noqa: F401

    HAS_LLAMACPP = True
except ImportError:
    HAS_LLAMACPP = False

pytestmark = [
    pytest.mark.skipif(HAS_LLAMACPP, reason="Tests for when llama-cpp-python is NOT installed"),
]


class TestLlamaCppImportGuard:
    """Test that the connector raises ImportError without llama-cpp-python."""

    def test_import_raises_without_llamacpp(self):
        from avp.connectors.llamacpp import LlamaCppConnector
        with pytest.raises(ImportError, match="llama-cpp-python"):
            LlamaCppConnector("nonexistent.gguf")


class TestBlockCountExtraction:
    """GGUF layer-count parsing (no model required)."""

    def test_prefers_exact_block_count_over_leading_dense(self):
        from avp.connectors.llamacpp import LlamaCppConnector

        meta = {
            "general.architecture": "bailingmoe3",
            "bailingmoe3.leading_dense_block_count": "1",
            "bailingmoe3.block_count": "24",
        }
        assert LlamaCppConnector._extract_block_count(meta) == 24

    def test_falls_back_to_generic_key(self):
        from avp.connectors.llamacpp import LlamaCppConnector

        assert LlamaCppConnector._extract_block_count({"block_count": "12"}) == 12

    def test_returns_none_when_absent(self):
        from avp.connectors.llamacpp import LlamaCppConnector

        assert LlamaCppConnector._extract_block_count({}) is None

    def test_handles_non_numeric_value(self):
        from avp.connectors.llamacpp import LlamaCppConnector

        assert LlamaCppConnector._extract_block_count({"block_count": "lots"}) is None


