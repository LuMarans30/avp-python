"""Tests for the latent server's VRAM context registry."""

import itertools
import time
from types import SimpleNamespace

import pytest
from server_fakes import FakeContext

from avp.server.registry import ContextRegistry


def _ctx(model_hash="h", seq_len=10):
    return FakeContext(model_hash=model_hash, seq_len=seq_len, past_key_values=object())


def test_put_get_remove_roundtrip():
    reg = ContextRegistry(default_ttl=60, max_entries=4)
    entry = reg.put(_ctx("abc"), source_model_id="m1")

    assert entry.context_id
    assert entry.source_model_hash == "abc"
    assert entry.source_model_id == "m1"
    assert reg.active_count == 1

    fetched = reg.get(entry.context_id)
    assert fetched is entry
    assert fetched.num_steps == 0
    assert fetched.payload_type == "KV_CACHE"

    assert reg.remove(entry.context_id) is True
    assert reg.get(entry.context_id) is None
    assert reg.remove(entry.context_id) is False


def test_registry_uses_context_model_hash_by_default():
    reg = ContextRegistry()
    ctx = _ctx("derived-hash")
    entry = reg.put(ctx, source_model_id="m")
    assert entry.source_model_hash == "derived-hash"


def test_ttl_expiry():
    reg = ContextRegistry(default_ttl=0.02, max_entries=4)
    entry = reg.put(_ctx("h"))
    assert reg.get(entry.context_id) is not None
    time.sleep(0.05)
    assert reg.get(entry.context_id) is None
    assert reg.active_count == 0


def test_lru_eviction_on_capacity(monkeypatch):
    # Windows' time.time() has ~1 ms resolution, so four back-to-back calls can
    # all share one timestamp; min() then breaks the recency tie by insertion
    # order and evicts the wrong entry.  Use a monotonic fake clock so recency
    # is unambiguous on every platform.
    ticks = itertools.count(1000)
    monkeypatch.setattr(
        "avp.server.registry.time", SimpleNamespace(time=lambda: float(next(ticks)))
    )

    reg = ContextRegistry(default_ttl=60, max_entries=2)
    a = reg.put(_ctx("a"), source_model_id="a")
    b = reg.put(_ctx("b"), source_model_id="b")
    # Touch a so b becomes least-recently-used.
    reg.get(a.context_id)
    c = reg.put(_ctx("c"), source_model_id="c")

    assert reg.get(b.context_id) is None
    assert reg.get(a.context_id) is not None
    assert reg.get(c.context_id) is not None
    assert reg.active_count == 2


def test_expired_entries_evicted_before_lru():
    reg = ContextRegistry(default_ttl=0.02, max_entries=2)
    old = reg.put(_ctx("old"), source_model_id="old")
    time.sleep(0.05)
    reg.put(_ctx("new"), source_model_id="new")
    # The expired entry should be gone without evicting the live one.
    assert reg.get(old.context_id) is None
    assert reg.active_count == 1


def test_stats_and_keys():
    reg = ContextRegistry(default_ttl=60, max_entries=4)
    entry = reg.put(_ctx("h", seq_len=7), source_model_id="m")
    stats = reg.stats()
    assert stats["active_count"] == 1
    assert stats["max_entries"] == 4
    assert stats["contexts"][0]["context_id"] == entry.context_id
    assert stats["contexts"][0]["seq_len"] == 7
    assert entry.context_id in reg


def test_invalid_construction():
    with pytest.raises(ValueError):
        ContextRegistry(default_ttl=0)
    with pytest.raises(ValueError):
        ContextRegistry(max_entries=0)
