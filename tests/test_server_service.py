"""Tests for the latent server routing logic (same-model / cross-model)."""

import pytest
from server_fakes import FakeConnector, build_manager

from avp.errors import ConfigurationError
from avp.server.registry import ContextRegistry
from avp.server.service import LatentService, ServerConfig, ServerError
from avp.types import OutputType


@pytest.fixture
def connectors():
    return {
        "big": FakeConnector("big", "hash-big"),
        "small": FakeConnector("small", "hash-small"),
    }


@pytest.fixture
def service(connectors):
    config = ServerConfig(default_model="big", default_ttl=60, max_contexts=4)
    return LatentService(
        config=config,
        manager=build_manager(connectors),
        registry=ContextRegistry(default_ttl=60, max_entries=4),
    )


def test_think_stores_context_and_returns_id(service, connectors):
    result = service.latent_think("analyze this", model="big", steps=5)

    assert result["context_id"]
    assert result["source_model_id"] == "big"
    assert result["source_model_hash"] == "hash-big"
    assert result["num_steps"] == 5
    assert result["payload_type"] == "KV_CACHE"

    call = connectors["big"].think_calls[0]
    assert call["steps"] == 5
    assert call["output"] == OutputType.AUTO


def test_think_honours_hidden_state_output(service):
    result = service.latent_think("analyze", model="big", steps=3, output="hidden_state")
    assert result["payload_type"] == "HIDDEN_STATE"
    assert result["size_bytes"] == 0


def test_think_uses_default_model(service, connectors):
    result = service.latent_think("hello", steps=2)
    assert result["source_model_id"] == "big"


def test_generate_same_model(service, connectors):
    think = service.latent_think("analyze", model="big", steps=4)
    gen = service.latent_generate("solve", model="big", context_id=think["context_id"])

    assert gen["mode"] == "same_model"
    assert gen["used_context"] is True
    assert gen["source_model"] is None
    assert gen["text"].startswith("[big:latent]")

    call = connectors["big"].generate_calls[-1]
    assert call["has_context"] is True
    assert call["source"] is None
    assert call["cross_model"] is False


def test_generate_cross_model_passes_source(service, connectors):
    think = service.latent_think("analyze", model="big", steps=4)
    gen = service.latent_generate("solve", model="small", context_id=think["context_id"])

    assert gen["mode"] == "cross_model"
    assert gen["source_model"] == "big"
    assert gen["text"].startswith("[small:cross]")

    call = connectors["small"].generate_calls[-1]
    assert call["source"] == "big"
    assert call["cross_model"] is True
    assert call["has_context"] is True


def test_generate_plain_text_without_context(service, connectors):
    gen = service.latent_generate("just write", model="big")
    assert gen["mode"] == "text"
    assert gen["used_context"] is False
    assert gen["text"].startswith("[big:text]")


def test_generate_think_and_store(service, connectors):
    gen = service.latent_generate(
        "question", model="big", steps=3, store_context=True
    )
    assert gen["mode"] == "same_model"
    assert gen["stored_context_id"]
    assert service.registry.get(gen["stored_context_id"]) is not None
    assert connectors["big"].think_calls[-1]["steps"] == 3


def test_context_not_found(service):
    with pytest.raises(ServerError) as exc:
        service.latent_generate("solve", model="big", context_id="missing")
    assert exc.value.code == "context_not_found"


def test_cross_model_disabled(service, connectors):
    service.config.cross_model = False
    think = service.latent_think("analyze", model="big", steps=2)
    with pytest.raises(ServerError) as exc:
        service.latent_generate("solve", model="small", context_id=think["context_id"])
    assert exc.value.code == "cross_model_disabled"


def test_source_model_unavailable(service, connectors):
    think = service.latent_think("analyze", model="big", steps=2)
    service.manager.unload("big")
    with pytest.raises(ServerError) as exc:
        service.latent_generate("solve", model="small", context_id=think["context_id"])
    assert exc.value.code == "source_model_unavailable"


def test_max_new_tokens_clamped(service, connectors):
    service.config.max_new_tokens = 100
    gen = service.latent_generate("x", model="big", max_new_tokens=10_000)
    assert gen["max_new_tokens"] == 100
    assert connectors["big"].generate_calls[-1]["max_new_tokens"] == 100


def test_think_continuation_replaces_entry(service, connectors):
    first = service.latent_think("part 1", model="big", steps=2)
    second = service.latent_think(
        "part 2", model="big", steps=2, context_id=first["context_id"]
    )
    assert second["context_id"] != first["context_id"]
    assert service.registry.get(first["context_id"]) is None
    assert connectors["big"].think_calls[-1]["continued"] is True


def test_think_continuation_model_mismatch(service):
    first = service.latent_think("part 1", model="big", steps=2)
    with pytest.raises(ServerError) as exc:
        service.latent_think("part 2", model="small", context_id=first["context_id"])
    assert exc.value.code == "context_model_mismatch"


def test_cannot_continue_hidden_state_context(service):
    first = service.latent_think("part 1", model="big", steps=2, output="hidden_state")
    with pytest.raises(ServerError) as exc:
        service.latent_think("part 2", model="big", context_id=first["context_id"])
    assert exc.value.code == "context_not_continuable"


def test_allowlist_rejects_unknown_model(service):
    service.config.allowed_models = ("big",)
    with pytest.raises(ConfigurationError):
        service.latent_think("x", model="small")


def test_release_and_status(service):
    think = service.latent_think("x", model="big", steps=2)
    assert service.status()["registry"]["active_count"] == 1

    released = service.release(think["context_id"])
    assert released["released"] is True
    assert service.release(think["context_id"])["released"] is False
    assert service.status()["registry"]["active_count"] == 0


def test_unload_model_refuses_when_in_use(service):
    service.latent_think("x", model="big", steps=2)
    with pytest.raises(ServerError) as exc:
        service.unload_model("big")
    assert exc.value.code == "model_in_use"


def test_invalid_prompt_and_output(service):
    with pytest.raises(ConfigurationError):
        service.latent_think("", model="big")
    with pytest.raises(ConfigurationError):
        service.latent_think("x", model="big", output="bogus")
