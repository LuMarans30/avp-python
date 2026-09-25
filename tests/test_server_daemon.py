"""Tests for the latent server's HTTP surface.

These run without torch/GPU by injecting fake connectors.  The MCP transport is
exercised only insofar as it hosts the plain-HTTP routes the Pi extension uses.
"""

import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("httpx")

from server_fakes import FakeConnector, build_manager
from starlette.testclient import TestClient

from avp.server.daemon import create_app
from avp.server.registry import ContextRegistry
from avp.server.service import LatentService, ServerConfig


def _service(token=None):
    connectors = {
        "big": FakeConnector("big", "hash-big"),
        "small": FakeConnector("small", "hash-small"),
    }
    config = ServerConfig(
        default_model="big", default_ttl=60, max_contexts=4, token=token
    )
    return LatentService(
        config=config,
        manager=build_manager(connectors),
        registry=ContextRegistry(default_ttl=60, max_entries=4),
    )


@pytest.fixture
def client():
    app = create_app(_service())
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_open(client):
    """Liveness probes need no credentials and must leak no model detail."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_latent_think_and_generate_roundtrip(client):
    think = client.post(
        "/api/latent_think", json={"prompt": "analyze", "model": "big", "steps": 4}
    )
    assert think.status_code == 200
    context_id = think.json()["context_id"]

    gen = client.post(
        "/api/latent_generate",
        json={"prompt": "solve", "model": "big", "context_id": context_id},
    )
    assert gen.status_code == 200
    body = gen.json()
    assert body["mode"] == "same_model"
    assert body["text"].startswith("[big:latent]")


def test_cross_model_roundtrip(client):
    think = client.post(
        "/api/latent_think", json={"prompt": "analyze", "model": "big", "steps": 4}
    )
    context_id = think.json()["context_id"]
    gen = client.post(
        "/api/latent_generate",
        json={"prompt": "solve", "model": "small", "context_id": context_id},
    )
    assert gen.status_code == 200
    assert gen.json()["mode"] == "cross_model"
    assert gen.json()["source_model"] == "big"


def test_status_and_release(client):
    think = client.post("/api/latent_think", json={"prompt": "x", "model": "big"})
    context_id = think.json()["context_id"]

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["registry"]["active_count"] == 1

    release = client.post("/api/latent_release", json={"context_id": context_id})
    assert release.status_code == 200
    assert release.json()["released"] is True


def test_missing_context_returns_404(client):
    resp = client.post(
        "/api/latent_generate", json={"prompt": "x", "model": "big", "context_id": "nope"}
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "context_not_found"


def test_invalid_body_returns_400(client):
    resp = client.post("/api/latent_think", json={"prompt": ""})
    assert resp.status_code == 400


def test_bearer_token_enforced():
    app = create_app(_service(token="s3cret"))
    with TestClient(app) as client:
        assert client.get("/api/status").status_code == 401
        assert client.get("/health").status_code == 200
        ok = client.get("/api/status", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        bad = client.get("/api/status", headers={"Authorization": "Bearer wrong"})
        assert bad.status_code == 401


def test_health_stays_open_with_a_token():
    app = create_app(_service(token="s3cret"))
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_mcp_surface_requires_token():
    """The token must cover /mcp too, not just the /api/* mirror."""
    app = create_app(_service(token="s3cret"))
    with TestClient(app) as client:
        assert client.post("/mcp", json={}).status_code == 401
        assert client.get("/mcp").status_code == 401
        assert client.get("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
        # The token unlocks the surface; the MCP handshake itself may still 4xx.
        authorized = client.post(
            "/mcp", json={}, headers={"Authorization": "Bearer s3cret"}
        )
        assert authorized.status_code != 401


def test_mcp_surface_open_without_token(client):
    """With no token configured the middleware is a pass-through."""
    assert client.post("/mcp", json={}).status_code != 401
