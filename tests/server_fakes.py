"""Fake connectors for exercising the latent server without torch/GPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from avp.types import OutputType, PayloadType


@dataclass
class FakeIdentity:
    model_family: str = "fake"
    model_id: str = "fake"
    model_hash: str = ""
    hidden_dim: int = 8
    num_layers: int = 2
    num_kv_heads: int = 1
    head_dim: int = 8


@dataclass
class FakeContext:
    """Duck-typed stand-in for :class:`avp.context.AVPContext`."""

    model_hash: str
    num_steps: int = 0
    seq_len: int = 0
    past_key_values: Any = None
    last_hidden_state: Any = None
    model_family: str = "fake"
    hidden_dim: int = 8
    num_layers: int = 2

    @property
    def payload_type(self) -> PayloadType:
        return (
            PayloadType.KV_CACHE
            if self.past_key_values is not None
            else PayloadType.HIDDEN_STATE
        )


class FakeConnector:
    """Records calls and returns deterministic contexts/text."""

    def __init__(self, model_id: str, model_hash: str) -> None:
        self.model_id = model_id
        self.model_hash = model_hash
        self.think_calls: list[dict] = []
        self.generate_calls: list[dict] = []

    def get_model_identity(self) -> FakeIdentity:
        return FakeIdentity(model_id=self.model_id, model_hash=self.model_hash)

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def dtype(self) -> str:
        return "float32"

    @property
    def can_think(self) -> bool:
        return True

    def think(
        self,
        prompt: str,
        steps: int = 20,
        context: FakeContext | None = None,
        output: OutputType = OutputType.AUTO,
    ) -> FakeContext:
        self.think_calls.append(
            {"prompt": prompt, "steps": steps, "output": output, "continued": context is not None}
        )
        resolved = output.resolve()
        return FakeContext(
            model_hash=self.model_hash,
            num_steps=(context.num_steps if context else 0) + steps,
            seq_len=len(prompt) + steps,
            past_key_values=object() if resolved == PayloadType.KV_CACHE else None,
            last_hidden_state=object(),
        )

    def generate(
        self,
        prompt: str,
        context: FakeContext | None = None,
        source: FakeConnector | None = None,
        cross_model: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "has_context": context is not None,
                "source": source.model_id if source else None,
                "cross_model": cross_model,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
            }
        )
        origin = "cross" if source is not None else ("latent" if context else "text")
        return f"[{self.model_id}:{origin}] {prompt}"


def build_manager(connectors: dict) -> Any:
    """ConnectorManager whose factory serves pre-built fakes."""
    from avp.server.manager import ConnectorManager

    return ConnectorManager(factory=lambda model_id: connectors[model_id])
