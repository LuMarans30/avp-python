# Cross-Model Transfer: Findings and Open Problems

This note records an investigation into whether AVP's cross-model path can be
generalized into a *learned translator* so that arbitrary model pairs
communicate. The short version: **the payload and the injection are solvable;
translation fidelity is the open problem, and reaching usable fidelity is a
training problem, not a plumbing problem.**

Everything below was measured with two tiny GGUF models on CPU (Ling-3.0-tiny,
`bailingmoe3`; Gemma-4-E4B, `gemma4`). The harness is in
[`experiments/`](../experiments/).

## 1. The payload decides what can be transferred

Same-model test (Ling): inject a candidate representation as a soft prompt,
then ask for a fact that appears only in the original prompt.

| payload injected | recalls the fact? |
|---|---|
| full KV-cache (default `think`) | yes |
| final latent hidden state (raw) | no |
| final latent hidden state (rescaled) | no |
| hidden state → LM head → embedding | no |
| prompt **input embeddings** | yes |

The final hidden state — even after 20 latent steps — is a compressed *reasoning*
vector; it does not carry the prompt's facts. No projection can recover
information that the payload never contained. So:

- **facts/content** need a token-level payload (input embeddings or tokens),
- **reasoning state** is what hidden states transfer, and it is only useful for
  structured tasks.

## 2. Injection is model-specific (and had a bug)

llama.cpp applies architecture-specific scaling to token embeddings
(Gemma/T5 use `sqrt(n_embd)`) when building from token ids, but **not** when
embeddings are injected via `batch.embd`. Gemma could not read even its *own*
token embeddings until this scale was applied:

```
raw        norm 1.15  -> "I do not have enough information…"
× sqrt(d)  norm 58.3  -> "NIGHTFALL"
```

Fixed in `LlamaCppConnector._embedding_scale_for` / `_generate_with_embedding`.

Injection is also **template-structure sensitive**: Gemma reads its own
*chat-templated* embeddings but not the same content as a bare prefix. A
translated soft prompt must be assembled *inside the target's chat structure*.

## 3. Translation fidelity is the bottleneck

### Token-embedding domain

25,070 shared-token pairs, 80/20 split:

| method | held-out cosine | top-1 match |
|---|---|---|
| vocab_overlap | 0.363 | 0.02% (chance) |
| linear (ridge) | 0.519 | **97.65%** |
| MLP (no VSP) | 0.532 peak / 0.515 final | 97.55% |
| MLP + VSP (λ=1, 10) | 0.493 / 0.428 | – |

The linear map identifies the correct target embedding almost perfectly, yet
sits ~60° off in absolute terms. The MLP does not beat it; VSP hurts. The
supervised shared-token signal is exhausted.

### Contextual (sentence) domain

300 wiki texts, last-token hidden states, 80/20 split:

| method | held-out cosine | top-1 match (60 candidates) |
|---|---|---|
| ridge | 0.472 | – |
| MLP | 0.525 | **41.7%** |
| chance | – | 1.7% |

Same ~0.53 cosine ceiling, but the contextual embeddings *identify* the correct
held-out text 41.7% of the time (25× chance) — more signal than the raw cosine
suggests. Still **inconclusive** at this scale: only 300 samples, the MLP
overfits, and the embedding is weak because `think(steps=0)` takes the hidden
state at the *assistant-turn marker*, not a content summary.

## 4. What the literature suggests

vec2vec (_Harnessing the Universal Geometry of Embeddings_, arXiv 2505.12540)
learns a universal latent with space-specific adapters and a shared backbone,
trained with adversarial + cycle-consistency + vector-space-preservation losses
on **unpaired** samples. It reports cosine up to ~0.92 across backbones — but on
**encoder sentence embeddings**, with ~11M samples, and its own ablations show
every component matters (and GAN training is unstable).

Our prototypes use paired shared tokens and a small MLP. They establish the
ceiling of the *cheap* approach, not of the full method.

## 5. A design for a general translator

```
   model A ── A_A ──┐                       ┌── B_B ── model B
   model C ── A_C ──┤   shared backbone T   ├── B_D ── model D
   model E ── A_E ──┘   (universal latent) └── B_F ── model F
```

- **Hub-and-spoke**: per-model adapters + one shared backbone → O(N) adapters
  instead of O(N²) pairwise maps.
- **`TranslatorRegistry`** keyed by model hash; auto-calibrate on first contact
  from unpaired samples; cache under `$AVP_CACHE_DIR/translators/`.
- **Validation is mandatory**: held-out cosine *and* a downstream probe. Cosine
  alone can look fine while behavior is useless (we observed exactly this).
- **Fallback ladder**: translated soft prompt → latent state → text/JSON.
- **Task routing**: latent for reasoning, token/soft-prompt for content.

## 6. Limits and risks

- API-only models expose no embeddings → text fallback only.
- Fidelity degrades with architectural/domain distance; the registry needs a
  "cannot translate reliably" verdict.
- It is a learned decoder between representations, not shared understanding.
- Automatic translation is also automatic embedding inversion; treat latent
  transfer as an explicit, auditable opt-in.

## 7. Go/no-go gate before building further

Only invest in the full translator if a scaled experiment clears all of:

1. content-token pooling (not the chat-template marker) for the embeddings,
2. ≥100k unpaired samples per model,
3. the full vec2vec objective (adversarial + cycle + VSP),
4. held-out cosine ≥ ~0.8 **and** a downstream gain on the soft-prompt probe.

Otherwise, keep the text/JSON fallback and restrict latent transfer to
same-family pairs and structured reasoning.

## Bugs fixed during this investigation

- **Embedding scaling** for injected soft prompts (Gemma/T5) — `0c8935d`.
- **`do_sample` leaking** into `llama_cpp.Llama.__call__` on fallback paths —
  `87602df`.
