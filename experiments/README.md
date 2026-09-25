# Cross-model translator experiments

Prototypes behind [`docs/CROSS_MODEL_TRANSFER.md`](../docs/CROSS_MODEL_TRANSFER.md).
These are research harnesses, **not** part of the `avp` package.

## Requirements

```bash
pip install torch          # CPU is enough for the prototypes
pip install "avp[llamacpp]"  # for contextual_translator.py
```

Both scripts need a `gguf`/`llama-cpp-python` environment for the contextual
path. `token_translator.py` only needs the cached `.npz` map produced by
`avp.rosetta.gguf_map` (see `docs/LATENT_SERVER.md`).

## token_translator.py

Trains a small MLP `D_src -> D_tgt` on the shared-token embedding pairs stored
in a cached GGUF projection map, with an optional vector-space-preservation
(VSP) penalty, and compares held-out cosine / top-1 against the linear and
vocab-overlap baselines.

```bash
python experiments/token_translator.py \
  --map ~/.avp/gguf_maps/<src_hash16>_<tgt_hash16>.npz \
  --epochs 20 --vsp 0.0
```

Result (Ling→Gemma, 25,070 pairs): linear ≈ 0.52 cosine / 97.6% top-1; the MLP
does not beat it; VSP reduces alignment.

## contextual_translator.py

Collects last-token contextual embeddings from two GGUF models over a text
corpus, trains a translator on them, and reports the same metrics.

```bash
# collect (slow on CPU) then train
python experiments/contextual_translator.py \
  --corpus /tmp/corpus.json \
  --ling  /path/to/Ling-3.0-tiny-Q4_K_M.gguf \
  --gemma /path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  --embeddings /tmp/ctx_embeddings.npz --limit 300 --epochs 300

# re-train from saved embeddings
python experiments/contextual_translator.py \
  --embeddings /tmp/ctx_embeddings.npz --epochs 300
```

`corpus.json` is a JSON list of strings. It can be fetched from the HF
datasets-server (see the comment at the top of the script).

Result (300 wiki texts): ~0.53 cosine / 11.7% top-1 (chance 0.83%) —
inconclusive at this scale; see the design note's go/no-go gate.
