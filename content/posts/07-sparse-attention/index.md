---
title: "Sparse Attention (DeepSeek NSA/DSA)"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["attention", "sparse", "deepseek", "nsa", "dsa", "indexer", "top-k"]
description: "From fixed patterns to learned token selection — DeepSeek's Native Sparse Attention uses a lightweight indexer to dynamically pick the top-k most relevant tokens per query, making decode cost sublinear in sequence length."
weight: 1032
---

## TL;DR

Sliding window attention uses a fixed local pattern — simple but blind to which distant tokens actually matter. DeepSeek's Native Sparse Attention (NSA/DSA) replaces the fixed pattern with **learned dynamic selection**: a lightweight MQA indexer scores all cached tokens and selects the top-$k$ most relevant ones per query. The main attention then runs only on this subset. Decode cost becomes $O(k)$ instead of $O(s)$, where $k$ is the number of selected tokens (typically 2,048) regardless of sequence length. (Note: $k$ here refers to selected tokens, not [MoE experts]({{< relref "05-moe" >}}) — see [Notation]({{< relref "00-prologue#top-k" >}}).)

## Motivation

The [previous post]({{< relref "06-sliding-window" >}}) showed that sliding window attention caps compute and memory at $O(w)$ per token. But the window is a blunt instrument — it always attends to the *nearest* $w$ tokens, whether or not they're relevant. A token in a long document might need information from a specific paragraph 50K tokens ago, which a 4K sliding window would miss entirely.

The ideal: attend to the tokens that **actually matter**, wherever they are in the sequence. But determining which tokens matter requires looking at all of them — which is the full attention we're trying to avoid.

DeepSeek's insight: use a **cheap approximate attention** to identify the important tokens, then run **expensive full attention** only on those. The cheap pass costs much less than full attention because it uses a smaller model (fewer heads, lower precision).

## Mechanism

### Two-stage architecture

Sparse attention adds a lightweight **indexer** that runs before the main attention:

```
Stage 1 — Indexer (cheap):
  Score all N cached tokens → select top-k (e.g., 2048)

Stage 2 — Main attention (expensive, but only on k tokens):
  Run [MLA]({{< relref "04-mla" >}}) attention on the selected subset
```

### The indexer: a mini MQA attention

The indexer is a small, separate attention mechanism with its own weights:

$$
Q^{\text{idx}} = \mathbf{c}_q \; W_{\text{idx}}^Q \quad \in \mathbb{R}^{H_{\text{idx}} \times d_{\text{idx}}}
$$
$$
K^{\text{idx}} = \mathbf{x} \; W_{\text{idx}}^K \quad \in \mathbb{R}^{d_{\text{idx}}}
$$

Key design choices:

- **MQA structure**: $H_{\text{idx}} = 64$ query heads but only **1 shared key head** — so the cached key is a single $d_{\text{idx}}$-dimensional vector per token
- **FP8 quantized**: keys are cached in FP8 to minimize indexer cache overhead
- **Reuses the query latent**: $\mathbf{c}_q$ is the same compressed query from MLA — no separate query compression
- **Separate key projection**: $W_{\text{idx}}^K$ is independent from the main attention's key projection

### Scoring and selection

For each query token, the indexer computes rough attention scores against all $s$ cached tokens:

$$
\text{scores}_{h} = Q^{\text{idx}}_h \; {K^{\text{idx}}}^\top \quad \in \mathbb{R}^{N} \quad \text{for each of } H_{\text{idx}} \text{ heads}
$$

Since there's one shared key head and 64 query heads, this produces 64 sets of scores over $s$ tokens. A **learned weight projection** combines the heads into a single ranking:

$$
\mathbf{w} = \mathbf{x} \; W^{\text{weights}} \quad \in \mathbb{R}^{H_{\text{idx}}}
$$

$$
\text{combined\_score}_i = \sum_{h=1}^{H_{\text{idx}}} w_h \cdot \text{scores}_{h,i}
$$

$$
\text{selected} = \text{top-}k(\text{combined\_scores}) \quad \in \mathbb{Z}^{k}
$$

Each head "votes" on which tokens matter, and the learned weights determine how much each vote counts. The result: $k$ token indices (e.g., 2,048 out of 128K).

### Indexer cache cost

Per token, the indexer caches:

$$
\text{Indexer cache} = d_{\text{idx}} \times 1\text{ byte (FP8)} + \text{scale (4 bytes)}
$$

For $d_{\text{idx}} = 128$: $132$ bytes per token. Compare to the main MLA cache of $\sim 1{,}152$ bytes per token. The indexer adds about 11% overhead — genuinely cheap.

### Main attention on the subset

The selected indices are passed to the main MLA attention kernel, which reads KV cache **only at those positions**:

$$
\text{output} = \text{MLA\_Attention}(Q, \; \text{Cache}[\text{selected}])
$$

Instead of reading all $s$ cached latent vectors, the kernel reads only $k = 2{,}048$. The paged cache positions are looked up via block tables — each logical index is converted to a physical cache slot.

### Why all tokens go through the decode path

In standard [MLA]({{< relref "04-mla" >}}), prefill tokens use `forward_mha` (dense attention) and decode tokens use `forward_mqa` (attend to cache). With sparse attention, **all tokens** use the decode-style path:

The sparse kernel is designed around fetching specific cache positions by index — this works the same way whether the token is a "prefill" token or a "decode" token. There's no dense `forward_mha` path because dense attention would defeat the purpose of sparsity.

For a prefill token with 4,000 prior tokens in cache, it still only attends to $\text{top-}k = 2{,}048$ of them. If $s < k$, it attends to all of them (sparsity provides no benefit for short sequences).

## Training

### Joint training

The indexer is trained jointly with the main model. The top-$k$ selection is non-differentiable (discrete selection), but the router weights and indexer projections receive gradients through the selected tokens' attention outputs.

### When sparsity helps during training

During training with long sequences, sparse attention reduces the per-layer compute from $O(s^2)$ to $O(s \cdot k)$, enabling longer training contexts without proportional compute increase.

### Quality

The indexer must be good enough that the top-$k$ selected tokens capture most of the attention mass. If it misses important tokens, the main attention produces lower-quality outputs. In practice, the top-2,048 tokens typically capture $>95\%$ of the attention probability mass for most query positions, since attention distributions are highly concentrated.

## Inference

### Decode cost

The headline benefit — decode attention cost becomes **independent of sequence length** (up to a constant):

| Sequence length | Full attention reads | Sparse attention reads |
|---|---|---|
| 4K | 4,096 | 2,048 |
| 32K | 32,768 | 2,048 |
| 128K | 131,072 | 2,048 |

At 128K context, sparse attention reads 64× fewer cache entries. Since decode is memory-bandwidth bound (loading cached KV from HBM), this translates directly to latency reduction.

The indexer still scores all $s$ tokens, but it uses a much smaller model (one key head in FP8 vs. full MLA latent in BF16), so its cost is a fraction of full attention.

### KV cache

Two caches must be maintained:

| Cache | Per token | Purpose |
|---|---|---|
| Main MLA cache | $\sim 1{,}152$ bytes | Latent $\mathbf{c}$ + RoPE $K^{\text{pe}}$ |
| Indexer cache | $\sim 132$ bytes | FP8 key + scale |
| **Total** | $\sim 1{,}284$ bytes | 11% overhead from indexer |

Both caches grow with sequence length — sparsity doesn't reduce cache *storage*, only cache *reads* per decode step.

### Kernel support

Sparse attention requires specialized kernels that can read cache at arbitrary (non-contiguous) positions:

- **FlashMLA Sparse**: custom kernel for Hopper/Blackwell, handles the scatter-gather reads from paged cache at selected positions
- **FlashInfer MLA Sparse**: Blackwell-optimized variant
- **ROCm Aiter MLA Sparse**: AMD support

The kernel receives `topk_indices` of shape `(num_tokens, k)` — for each token, the $k$ physical cache positions to attend to.

### Index conversion

The selected indices are *logical* (token position 0, 42, 1337, ...) but the KV cache is *paged* (block tables map logical to physical). A Triton kernel converts logical indices to physical cache slots before the attention kernel runs:

$$
\text{physical} = \text{block\_table}[\text{logical} \; // \; \text{block\_size}] \times \text{block\_size} + \text{logical} \; \% \; \text{block\_size}
$$

## Trade-offs

| | Compute per token | KV cache storage | Long-range | Complexity |
|---|---|---|---|---|
| **Full attention** | $O(s \cdot d)$ | $O(s)$ | Direct to all tokens | Baseline |
| **Sliding window** | $O(w \cdot d)$ | $O(w)$ bounded | Only within window | Simple |
| **Sparse (DSA)** | $O(k \cdot d)$ + indexer | $O(s)$ unbounded | Direct to selected tokens | High |

**Sparse vs sliding window**: Sparse attention can reach *any* token in the sequence — it's not limited by distance. But it has higher complexity (indexer + main attention, two caches) and the cache still grows with sequence length (unlike sliding window's bounded cache).

**Sparse vs full attention**: At long sequences, sparse attention is dramatically faster for decode. At short sequences ($s < k$), it provides no benefit and adds indexer overhead.

**The indexer's accuracy**: If the indexer's top-$k$ misses important tokens, quality degrades. The indexer is a small model making a hard selection — it's an approximation, unlike full attention which is exact. In practice, attention distributions are concentrated enough that top-2,048 is sufficient for most tasks.

## Adoption

Sparse attention with learned token selection is currently specific to DeepSeek:

- **DeepSeek-V3.2** (2025): introduced the NSA/DSA indexer with $k = 2{,}048$, 64 indexer heads, FP8 cached keys

The approach is newer than sliding window or fixed sparse patterns. Broader adoption depends on:
- Backend support maturing (currently requires specialized kernels)
- Demonstrating quality parity with full attention across diverse tasks
- Other model families experimenting with similar learned sparsity

Earlier fixed-pattern sparse attention (Sparse Transformer [[1]](#ref-1), Longformer, BigBird) used predetermined patterns rather than learned selection. DeepSeek's contribution is making the selection **dynamic and learned**, adapting per-token based on content.

## References

<span id="ref-1">[1]</span> Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509). *arXiv preprint*.

<span id="ref-2">[2]</span> DeepSeek-AI. (2025). [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089). *arXiv preprint*.

*Last updated: March 2026*
