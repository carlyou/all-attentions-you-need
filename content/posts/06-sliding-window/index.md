---
title: "Sliding Window & Local Attention"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["attention", "sliding-window", "local", "longformer", "bigbird", "mistral"]
description: "Limiting attention to a fixed local window — a simple constraint that caps KV cache at O(w) instead of O(s), trading long-range attention for bounded memory and compute."
weight: 6
---

## TL;DR

Sliding window attention restricts each token to attend only to the $w$ nearest tokens instead of the full sequence. This caps memory and compute at $O(s \cdot w)$ instead of $O(s^2)$, and bounds the KV cache per request. The trade-off: the model cannot directly attend beyond the window. Information must propagate through intermediate layers to cross the boundary. Mistral, Longformer, and BigBird all use variants of this idea.

## Motivation

Full self-attention has $O(s^2)$ complexity — every token attends to every other token. For a 128K-token sequence, the attention matrix has 16 billion entries per head. Even with FlashAttention (which avoids materializing this matrix in HBM), the compute is still quadratic.

But do all token pairs actually matter? In many tasks, the most relevant context is nearby — adjacent sentences, the current paragraph, the recent dialogue turn. Distant tokens matter less, and their contribution to the attention output is often negligible after softmax normalization.

Sliding window attention formalizes this observation: **restrict attention to a local window and skip the rest.**

## Mechanism

### The window constraint

For a window size $w$, token at position $i$ can attend to positions $[\max(0, i - w + 1), \; i]$:

$$
\text{Attention}(Q_i, K, V) = \text{softmax}\!\left(\frac{Q_i K_{[i-w+1:i]}^\top}{\sqrt{d_k}}\right) V_{[i-w+1:i]}
$$

Visualized as a mask on the $s \times s$ attention matrix:

```
Full attention (s=8):          Sliding window (w=3):
  1 0 0 0 0 0 0 0               1 0 0 0 0 0 0 0
  1 1 0 0 0 0 0 0               1 1 0 0 0 0 0 0
  1 1 1 0 0 0 0 0               1 1 1 0 0 0 0 0
  1 1 1 1 0 0 0 0               0 1 1 1 0 0 0 0
  1 1 1 1 1 0 0 0               0 0 1 1 1 0 0 0
  1 1 1 1 1 1 0 0               0 0 0 1 1 1 0 0
  1 1 1 1 1 1 1 0               0 0 0 0 1 1 1 0
  1 1 1 1 1 1 1 1               0 0 0 0 0 1 1 1
```

The lower-left triangle becomes a diagonal band. Each row has at most $w$ entries instead of up to $s$.

### Effective receptive field

A single sliding window layer has a receptive field of $w$ tokens. But stacking $L$ layers expands this — layer 2 can attend to layer 1's outputs, which themselves incorporated $w$ tokens of context:

$$
\text{Effective receptive field after } L \text{ layers} = L \times w
$$

With $L = 32$ layers and $w = 4096$, the effective receptive field is 131K tokens — information from any token can theoretically propagate to any other through the intermediate layers. However, this is indirect — the signal degrades as it passes through many layers. Direct attention (as in full self-attention) is stronger than multi-hop propagation.

### Hybrid patterns: global + local

Longformer [[1]](#ref-1) and BigBird [[2]](#ref-2) augment the sliding window with **global tokens** that attend to (and are attended by) all positions:

```
Longformer pattern:
  [CLS] token: global attention (full row and column)
  All other tokens: sliding window only

BigBird pattern:
  Global tokens + sliding window + random sparse connections
```

The global tokens act as information bottlenecks — they aggregate information from the full sequence and broadcast it to every local window. This preserves some long-range capability without quadratic cost.

### Mistral's approach

Mistral [[3]](#ref-3) uses a pure sliding window without global tokens, relying on the multi-layer receptive field expansion. Configuration:

- $w = 4096$ tokens
- 32 layers → effective receptive field of 131K tokens
- No global tokens, no hybrid patterns — simplicity over expressiveness

This works well in practice because the SwiGLU FFN and residual connections between layers provide additional information mixing beyond what attention alone achieves.

## Training

### Compute

Per layer, the attention compute is:

$$
O(s \cdot w \cdot d) \quad \text{instead of} \quad O(s^2 \cdot d)
$$

For $s \gg w$, this is a significant reduction. With $s = 128\text{K}$ and $w = 4096$, the speedup is $128\text{K} / 4096 = 32\times$ per layer.

### Memory

The attention score matrix per head shrinks from $(s, s)$ to $(s, w)$:

$$
\text{Scores memory} = H \cdot s \cdot w \cdot \text{sizeof(dtype)}
$$

With FlashAttention, this intermediate is never materialized, so the practical benefit is in compute savings rather than memory. The activation memory for the backward pass is also reduced proportionally.

### Quality impact

Sliding window attention loses the ability to directly attend to distant tokens. For tasks requiring long-range reasoning (multi-document QA, long-form summarization), this can hurt quality. For tasks where locality dominates (code generation, conversational AI), the impact is minimal.

The empirical finding: with sufficient layers and a window of 4K–8K tokens, sliding window models match full-attention models on most benchmarks, with quality degradation mainly on tasks explicitly testing long-range dependency.

## Inference

### KV cache

The key serving benefit: the KV cache per request is **bounded**:

$$
\text{KV cache per request} = 2 \times L \times H \times d_k \times w \times \text{sizeof(dtype)}
$$

(Note: unlike [MHA]({{< relref "01-multi-head-attention" >}}) where cache grows with $s$, this is bounded by $w$ regardless of sequence length.)

With full attention, cache grows with sequence length $s$. With sliding window, it caps at $w$ regardless of how long the sequence gets. Tokens beyond the window are **evicted** from the cache — their KV entries are no longer needed.

For Mistral with $w = 4096$, $L = 32$, $H = 8$ (GQA), $d_k = 128$, FP16:

$$
2 \times 32 \times 8 \times 128 \times 4096 \times 2 \approx 537\text{ MB per request}
$$

This is the same regardless of whether the sequence has 4K or 128K tokens.

### Rolling buffer

The cache operates as a **circular buffer** of size $w$:

```
Positions:  0  1  2  3  4  5  6  7  8  9  ...
Cache slot: 0  1  2  3  0  1  2  3  0  1  ...  (w=4 example)

At position 8: cache holds tokens [5, 6, 7, 8]
               token 4's KV was overwritten when token 8 was written to slot 0
```

Each new token overwrites the oldest token's cache slot. No dynamic allocation, no fragmentation, no block management — just a fixed-size ring buffer per request.

This is simpler than paged attention — no block tables, no scatter-gather. But it only works because the window is fixed. With full attention, you need all prior tokens and can't overwrite anything.

### Kernel support

Sliding window is supported natively by most attention kernels via a `window_size` parameter:

- **FlashAttention**: `window_size=(w, 0)` for causal sliding window
- **FlashInfer**: native sliding window support
- **Triton**: configurable window masking

The kernel simply skips score computation for positions outside the window. With FlashAttention's tiling, this means entire K/V tiles outside the band are skipped — no wasted compute.

## Trade-offs

| | Compute | KV cache | Long-range | Simplicity |
|---|---|---|---|---|
| **Full attention** | $O(s^2 d)$ | $O(s)$ unbounded | Direct | Baseline |
| **Sliding window** | $O(s w d)$ | $O(w)$ bounded | Indirect (multi-hop) | Very simple |
| **Sliding + global** | $O(s(w + g) d)$ | $O(s)$ for global tokens | Direct for globals | Moderate |

**When to use sliding window**:
- Serving with strict memory budgets (bounded cache is the main draw)
- Tasks dominated by local context (code, chat, translation)
- Models that will be stacked deep enough for multi-hop propagation

**When NOT to use sliding window**:
- Tasks requiring direct long-range attention (retrieval-augmented generation, long-document QA)
- When the window would need to be so large it approaches full attention anyway

## Adoption

- **Mistral 7B / Mixtral**: $w = 4096$, pure sliding window
- **Longformer**: $w = 512$, global + local hybrid (classification tasks)
- **BigBird**: $w = 64$, global + local + random (long document understanding)
- **Gemma 2**: alternating layers of full attention and sliding window

The trend in decoder-only LLMs is toward larger windows (4K–8K) or full attention with FlashAttention, rather than narrow windows with global tokens. The Longformer/BigBird pattern is more common in encoder models for classification/extraction tasks.

## References

<span id="ref-1">[1]</span> Beltagy, I., Peters, M. E., & Cohan, A. (2020). [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150). *arXiv preprint*.

<span id="ref-2">[2]</span> Zaheer, M., Guruganesh, G., Dubey, K. A., et al. (2020). [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062). *NeurIPS 2020*.

<span id="ref-3">[3]</span> Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825). *arXiv preprint*.

*Last updated: March 2026*
