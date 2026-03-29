---
title: "Multi-head Latent Attention (MLA)"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["attention", "mla", "deepseek", "kv-cache", "low-rank"]
description: "DeepSeek's MLA compresses all KV heads into a single low-rank latent vector — achieving near-MQA cache size with near-MHA quality, at the cost of fundamentally different prefill and decode code paths."
weight: 4
---

## TL;DR

Multi-head Latent Attention (MLA) compresses all KV heads into a small latent vector $\mathbf{c}$ via a learned down-projection, storing only $\mathbf{c}$ in the KV cache instead of the full per-head $K$ and $V$. During prefill, the latent is decompressed back to full KV for standard attention. During decode, a mathematical trick absorbs the decompression into the query, enabling attention directly in the latent space. The result: near-MQA cache size, near-MHA quality, but two completely different code paths.

## Motivation

[GQA]({{< relref "03-mqa-gqa" >}}) (previous post) reduces the KV cache by sharing KV heads across query groups. But the sharing is coarse — you choose a fixed number of groups $G$ and each group stores a full-dimensional KV head. The information capacity of each cached KV head is the same as in MHA; there are just fewer of them.

DeepSeek [[1]](#ref-1) asked a different question: across all heads, the $K$ and $V$ vectors are all derived from the same hidden state $\mathbf{x}$. Is there redundancy in storing them independently? The answer: yes. A low-rank projection can capture most of the information in far fewer dimensions.

The key insight: **compress first, cache the compression, decompress later when needed.**

## Mechanism

### The latent bottleneck

In MHA, each token produces KV for every head independently:

$$
K_h = \mathbf{x} W_h^K, \quad V_h = \mathbf{x} W_h^V \quad \text{for } h = 1, \dots, H
$$

MLA replaces this with a two-step process — compress into a latent, then (when needed) decompress back:

$$
\mathbf{c} = \mathbf{x} W^{\text{down}} \quad \in \mathbb{R}^{d_c}
$$

$$
[K_1, V_1, \dots, K_H, V_H] = \mathbf{c} W^{\text{up}}
$$

where $W^{\text{down}} \in \mathbb{R}^{d \times d_c}$ compresses from the hidden dimension $d$ to the latent dimension $d_c$, and $W^{\text{up}} \in \mathbb{R}^{d_c \times H(d_k + d_v)}$ decompresses back to all heads' KV.

**Only $\mathbf{c}$ is stored in the KV cache** — not the full $K$ and $V$ heads.

### Cache comparison

For DeepSeek-V2 with $H = 128$, $d_k = 128$, $d_c = 512$:

| Variant | Cached per token | Relative |
|---|---|---|
| MHA ($H = 128$) | $2 \times 128 \times 128 = 32{,}768$ values | 1× |
| GQA ($G = 8$) | $2 \times 8 \times 128 = 2{,}048$ values | 1/16× |
| MQA ($G = 1$) | $2 \times 128 = 256$ values | 1/128× |
| MLA ($d_c = 512$) | $512 + 64 = 576$ values | 1/57× |

The extra 64 values in MLA come from the positional encoding component (explained below). MLA achieves cache sizes between MQA and GQA, with quality near MHA.

### The RoPE problem

There's a complication: Rotary Position Embeddings (RoPE) are applied to $K$ *after* projection. RoPE encodes each token's position by rotating the key vector — it needs to know the token's position, and it operates on the full head dimension.

If $K$ is compressed into $\mathbf{c}$, you can't apply RoPE to it — the latent mixes all heads' information together, destroying the per-head structure RoPE needs.

MLA's solution: split $K$ into two parts:

$$
K_h = [K_h^{\text{nope}}, \; K_h^{\text{pe}}]
$$

- $K_h^{\text{nope}}$ (no positional encoding): derived from the latent $\mathbf{c}$ via $W^{\text{up}}$, compressed in the cache
- $K_h^{\text{pe}}$ (positional encoding): carries RoPE, stored **separately** in the cache as a small shared vector

The query is split the same way:

$$
Q_h = [Q_h^{\text{nope}}, \; Q_h^{\text{pe}}]
$$

The attention score for head $h$ is the sum of the nope and pe components:

$$
\text{score}_h = Q_h^{\text{nope}} {K_h^{\text{nope}}}^\top + Q_h^{\text{pe}} {K^{\text{pe}}}^\top
$$

Note that $K^{\text{pe}}$ is shared across heads (like MQA) since it's a single position embedding, while $K_h^{\text{nope}}$ is per-head (recovered from the latent).

### The full cache

$$
\text{Cache per token} = \underbrace{\mathbf{c}}_{\text{latent } (d_c)} + \underbrace{K^{\text{pe}}}_{\text{RoPE } (d_{\text{pe}})}
$$

For DeepSeek-V2: $d_c = 512$, $d_{\text{pe}} = 64$, totaling 576 values per token.

### Query compression

MLA also compresses the query through a latent bottleneck:

$$
\mathbf{c}_q = \mathbf{x} W^{\text{down}}_Q \quad \in \mathbb{R}^{d_{cq}}
$$
$$
Q = \mathbf{c}_q W^{\text{up}}_Q
$$

This reduces the parameter count but doesn't affect the KV cache — the query is never cached.

## The Two Code Paths

The most distinctive aspect of MLA: prefill and decode use **fundamentally different algorithms**, not just different kernels.

### Prefill: decompress and run standard MHA

During prefill, all tokens are available in the current batch. The strategy:

1. Compress: $\mathbf{c} = \mathbf{x} W^{\text{down}}$
2. Immediately decompress: $K_h^{\text{nope}}, V_h = \text{split}(\mathbf{c} \; W^{\text{up}})$
3. Apply RoPE to get $K_h^{\text{pe}}$
4. Run standard FlashAttention on the full-sized $Q$, $K$, $V$
5. Store only $\mathbf{c}$ and $K^{\text{pe}}$ in the cache

Compressing then immediately decompressing seems pointless — but the compression only matters for *what goes into the cache*. The attention itself operates on full-size heads because this is cheaper (smaller inner dimension for the dot product).

Why not absorb during prefill? With $s$ prefill tokens, both $Q$ and $K$ are $s$ rows. Decompressing $K$ is one GEMM of $s$ tokens — the same cost as transforming $Q$. No asymmetry to exploit. And working in the decompressed space gives a smaller inner dimension ($d_k$) for the $Q K^\top$ product compared to the latent dimension ($d_c > d_k$), so it's actually *fewer* FLOPs.

### Decode: the absorption trick

During decode, there is 1 new query token and $s$ cached tokens. The naive approach:

$$
K_h = \mathbf{c}_i \; W^{\text{up}}_K \quad \text{for all } s \text{ cached tokens}
$$

This requires running $W^{\text{up}}_K$ on every cached token — $s$ matrix multiplies, which is expensive and defeats the purpose of the small cache.

The absorption trick rearranges the math. The attention score between query $\mathbf{q}$ and cached token $i$ (nope part only):

$$
\mathbf{q}^{\text{nope}} {K_h^{\text{nope}}}^\top
= \mathbf{q}^{\text{nope}} (W^{\text{up}}_K \; \mathbf{c}_i)^\top
= \mathbf{q}^{\text{nope}} \; \mathbf{c}_i^\top \; {W^{\text{up}}_K}^\top
= (\mathbf{q}^{\text{nope}} \; {W^{\text{up}}_K}^\top) \; \mathbf{c}_i^\top
= {\mathbf{q}'}^{\text{nope}} \; \mathbf{c}_i^\top
$$

where ${\mathbf{q}'}^{\text{nope}} = \mathbf{q}^{\text{nope}} \; {W^{\text{up}}_K}^\top$ absorbs the decompression into the query.

**Transform 1 query instead of $s$ cached tokens.** The cost goes from $O(s)$ matrix multiplies to $O(1)$.

The same trick applies to the value side. The final attention output:

$$
\text{output} = \left(\sum_i \alpha_i \; \mathbf{c}_i\right) {W^{\text{up}}_V}^\top
$$

Compute the weighted sum in latent space first (cheap, dimension $d_c$), then decompress the result once.

### Why decode looks like MQA

After absorption, each query head dot-products against the **same shared $\mathbf{c}$** vectors. Different heads have different absorbed queries $\mathbf{q}'_h$ (because each head has a different $W^Q_h$), but they all attend against the same cached latents. This is structurally identical to MQA — one shared "key" per cached token, multiple query heads.

The difference from true MQA: $\mathbf{c}$ is a richer representation than a single KV head. The per-head diversity comes from $W^{\text{up}}$, which produces different $K$ and $V$ for each head from the same $\mathbf{c}$. MQA's single KV head is genuinely low-capacity; MLA's latent is a compressed encoding of all heads.

### Summary of the two paths

| | Prefill | Decode |
|---|---|---|
| **Strategy** | Decompress $\mathbf{c}$ → full $K$, $V$ → standard MHA | Absorb $W^{\text{up}}$ into $Q$ → attend in latent space |
| **Why** | Both $Q$ and $K$ are $s$ tokens, no asymmetry | 1 query vs $s$ cached: transform the 1, not the $s$ |
| **Compute** | Compute-bound (many tokens, standard FA) | Memory-bound (read $s$ cached latents) |
| **Cache interaction** | Write $\mathbf{c}$ to cache | Read $\mathbf{c}$ from cache |

## Training

### Parameters

MLA adds the down/up projection matrices ($W^{\text{down}}$, $W^{\text{up}}$) but removes per-head KV projections. The total parameter count is comparable to MHA — the bottleneck matrices are smaller than the sum of per-head projections.

For DeepSeek-V2:
- $W^{\text{down}}$: $(7168, 512)$ — shared across all heads
- $W^{\text{up}}$: $(512, 128 \times (128 + 128))$ — produces all heads' $K^{\text{nope}}$ and $V$

### Quality

DeepSeek-V2 [[1]](#ref-1) showed MLA matches or exceeds MHA quality on standard benchmarks, despite the aggressive KV compression. The low-rank bottleneck acts as a form of regularization — forcing the model to find a compact representation of the key-value information.

### Compute

Training compute is similar to MHA. The down/up projections replace per-head KV projections with roughly equivalent FLOPs. The attention computation itself is identical to MHA during training (since the latent is decompressed for the forward pass).

## Inference

### KV cache

The headline number: **576 values per token** (512 latent + 64 RoPE) compared to MHA's 32,768 for DeepSeek-V2's 128-head configuration. This 57× reduction means:

- More concurrent requests in the same GPU memory
- Smaller paged attention blocks
- Less HBM bandwidth per decode step

### Decode compute overhead

The absorption trick eliminates decompressing $s$ cached tokens, but adds:

1. $\mathbf{q}' = \mathbf{q} \; {W^{\text{up}}_K}^\top$ — one matrix multiply per query token (cheap)
2. $\text{output} = \text{intermediate} \; {W^{\text{up}}_V}^\top$ — one matrix multiply for the final decompression (cheap)

These are small fixed costs compared to the attention over $s$ cached tokens.

### Kernel support

MLA requires specialized attention backends because the decode path operates in latent space with a non-standard head dimension. In vLLM, MLA has dedicated backends:

- **FlashMLA**: custom kernel for MLA decode on Hopper/Blackwell
- **FlashInfer MLA**: TRT-LLM kernel for Blackwell
- **CUTLASS MLA**: Blackwell-optimized with larger block sizes
- **Triton MLA**: portable fallback for all architectures

Prefill uses standard FlashAttention (after decompression, it's normal MHA).

### Tensor parallelism

$W^{\text{down}}$ is shared across all heads and replicated on every GPU. $W^{\text{up}}$ is conceptually per-head (different output slices for different heads), so it shards the same way as MHA's per-head projections — split output columns across GPUs.

## Trade-offs

| | KV cache | Decode compute | Quality | Code complexity |
|---|---|---|---|---|
| **MHA** | Largest | Lowest | Best (baseline) | Simple |
| **GQA** | Medium | Low | Near MHA | Minimal change from MHA |
| **MQA** | Smallest | Lowest | Reduced | Minimal change from MHA |
| **MLA** | Small | Medium (absorption) | Near MHA | High (two code paths) |

**MLA vs GQA**: MLA has a smaller cache and better quality retention, but requires two distinct code paths (prefill and decode), specialized kernels, and the RoPE split adds complexity. GQA is a drop-in replacement with standard kernels.

**MLA vs MQA**: Both have small caches, but MLA preserves per-head diversity through the up-projection while MQA genuinely collapses to one KV head. MLA's quality is significantly better.

**The fundamental trade-off**: MLA trades **architectural simplicity and code complexity** for **the best cache-size-to-quality ratio** of any variant.

## Adoption

MLA is used exclusively by DeepSeek models (as of March 2026):

- **DeepSeek-V2**: introduced MLA ($d_c = 512$, $d_{\text{pe}} = 64$, 128 heads)
- **DeepSeek-V3**: MLA + MoE + sparse attention
- **DeepSeek-V3.2**: adds the learned sparse indexer (NSA/DSA) on top of MLA

Other model families have not adopted MLA, preferring GQA for its simplicity. However, as serving costs become the dominant concern and backend support matures, MLA's cache efficiency may drive broader adoption.

## References

<span id="ref-1">[1]</span> DeepSeek-AI. (2024). [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434). *arXiv preprint*.

<span id="ref-2">[2]</span> Shazeer, N. (2019). [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). *arXiv preprint*.

*Last updated: March 2026*
