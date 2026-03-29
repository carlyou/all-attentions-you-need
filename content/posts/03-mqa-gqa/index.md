---
title: "Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["attention", "mqa", "gqa", "kv-cache"]
description: "Sharing KV heads across query heads — a simple architectural change that dramatically shrinks the KV cache, with GQA as the sweet spot most frontier models have adopted."
weight: 3
---

## TL;DR

Multi-Query Attention (MQA) uses a single shared KV head for all query heads, cutting the KV cache by $H\times$. Grouped-Query Attention (GQA) is the compromise — groups of query heads share KV heads, balancing cache size and model quality. GQA is now the dominant choice in frontier models (Llama, Mistral, Gemma).

## Motivation

As we saw in the [MHA post]({{< relref "01-multi-head-attention" >}}), the KV cache scales linearly with the number of heads:

$$
\text{KV cache} = 2 \times L \times H \times d_k \times s \times B \times \text{sizeof(dtype)}
$$

During decode, the model loads the entire cached $K$ and $V$ for every head at every layer, just to produce one new token. At large batch sizes and long sequences, this memory-bandwidth cost dominates inference latency.

Shazeer [[1]](#ref-1) observed that the KV heads are the bottleneck — not the query heads. The query projection runs on the *new* token (just 1 row), while the KV cache stores *all prior tokens* (thousands of rows). What if multiple query heads could share a single set of KV heads?

## Mechanism

### MQA: the extreme — one KV head for all queries

In standard MHA, each of $H$ query heads has its own dedicated KV head. MQA collapses all KV heads into one:

$$
Q_h = X W_h^Q \quad \text{(per-head, as before)}
$$
$$
K = X W^K, \quad V = X W^V \quad \text{(single head, shared)}
$$

Each query head computes attention against the **same** $K$ and $V$:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K^\top}{\sqrt{d_k}}\right) V
$$

The output is still concatenated and projected through $W^O$ as in MHA.

**KV cache savings**: from $2 H d_k$ per token to $2 d_k$ per token — an $H\times$ reduction. For a model with 32 heads, MQA uses 1/32 the KV cache memory of MHA.

### GQA: the sweet spot — grouped KV heads

Ainslie et al. [[2]](#ref-2) generalized MQA by introducing $G$ KV head groups, where $1 \leq G \leq H$:

$$
K_g = X W_g^K, \quad V_g = X W_g^V \quad \text{for } g = 1, \dots, G
$$

Query heads are divided into $G$ groups of $H/G$ heads each. All query heads within a group share the same $K_g$ and $V_g$:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_{g(h)}^\top}{\sqrt{d_k}}\right) V_{g(h)}
$$

where $g(h) = \lfloor h \cdot G / H \rfloor$ maps query head $h$ to its KV group.

The endpoints of the spectrum:

- $G = 1$: MQA (one shared KV head)
- $G = H$: standard MHA (per-head KV)
- $1 < G < H$: GQA (the compromise)

**KV cache**: $2 G d_k$ per token. Typical configurations use $G = 8$ with $H = 32$, giving a $4\times$ reduction over MHA.

### How it looks in practice

```
MHA (H=32, G=32):
  Q heads: [Q0] [Q1] [Q2] [Q3] ... [Q31]
  KV heads: [K0] [K1] [K2] [K3] ... [K31]
  Each Q has its own KV. Cache: 32 KV heads.

GQA (H=32, G=8):
  Q heads: [Q0  Q1  Q2  Q3] [Q4  Q5  Q6  Q7] ... [Q28 Q29 Q30 Q31]
  KV heads:      [K0]             [K1]        ...        [K7]
  4 Q heads share each KV head. Cache: 8 KV heads.

MQA (H=32, G=1):
  Q heads: [Q0  Q1  Q2  ... Q31]
  KV heads:        [K0]
  All Q heads share one KV head. Cache: 1 KV head.
```

## Training

### From scratch

GQA/MQA models are trained from scratch with the reduced number of KV heads — the architecture is defined upfront. The KV projection matrices are simply smaller:

$$
W^K \in \mathbb{R}^{d \times G \cdot d_k} \quad \text{instead of} \quad \mathbb{R}^{d \times H \cdot d_k}
$$

Training compute is slightly reduced (fewer KV parameters), but the effect is small since the KV projections are a fraction of total model parameters.

### Uptrained from MHA

Ainslie et al. [[2]](#ref-2) showed that an existing MHA model can be *uptrained* into a GQA model. The conversion takes the $H$ KV heads and groups them:

- **Mean pooling**: average the KV weights within each group to initialize the shared KV head
- **Selection**: pick one representative KV head per group (e.g., the first)

The model is then fine-tuned for a small fraction of the original training budget (typically 5–10%) to recover quality. This is how Llama 2 70B was converted — the 34B and 7B variants use MHA, but the 70B variant uses GQA to manage its larger KV cache.

### Quality impact

MQA ($G = 1$) shows measurable quality degradation — all query heads are forced to attend through the same lens. GQA with $G = 8$ recovers nearly all of MHA's quality while still providing a $4\times$ cache reduction. The empirical finding from [[2]](#ref-2): GQA with a modest number of groups matches MHA quality on most benchmarks.

## Inference

### KV cache

The direct benefit — smaller cache, more concurrent requests:

| Config ($H = 128$, $d_k = 128$, FP16) | KV per token | Relative |
|---|---|---|
| MHA ($G = 128$) | 65,536 bytes | 1× |
| GQA ($G = 8$) | 4,096 bytes | 1/16× |
| MQA ($G = 1$) | 512 bytes | 1/128× |

### Decode bandwidth

During decode, the attention kernel loads the cached KV for all prior tokens. With GQA, the same $K_g$ and $V_g$ are broadcast to multiple query heads:

$$
\text{Load } K_g \text{ once} \rightarrow \text{compute attention for } H/G \text{ query heads}
$$

This improves **arithmetic intensity** — more useful FLOPs per byte loaded from HBM. With MHA, each KV head serves exactly one query head. With GQA ($G = 8$, $H = 32$), each KV head serves 4 query heads — $4\times$ better arithmetic intensity for the KV cache reads.

### Tensor parallelism

GQA shards naturally, with one constraint: each GPU should own complete KV groups. With $G = 8$ KV heads across 4 GPUs:

- Each GPU gets 2 KV heads and $32/4 = 8$ query heads
- The 8 query heads split into 2 groups of 4, each group mapping to one local KV head
- No cross-GPU KV sharing needed — attention is fully local

This works as long as $G$ is divisible by the TP degree. In practice, models choose $G$ with this in mind.

### Kernel support

GQA/MQA are supported by all major attention kernels. From the kernel's perspective, the only difference from MHA is a broadcasting step — the same $K$/$V$ tensor is used by multiple query heads. FlashAttention, FlashInfer, and Triton all handle this natively via a `num_kv_heads` parameter.

## Trade-offs

| | KV cache | Quality | Decode BW | Complexity |
|---|---|---|---|---|
| **MHA** | $2 H d_k$ | Best | 1× | Baseline |
| **GQA** ($G = 8$) | $2 G d_k$ | Near MHA | $H/G \times$ better | Minimal change |
| **MQA** ($G = 1$) | $2 d_k$ | Reduced | $H \times$ better | Minimal change |

GQA is the Pareto-optimal choice for most use cases — near-MHA quality with significant cache and bandwidth savings. MQA is only worth it when memory is extremely constrained and some quality loss is acceptable.

The elegance of GQA is that it requires **zero architectural innovation** beyond the original MHA — it's just a different choice of how many KV heads to use. No new projections, no new algorithms, no new kernels. This simplicity is why adoption has been so rapid.

## Adoption

GQA has become the default for inference-optimized models:

- **Llama 2 70B**: 64 Q heads, 8 KV heads (GQA)
- **Llama 3** (all sizes): GQA
- **Mistral / Mixtral**: 32 Q heads, 8 KV heads (GQA)
- **Gemma**: GQA
- **Qwen 2**: GQA

MQA is used in some earlier models:

- **PaLM** (Google): MQA
- **Falcon**: MQA
- **StarCoder**: MQA

MHA is largely limited to older or smaller models where KV cache is not the bottleneck (BERT, GPT-2, smaller fine-tunes).

Looking forward, [MLA]({{< relref "04-mla" >}}) (next post) takes a fundamentally different approach — rather than reducing the *number* of KV heads, it compresses the *representation* itself.

## References

<span id="ref-1">[1]</span> Shazeer, N. (2019). [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150). *arXiv preprint*.

<span id="ref-2">[2]</span> Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). *EMNLP 2023*.

*Last updated: March 2026*
