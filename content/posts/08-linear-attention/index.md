---
title: "Linear Attention & Gated DeltaNet (GDN)"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["attention", "linear", "gdn", "deltanet", "ssm", "subquadratic"]
description: "Replacing softmax attention with linear maps and gated delta rules — achieving O(s) complexity by reformulating attention as a recurrence, at the cost of losing the exact token-to-token attention distribution."
weight: 1033
---

## TL;DR

Linear attention replaces the softmax with a kernel trick that allows the computation order to be rearranged from $O(s^2)$ to $O(s \cdot d)$ — linear in sequence length. Gated DeltaNet (GDN) extends this with a gated delta rule that selectively updates a recurrent state, combining the strengths of linear attention with SSM-like recurrence. Both trade the exact softmax attention distribution for subquadratic scaling.

## Motivation

Every attention variant we've seen so far — [MHA]({{< relref "01-multi-head-attention" >}}), [GQA]({{< relref "03-mqa-gqa" >}}), [MLA]({{< relref "04-mla" >}}), [sliding window]({{< relref "06-sliding-window" >}}), [sparse]({{< relref "07-sparse-attention" >}}) — computes the same fundamental operation:

$$
\text{softmax}(Q K^\top) V
$$

The softmax is what creates the $O(s^2)$ bottleneck: you must compute all $s \times s$ pairwise scores before normalizing. FlashAttention hides the memory cost via tiling, but the compute remains quadratic.

What if we could replace softmax with something that doesn't require all pairwise scores?

Katharopoulos et al. [[1]](#ref-1) showed that by replacing softmax with a decomposable kernel function, the computation can be rearranged to avoid the $s \times s$ matrix entirely — reducing attention to $O(s \cdot d^2)$, which is linear in $s$ when $d$ is fixed.

## Mechanism

### Standard softmax attention

The standard formulation (ignoring scaling for clarity):

$$
O = \text{softmax}(Q K^\top) V
$$

For output row $i$:

$$
O_i = \frac{\sum_{j=1}^{s} \exp(Q_i K_j^\top) V_j}{\sum_{j=1}^{s} \exp(Q_i K_j^\top)}
$$

The $\exp$ and normalization couple all $s$ positions — you can't compute the denominator without seeing all keys.

### The kernel trick

Replace $\exp(Q_i K_j^\top)$ with a decomposable kernel $\phi(Q_i)^\top \phi(K_j)$ where $\phi$ is a feature map:

$$
O_i = \frac{\sum_{j=1}^{s} \phi(Q_i)^\top \phi(K_j) \; V_j}{\sum_{j=1}^{s} \phi(Q_i)^\top \phi(K_j)}
$$

Now rearrange using associativity of matrix multiplication:

$$
O_i = \frac{\phi(Q_i)^\top \sum_{j=1}^{s} \phi(K_j) V_j^\top}{\phi(Q_i)^\top \sum_{j=1}^{s} \phi(K_j)}
$$

Define the **recurrent state**:

$$
S = \sum_{j=1}^{s} \phi(K_j) V_j^\top \quad \in \mathbb{R}^{d_\phi \times d_v}
$$

$$
z = \sum_{j=1}^{s} \phi(K_j) \quad \in \mathbb{R}^{d_\phi}
$$

Then:

$$
O_i = \frac{\phi(Q_i)^\top S}{\phi(Q_i)^\top z}
$$

The crucial insight: $S$ and $z$ can be computed **incrementally**:

$$
S_j = S_{j-1} + \phi(K_j) V_j^\top
$$

$$
z_j = z_{j-1} + \phi(K_j)
$$

This is a **recurrence** — process tokens one at a time, updating a fixed-size state. No $s \times s$ matrix needed.

### Complexity comparison

| | Training (parallel) | Decode (sequential) |
|---|---|---|
| **Softmax attention** | $O(s^2 d)$ | $O(s \cdot d)$ per step |
| **Linear attention** | $O(s \cdot d_\phi \cdot d_v)$ | $O(d_\phi \cdot d_v)$ per step |

For decode, linear attention is $O(1)$ per step (update state, query state) instead of $O(s)$ (read entire KV cache). This is the holy grail for long-context serving: decode cost is constant regardless of sequence length.

### The catch: feature map quality

The choice of $\phi$ determines quality. Common options:

- $\phi(x) = \text{elu}(x) + 1$ (original linear attention [[1]](#ref-1))
- $\phi(x) = \text{ReLU}(x)$
- Random Fourier features (approximate softmax kernel)

None of these perfectly approximate the softmax kernel $\exp(Q K^\top)$. The sharply peaked, content-dependent attention distributions that softmax produces — where a few tokens dominate — are hard to replicate with linear kernels. This leads to quality degradation, particularly on tasks requiring precise token retrieval.

### The state size problem

The recurrent state $S \in \mathbb{R}^{d_\phi \times d_v}$ is a **matrix**, not a vector. For $d_\phi = d_v = 128$, that's 16K values per head per layer — comparable to caching 128 KV entries. If $d_\phi$ is large (for better approximation quality), the state can become larger than a reasonable KV cache, defeating the purpose.

## Gated DeltaNet (GDN)

### From linear attention to gated recurrence

Pure linear attention adds every token equally to the state — old tokens never decay, and the state grows noisy over long sequences. GDN [[2]](#ref-2) addresses this with two modifications:

**1. Gating**: a learned gate $\alpha_t$ controls how much of the previous state to retain:

$$
S_t = \alpha_t \odot S_{t-1} + \beta_t \; K_t V_t^\top
$$

where $\alpha_t \in (0, 1)$ is the retention gate and $\beta_t$ is the write strength. When $\alpha_t$ is small, old state is forgotten; when $\alpha_t \approx 1$, the state persists.

**2. Delta rule**: instead of simply adding $K_t V_t^\top$, update the state by *correcting* the current value:

$$
S_t = S_t + \beta_t \; K_t (V_t - K_t^\top S_t)^\top
$$

This is a delta rule (Widrow-Hoff learning rule) — if the state already predicts $V_t$ well from $K_t$, the update is small. If the prediction is poor, the update is large. This makes the state more efficient, selectively storing new information and overwriting stale information.

### Connection to SSMs

GDN's recurrence is structurally similar to State Space Models (SSMs) like Mamba [[3]](#ref-3):

$$
\text{SSM:} \quad h_t = A \; h_{t-1} + B \; x_t
$$
$$
\text{GDN:} \quad S_t = \alpha_t \odot S_{t-1} + \beta_t \; K_t V_t^\top
$$

Both are selective recurrences with input-dependent gating. The difference: GDN's state is a matrix (key-value associative memory), while SSMs typically use a vector state. This gives GDN more capacity for recall-intensive tasks.

### Training: parallel form

Despite being defined as a recurrence, GDN can be trained in parallel using a **chunkwise** algorithm:

1. Divide the sequence into chunks of size $c$
2. Within each chunk: compute attention scores as a small $(c \times c)$ matrix (quadratic, but $c$ is small)
3. Across chunks: propagate the recurrent state

This gives $O(s \cdot c \cdot d)$ complexity — linear in $s$ with a small constant from the chunk size.

## Training

### Compute

Both linear attention and GDN are $O(s \cdot d^2)$ for training (or $O(s \cdot c \cdot d)$ with chunking). This enables training on much longer sequences than softmax attention for the same compute budget.

### Memory

No $s \times s$ attention matrix — the state $S$ is fixed-size ($d \times d$ per head). Memory scales linearly with $s$ for activations, not quadratically. This is a genuine advantage over FlashAttention, which still does $O(s^2)$ compute even though it avoids $O(s^2)$ memory.

### Quality gap

The persistent challenge: linear attention and GDN underperform softmax attention on benchmarks that require **precise recall** — looking up a specific fact from a long context. The recurrent state is lossy by nature (finite matrix compressing an unbounded sequence), while the KV cache in softmax attention stores every token exactly.

Recent hybrid architectures address this by interleaving softmax attention layers with linear/GDN layers — using softmax for recall-critical layers and linear for the rest.

## Inference

### Decode: constant time per step

The killer feature for serving:

$$
\text{Softmax decode step:} \quad O(s \cdot d) \quad \text{(read entire KV cache)}
$$
$$
\text{GDN decode step:} \quad O(d^2) \quad \text{(update and query fixed-size state)}
$$

No KV cache to read — the entire history is compressed into the state matrix $S$. At $s = 128\text{K}$, this is orders of magnitude faster.

### No KV cache

Linear attention and GDN maintain a **recurrent state** instead of a KV cache:

| | Softmax attention | Linear attention / GDN |
|---|---|---|
| **Memory per layer** | $O(s \cdot d)$ — grows with sequence | $O(d^2)$ — fixed |
| **Decode cost per step** | $O(s \cdot d)$ — grows with sequence | $O(d^2)$ — fixed |

No paged attention needed. No block tables. No eviction. The state is a fixed-size tensor that gets updated in-place.

### Prefill: parallel computation

During prefill, the chunkwise parallel form is used — similar performance to FlashAttention for moderate sequence lengths. The state is built up chunk-by-chunk.

### Kernel support

In vLLM, linear attention and GDN have dedicated backends:

- **LINEAR_ATTN**: for linear attention layers
- **GDN_ATTN**: for Gated DeltaNet layers with causal conv1d
- **MAMBA1_ATTN** / **MAMBA2_ATTN**: for SSM layers (similar recurrent structure)

These handle the recurrent state update and the parallel training form.

## Trade-offs

| | Quality | Decode latency | Memory | Long-context |
|---|---|---|---|---|
| **Softmax + FlashAttention** | Best | $O(s)$ | $O(s)$ KV cache | Exact recall |
| **Linear Attention** | Reduced | $O(1)$ | $O(d^2)$ state | Lossy (no exact recall) |
| **GDN** | Better than linear | $O(1)$ | $O(d^2)$ state | Better retention via gating |
| **Hybrid (softmax + GDN)** | Near softmax | Mixed | Mixed | Best of both |

**The fundamental trade-off**: softmax attention stores every token exactly (KV cache) at the cost of $O(s)$ decode. Linear/GDN compress the entire history into a fixed-size state at the cost of lossy recall. There is no free lunch — $O(1)$ decode requires giving up exact token-level retrieval.

**When to use**:
- Very long sequences where $O(s)$ decode is prohibitive
- Tasks dominated by pattern matching rather than exact recall
- Hybrid architectures where a few softmax layers handle recall

**When NOT to use**:
- Tasks requiring precise fact retrieval from long contexts
- Models where quality is non-negotiable (current frontier LLMs still prefer softmax)

## Adoption

Linear attention and GDN are gaining traction but haven't displaced softmax:

- **RWKV** (2023): RNN-like architecture using linear attention, competitive with Transformers at moderate scale
- **Mamba** (2023) [[3]](#ref-3): SSM architecture with selective state updates, influential design
- **Gated DeltaNet** (2024) [[2]](#ref-2): combines linear attention with delta rule and gating
- **Hybrid models**: Jamba (AI21), Zamba, Griffin (Google) — interleave attention and recurrent layers

The consensus is moving toward **hybrids** — use softmax attention where recall matters, linear/recurrent layers where it doesn't. Pure linear-attention models haven't matched pure Transformer quality at frontier scale, but they offer compelling efficiency for specific deployment scenarios.

## References

<span id="ref-1">[1]</span> Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). *ICML 2020*.

<span id="ref-2">[2]</span> Yang, S., Wang, B., et al. (2024). [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). *arXiv preprint*.

<span id="ref-3">[3]</span> Gu, A. & Dao, T. (2023). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752). *arXiv preprint*.

*Last updated: March 2026*
