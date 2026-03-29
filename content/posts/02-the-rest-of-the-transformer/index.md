---
title: "The Rest of the Transformer"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["transformer", "ffn", "layernorm", "residual", "positional-encoding"]
description: "Beyond attention — the feed-forward network, residual connections, layer normalization, and positional encoding that complete the Transformer block."
weight: 2
---

## TL;DR

Attention gets the headlines, but a Transformer layer is more than $Q K^\top V$. The feed-forward network (FFN) provides per-token nonlinear transformation, residual connections enable gradient flow through deep stacks, layer normalization stabilizes training, and positional encoding gives the model a sense of token order. This post covers each component and traces the full forward pass through a single Transformer block.

## Motivation

The [previous post]({{< relref "01-multi-head-attention" >}}) covered Multi-Head Attention — the mechanism that lets tokens attend to each other. But attention alone is a linear operation over values (a weighted sum). Without the remaining components, a stack of attention layers would collapse into a single linear transformation.

Each component serves a distinct role:

- **FFN**: the only source of per-token nonlinearity and the majority of model parameters
- **Residual connections**: let gradients flow directly through 80+ layers without vanishing
- **Layer normalization**: keeps activation magnitudes stable across layers
- **Positional encoding**: breaks the permutation symmetry — without it, attention treats "the cat sat" and "sat the cat" identically

## The Full Forward Pass

Before diving into each component, here's the complete flow through one decoder layer. Modern models (Llama, DeepSeek, Mistral) use the **Pre-Norm** variant:

$$
\mathbf{h} = \mathbf{x} + \text{MHA}(\text{Norm}(\mathbf{x}))
$$
$$
\mathbf{y} = \mathbf{h} + \text{FFN}(\text{Norm}(\mathbf{h}))
$$

In pseudocode:

```
# Input: x of shape (seq_len, hidden_dim)

# Attention sub-layer
h = x + MHA(RMSNorm(x))

# FFN sub-layer
y = h + FFN(RMSNorm(h))

# Output: y of shape (seq_len, hidden_dim) → input to next layer
```

Two sub-layers, each following the same pattern: **normalize → transform → add residual**. The output has the same shape as the input, so layers can be stacked arbitrarily deep.

## Feed-Forward Network (FFN)

### The original FFN

The Transformer's FFN [[1]](#ref-1) is a simple two-layer MLP applied independently to each token:

$$
\text{FFN}(\mathbf{x}) = W^{\text{down}} \; \text{ReLU}(W^{\text{up}} \mathbf{x})
$$

where $W^{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ expands to a larger intermediate dimension and $W^{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ contracts back. Typically $d_{\text{ff}} = 4d$.

This is the per-token nonlinearity — while attention mixes information *across* tokens, the FFN transforms each token's representation *independently*. It's also where most parameters live: with $d = 4096$ and $d_{\text{ff}} = 16384$, the two FFN matrices contain $2 \times 4096 \times 16384 \approx 134\text{M}$ parameters per layer, compared to $\sim 67\text{M}$ for MHA.

### Gated FFN (SwiGLU)

Most modern models replace ReLU with a gated variant called SwiGLU [[2]](#ref-2):

$$
\text{FFN}(\mathbf{x}) = W^{\text{down}} \left( \text{SiLU}(W^{\text{gate}} \mathbf{x}) \odot W^{\text{up}} \mathbf{x} \right)
$$

where $\odot$ is element-wise multiplication and $\text{SiLU}(z) = z \cdot \sigma(z)$ is the Sigmoid Linear Unit. Now there are **three** weight matrices:

- $W^{\text{gate}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ — produces the gate signal
- $W^{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ — produces the values
- $W^{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$ — projects back to hidden dim

The gate branch controls *how much* of the value branch passes through, element-wise. This gives the model more expressivity — it can learn to selectively suppress dimensions for certain inputs. SwiGLU trains better than ReLU empirically, which is why Llama, DeepSeek, Mistral, and most modern architectures use it.

To compensate for the extra matrix, $d_{\text{ff}}$ is typically reduced to $\frac{8}{3}d$ (rounded to a multiple of 256 for hardware efficiency) to keep total parameter count similar.

### Why the FFN matters for inference

During decode, the FFN weight matrices must be loaded from HBM for every token. With three matrices of size $d \times d_{\text{ff}}$, the FFN is often the dominant memory-bandwidth cost per layer — even more than the attention KV cache reads at short-to-moderate sequence lengths.

This is also why MoE (covered in the [next post]({{< relref "05-moe" >}})) is so impactful: it replaces the single large FFN with many smaller experts, reducing the per-token FFN cost while increasing total model capacity.

## Residual Connections

### The idea

Each sub-layer's output is *added* to its input, not replacing it:

$$
\mathbf{h} = \mathbf{x} + f(\mathbf{x})
$$

This creates a **skip connection** — the gradient can flow directly from layer $L$ back to layer 0 through the addition, bypassing the nonlinearities. Without residuals, gradients through 80+ layers of attention and FFN would vanish or explode.

### Why it enables depth

Consider the gradient of the output with respect to the input through $L$ residual layers:

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = I + \frac{\partial f_L}{\partial \mathbf{x}} + \frac{\partial f_L}{\partial \mathbf{x}} \frac{\partial f_{L-1}}{\partial \mathbf{x}} + \cdots
$$

The identity term $I$ guarantees a minimum gradient magnitude of 1, regardless of how many layers the signal passes through. Each layer contributes an *additive* correction rather than a multiplicative transformation.

### Tensor parallelism interaction

With tensor parallelism, the residual add happens *after* the all-reduce that combines partial results from different GPUs. This is why the all-reduce cannot be easily deferred — the residual needs the full hidden-dim vector.

## Layer Normalization

### Why normalize?

Without normalization, activation magnitudes drift as they pass through layers — growing or shrinking unpredictably. This makes training unstable (exploding/vanishing activations) and learning rate sensitive.

### LayerNorm vs RMSNorm

**LayerNorm** [[3]](#ref-3) normalizes by subtracting the mean and dividing by the standard deviation:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where $\mu = \text{mean}(\mathbf{x})$, $\sigma^2 = \text{var}(\mathbf{x})$, and $\gamma$, $\beta$ are learned per-dimension scale and shift parameters.

**RMSNorm** [[4]](#ref-4) simplifies this by dropping the mean shift:

$$
\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x}) + \epsilon}
$$

where $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$.

RMSNorm is cheaper (no mean computation, no shift) and performs comparably in practice. Most modern models use RMSNorm: Llama, DeepSeek, Mistral, Gemma.

### Pre-Norm vs Post-Norm

The original Transformer [[1]](#ref-1) applied normalization *after* the sub-layer:

$$
\mathbf{h} = \text{Norm}(\mathbf{x} + \text{MHA}(\mathbf{x})) \quad \text{(Post-Norm)}
$$

Modern models apply it *before*:

$$
\mathbf{h} = \mathbf{x} + \text{MHA}(\text{Norm}(\mathbf{x})) \quad \text{(Pre-Norm)}
$$

Pre-Norm is more stable for deep models — the residual stream stays unnormalized, providing a cleaner gradient path. Post-Norm can produce slightly better final quality but requires careful learning rate warmup and is harder to train at depth.

### Tensor parallelism interaction

Normalization operates on the **full hidden dimension**. With tensor parallelism (where activations may be sharded), this means either:

- All-gather the full vector, then normalize (what most implementations do)
- Compute statistics across shards via a small all-reduce (possible for RMSNorm since $\text{RMS}(\mathbf{x})$ decomposes into per-shard sums of squares)

The first approach is simpler; the second saves a bit of redundant computation but adds complexity. In practice, normalization is so cheap relative to the GEMMs that the choice has negligible impact on throughput.

## Positional Encoding

### The problem

Self-attention is **permutation-equivariant** — swapping two tokens in the input produces the same swap in the output. The attention scores $Q K^\top$ depend on content, not position. Without positional information, "the cat chased the dog" and "the dog chased the cat" would produce identical attention patterns.

### Absolute positional encoding (original)

The original Transformer [[1]](#ref-1) adds a fixed sinusoidal signal to the input embeddings:

$$
\text{PE}(p, 2i) = \sin\left(\frac{p}{10000^{2i/d}}\right), \quad \text{PE}(p, 2i+1) = \cos\left(\frac{p}{10000^{2i/d}}\right)
$$

where $p$ is the position and $i$ is the dimension index. This gives each position a unique signature. Alternatively, learned position embeddings (a lookup table) achieve similar results.

The limitation: absolute encodings are fixed at training time. Generalizing to sequences longer than the training length is difficult.

### Rotary Position Embeddings (RoPE)

RoPE [[5]](#ref-5) encodes position by *rotating* the query and key vectors:

$$
\text{RoPE}(Q_h, p) = R_p \cdot Q_h, \quad \text{RoPE}(K_h, p) = R_p \cdot K_h
$$

where $R_p$ is a rotation matrix that depends on position $p$. The key property: the dot product $Q_h^\top K_h$ after rotation depends on the *relative* position $(p_q - p_k)$, not the absolute positions. This enables better length generalization.

RoPE is used by Llama, DeepSeek, Mistral, Qwen, and most modern architectures. It interacts directly with attention — particularly with MLA, where the key is split into RoPE and non-RoPE components (covered in the [MLA post]({{< relref "04-mla" >}})).

A deeper treatment of RoPE and other position encoding schemes is planned for Appendix A (Position Encodings).

## Putting It Together

A complete decoder-only Transformer:

```
Input tokens
  ↓
Token embedding + positional encoding
  ↓
┌─── Decoder Layer 1 ─────────────────────────┐
│  x_norm = RMSNorm(x)                        │
│  attn_out = MHA(x_norm)     ← attention      │
│  h = x + attn_out           ← residual       │
│  h_norm = RMSNorm(h)                         │
│  ffn_out = FFN(h_norm)      ← feed-forward   │
│  y = h + ffn_out            ← residual       │
└──────────────────────────────────────────────┘
  ↓
  ... (repeat for L layers)
  ↓
RMSNorm(y)
  ↓
Linear projection → vocabulary logits
  ↓
Token prediction
```

Each layer receives a $(s, d)$ tensor and produces a $(s, d)$ tensor. The attention sub-layer mixes across tokens; the FFN sub-layer transforms each token independently. Residuals and normalization keep the signal stable through all $L$ layers.

### Parameter distribution

For a typical model (e.g., Llama 2 7B: $d = 4096$, $d_{\text{ff}} = 11008$, $H = 32$, $L = 32$):

| Component | Params per layer | % of layer |
|---|---|---|
| QKV projection ($W^Q, W^K, W^V$) | $3 \times d \times d = 50\text{M}$ | ~36% |
| Output projection ($W^O$) | $d \times d = 17\text{M}$ | ~12% |
| FFN ($W^{\text{gate}}, W^{\text{up}}, W^{\text{down}}$) | $3 \times d \times d_{\text{ff}} = 135\text{M}$ | ~50% |
| Norms ($\gamma$ parameters) | $2 \times d = 8\text{K}$ | ~0% |

The FFN dominates — roughly half the parameters. This is why MoE's replacement of the FFN has such a large impact on model capacity.

## Trade-offs

| Component | Role | Modern choice | Why |
|---|---|---|---|
| FFN activation | Nonlinearity | SwiGLU (gated) | Better quality than ReLU, slight param overhead |
| Normalization | Stability | RMSNorm | Cheaper than LayerNorm, comparable quality |
| Norm placement | Gradient flow | Pre-Norm | More stable at depth than Post-Norm |
| Position encoding | Token order | RoPE | Relative encoding, better length generalization |

These choices are largely settled — the community has converged on SwiGLU + RMSNorm + Pre-Norm + RoPE as the standard recipe. The innovations in Parts 2–4 of this series focus on attention and the FFN itself, building on this foundation.

## References

<span id="ref-1">[1]</span> Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS 2017*.

<span id="ref-2">[2]</span> Shazeer, N. (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202). *arXiv preprint*.

<span id="ref-3">[3]</span> Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). [Layer Normalization](https://arxiv.org/abs/1607.06450). *arXiv preprint*.

<span id="ref-4">[4]</span> Zhang, B. & Sennrich, R. (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). *NeurIPS 2019*.

<span id="ref-5">[5]</span> Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). *arXiv preprint*.

*Last updated: March 2026*
