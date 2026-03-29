---
title: "Prologue: The Attention Landscape"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["attention", "overview"]
description: "A map of the attention mechanism landscape — what this series covers, how the posts connect, and how to read them based on what you care about."
weight: 0
---

## Why This Series?

Attention is the engine of modern AI. Since the original Transformer in 2017,
the core idea — *let every token look at every other token* — has spawned
dozens of variants, each attacking a different bottleneck: memory, compute,
sequence length, or serving cost.

But understanding attention mechanisms in isolation isn't enough. The gap
between a paper's equations and a production serving system is vast — tiled
GPU kernels, paged memory management, multi-GPU parallelism, quantized
arithmetic. This series bridges that gap, covering attention from math to
metal.

The problem is that these topics are scattered across papers, blog posts,
and framework docs. It's hard to see how they relate, what trade-offs they
make, or which ones actually matter in practice.

This series puts them all in one place. Each post covers one topic with
the same structure, so you can compare them directly.

## Notation

The same symbols are used consistently across every post in this series.

### Dimensions

| Symbol | Meaning |
|---|---|
| $d$ | Hidden dimension (model width) |
| $d_k$ | Key/query head dimension ($= d / H$ in standard MHA) |
| $d_v$ | Value head dimension (often $= d_k$) |
| $d_{\text{ff}}$ | FFN intermediate dimension |
| $d_c$ | MLA latent (compressed KV) dimension |
| $d_{\text{pe}}$ | RoPE positional encoding dimension |
| $d_\phi$ | Linear attention feature map dimension |

### Counts

| Symbol | Meaning |
|---|---|
| $s$ | Sequence length (number of tokens — input, cached, or total depending on context) |
| $H$ | Number of query/attention heads |
| $G$ | Number of KV head groups (GQA); $G = H$ is MHA, $G = 1$ is MQA |
| $L$ | Number of transformer layers |
| $B$ | Batch size (number of concurrent requests) |
| $E$ | Number of MoE experts |
| $w$ | Sliding window size |

### Top-k

The symbol $k$ appears in two contexts:

| Context | Meaning | Typical value |
|---|---|---|
| MoE routing | Number of experts activated per token | 8 |
| Sparse attention | Number of tokens selected per query | 2,048 |

Where ambiguous, the post will clarify which $k$ is meant.

### Tensors and matrices

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | Hidden state / input to a layer |
| $Q, K, V$ | Query, key, value tensors |
| $W^Q, W^K, W^V, W^O$ | Projection weight matrices |
| $W^{\text{up}}, W^{\text{down}}$ | MLA up/down projection (also written $W^{\text{up}}_Q$, $W^{\text{up}}_K$, etc.) |
| $W^{\text{gate}}$ | FFN gating projection (SwiGLU) |
| $\mathbf{c}$ | MLA latent vector (compressed KV, cached) |
| $\mathbf{c}_q$ | MLA compressed query latent |
| $S$ | Attention score matrix ($Q K^\top$), or recurrent state in linear attention (context will clarify) |

### Acronyms

| Acronym | Meaning |
|---|---|
| MHA | Multi-Head Attention |
| MQA | Multi-Query Attention |
| GQA | Grouped-Query Attention |
| MLA | Multi-head Latent Attention |
| MoE | Mixture of Experts |
| FFN | Feed-Forward Network |
| KV cache | Key-Value cache (stored K and V from prior tokens) |
| HBM | High Bandwidth Memory (GPU off-chip memory, ~TB/s) |
| SRAM | Static RAM (GPU on-chip memory per SM, ~19 TB/s) |
| TP | Tensor Parallelism |
| PP | Pipeline Parallelism |
| EP | Expert Parallelism |
| CP | Context Parallelism |
| DP | Data Parallelism |
| RoPE | Rotary Position Embedding |
| LSE | Log-Sum-Exp (softmax normalization statistic) |
| SM | Streaming Multiprocessor (GPU compute unit) |

## The Roadmap

### Part 1: Attention Is All You Need

- [ ] **Multi-Head Attention (MHA)**
    - the baseline: Q/K/V projections, multi-head parallelism, and why the KV cache becomes a problem
- [ ] **The Rest of the Transformer**
    - FFN (dense and gated/SwiGLU), residual connections, layer normalization, positional encoding, and the full forward pass end-to-end

### Part 2: Architecture Innovations

- [ ] **Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)**
    - sharing KV heads across query heads to shrink the cache without destroying quality
- [ ] **Multi-head Latent Attention (MLA)**
    - compressing KV into a low-rank latent space, the absorption trick, and why prefill and decode take completely different code paths
- [ ] **Mixture of Experts (MoE)**
    - sparse FFN with learned routing: 256 experts but each token uses only 8, the permute-compute-unpermute pattern, and shared experts

### Part 3: Reducing Attention Cost

- [ ] **Sliding Window & Local Attention**
    - limiting attention to a fixed window to cap memory, from Longformer and BigBird to Mistral's sliding window
- [ ] **Sparse Attention (DeepSeek NSA/DSA)**
    - learned dynamic token selection: a lightweight MQA indexer picks the top-k most relevant tokens per query
- [ ] **Linear Attention & Gated DeltaNet (GDN)**
    - replacing softmax with linear maps and gated delta rules for O(s) attention

### Part 4: Flash Attention

- [ ] **Flash Attention v1 & v2**
    - tiling, online softmax, the log-sum-exp trick, why swapping the loop order matters, and causal masking skip
- [ ] **Flash Attention v3 & v4**
    - Hopper: TMA async loads, WGMMA, two-stage pipelining;
      Blackwell: ping-pong scheduling, TMEM, software exponential emulation, lazy rescaling

### Part 5: Parallelism Done Right

- [ ] **Tensor & Pipeline Parallelism**
    - splitting heads and weight matrices across GPUs, column/row parallel linear layers, all-reduce patterns
- [ ] **Expert Parallelism & MoE at Scale**
    - distributing experts across GPUs, all-to-all communication, load balancing
- [ ] **Context & Ring Parallelism**
    - splitting the KV context across devices, LSE-based merging for mathematically exact results

### Part 6: Training at Scale

- [ ] **Attention in Training**
    - memory profile (activations vs weights), gradient checkpointing, mixed precision, backward pass through FlashAttention
- [ ] **Distributed Training (ZeRO & FSDP)**
    - ZeRO stages 1/2/3, FSDP as PyTorch-native ZeRO-3, how attention layers are sharded during training

### Part 7: Serving Infrastructure

- [ ] **Paged Attention & KV Cache Management**
    - virtual memory for KV cache: block allocation, block tables, the scheduler as sole owner of GPU memory
- [ ] **Batching & Scheduling**
    - continuous batching, mixed prefill-decode, chunked prefill, ragged batching, and the CPU-GPU scheduling loop
- [ ] **Speculative Decoding**
    - draft-then-verify for multi-token generation, rejection sampling, and variants: Medusa, EAGLE, layer-skipping, n-gram
- [ ] **Attention Backends**
    - FlashAttention, FlashInfer, FlashMLA, CUTLASS, Triton, cuDNN — how vLLM selects and dispatches to the right kernel

### Appendices

- [ ] **GPU Architecture Primer** — compute hierarchy (threads to SMs), memory hierarchy (registers to HBM), data movement (TMA, NCCL), and what each GPU generation added for attention
- [ ] **Position Encodings (RoPE, ALiBi, NoPE)** — how models encode token position, entangled with every attention variant
- [ ] **Quantization for Attention** — FP8, FP4, scale granularity, and fusing quantization into attention kernels
- [ ] **Differential & Residual Attention** — noise cancellation via dual softmax maps, and explicit residual connections
