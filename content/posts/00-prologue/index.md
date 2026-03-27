---
title: "Prologue: The Attention Landscape"
date: 2026-03-26
draft: true
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

The problem is that these variants are scattered across papers, blog posts,
and framework docs. It's hard to see how they relate, what trade-offs they
make, or which ones actually matter in practice.

This series puts them all in one place. Each post covers one mechanism with
the same structure, so you can compare them directly.

## The Roadmap

### Part 1: The Foundations

- [ ] **[WIP] Multi-Head Attention (MHA)**
    - the baseline: Q/K/V projections, multi-head parallelism, and why the KV cache becomes a problem
- [ ] **Multi-Query Attention (MQA) & Group Query Attention (GQA)**
    - sharing key/value heads across query heads to shrink the KV cache without destroying quality

### Part 2: The KV Cache Problem

- [ ] **Multi-head Latent Attention (MLA)** — DeepSeek's approach: compressing KV into a low-rank latent space
- [ ] **Paged Attention** — how vLLM manages KV cache memory like virtual memory pages

### Part 3: The Quadratic Problem

- [ ] **Sliding Window Attention** — limiting attention to a local window to cap memory at O(s·w) instead of O(s²)
- [ ] **Flash Attention (v1/v2/v3)** — IO-aware exact attention: same math, dramatically less memory
- [ ] **Sparse Attention** — from Longformer/BigBird to DeepSeek's NSA/DSA: attending to only the tokens that matter

### Part 4: Beyond Softmax

- [ ] **Linear Attention & Gated DeltaNet (GDN)** — replacing softmax with linear maps and gated delta rules for O(s) attention
- [ ] **Differential Attention** — computing attention as the difference of two softmax maps for noise cancellation
- [ ] **Residual Attention** — Moonshot AI's approach: augmenting attention with explicit residual connections

### Part 5: Scaling Out

- [ ] **Ring Attention** — distributing attention computation across devices for near-infinite context

### Appendix

- [ ] **Position Encodings (RoPE, ALiBi, NoPE)** — how models encode token position, entangled with every variant above
