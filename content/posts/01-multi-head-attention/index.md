---
title: "Multi-Head Attention (MHA)"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["attention", "transformer", "mha"]
description: "The original attention mechanism from 'Attention Is All You Need' — how it works, why the KV cache blows up, and where it stands today."
---

## TL;DR

Multi-Head Attention (MHA) is the original attention mechanism from the Transformer. It projects input into multiple independent query/key/value heads, computes scaled dot-product attention in parallel, and concatenates the results. Simple and effective, but its per-head KV cache grows linearly with sequence length, motivating the variants that follow.

## Motivation

In 2017, Vaswani et al. introduced the Transformer architecture in
*"Attention Is All You Need"* [[1]](#ref-1), replacing recurrence entirely with
self-attention. Multi-Head Attention (MHA) is the core building block.

<!-- TODO: brief history — RNN/LSTM bottleneck, Bahdanau attention, then full self-attention -->

## Mechanism

Given input sequence $X \in \mathbb{R}^{s \times d}$, MHA projects it into
queries, keys, and values for each head $h$:

$$
Q_h = X W_h^Q, \quad K_h = X W_h^K, \quad V_h = X W_h^V
$$

where $W_h^Q, W_h^K \in \mathbb{R}^{d \times d_k}$ and
$W_h^V \in \mathbb{R}^{d \times d_v}$, with $d_k = d_v = d / n_{\text{heads}}$.

Each head computes scaled dot-product attention:

$$
\text{Attention}(Q_h, K_h, V_h) = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_k}}\right) V_h
$$

The outputs are concatenated and projected:

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O
$$

![Multi-Head Attention architecture: input X is projected through per-head weight matrices into Q, K, V, each head computes scaled dot-product attention, outputs are concatenated and projected through W^O.](mha-diagram.svg)

## Training

<!-- TODO:
- Memory: O(s² · n_heads) for attention scores
- Compute: O(s² · d) per layer
- Parallelism: heads are embarrassingly parallel
-->

## Inference

<!-- TODO:
- KV cache: each head stores K, V tensors → total cache = 2 × n_layers × n_heads × d_k × seq_len × batch
- With a concrete example: 32 layers, 32 heads, d_k=128, seq_len=4096
- This motivates MQA/GQA (next post)
- Kernel support: FlashAttention, FlashInfer, xformers
- TP: heads shard trivially across GPUs
-->

## Trade-offs

<!-- TODO: comparison table — MHA vs MQA vs GQA on KV cache size, quality, parallelism -->

## Adoption

MHA remains the default in many model families, though most frontier models
have moved to GQA or MLA for inference efficiency. Still used as-is in:

- BERT, GPT-2, original GPT-3
- Smaller fine-tuned models where KV cache isn't the bottleneck

<!-- TODO: update with current framework support notes -->

## References

<span id="ref-1">[1]</span> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems, 30*.

*Last updated: March 2026*
