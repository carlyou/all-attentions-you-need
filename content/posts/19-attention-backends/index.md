---
title: "Attention Backends"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["serving", "backends", "flashattention", "flashinfer", "flashmla", "cutlass", "triton"]
description: "FlashAttention, FlashInfer, FlashMLA, CUTLASS, Triton, cuDNN — how vLLM selects and dispatches to the right kernel."
weight: 19
---

## TL;DR

<!-- TODO -->

## Motivation

<!-- TODO -->

## Design

<!-- TODO: the abstraction layers (backend → library → custom op → kernel → hardware), backend selection pipeline, MLA vs general backends, sparse backends -->

## Implementation

<!-- TODO: vLLM backend registry, kernel selection logic, block_size constraints, CUDA graph support levels -->

## Trade-offs

<!-- TODO -->

## References

*Last updated: March 2026*
