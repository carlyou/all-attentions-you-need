---
title: "Batching & Scheduling"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["serving", "batching", "scheduling", "chunked-prefill", "ragged", "continuous-batching"]
description: "Continuous batching, mixed prefill-decode, chunked prefill, ragged batching, and the CPU-GPU scheduling loop."
weight: 17
---

## TL;DR

<!-- TODO -->

## Motivation

<!-- TODO -->

## Design

<!-- TODO: static vs continuous batching, mixed prefill-decode, chunked prefill (has_context, merge_attn_states), ragged batching (cu_seq_lens, no padding), the iteration loop (CPU scheduler → GPU forward → sampling → back to CPU) -->

## Implementation

<!-- TODO: CUDA graphs, scheduling overlap, metadata management -->

## Trade-offs

<!-- TODO -->

## References

*Last updated: March 2026*
