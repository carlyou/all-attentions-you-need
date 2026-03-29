---
title: "Context & Ring Parallelism"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["parallelism", "context-parallel", "ring-attention", "lse", "cp"]
description: "Splitting the KV context across devices and merging results with LSE — enabling near-infinite context length."
weight: 13
---

## TL;DR

<!-- TODO -->

## Motivation

<!-- TODO -->

## Design

<!-- TODO: CP (KV sharding, partial attention + LSE merge), Ring Attention (ring topology, overlapping compute and communication) -->

## Implementation

<!-- TODO: merge_attn_states, all-gather patterns, chunked prefill connection -->

## Trade-offs

<!-- TODO -->

## References

*Last updated: March 2026*
