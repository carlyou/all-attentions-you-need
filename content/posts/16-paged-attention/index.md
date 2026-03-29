---
title: "Paged Attention & KV Cache Management"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["serving", "paged-attention", "kv-cache", "block-table", "scheduler"]
description: "Virtual memory for KV cache: block allocation, block tables, the scheduler as sole owner of GPU memory."
weight: 16
---

## TL;DR

<!-- TODO -->

## Motivation

<!-- TODO -->

## Design

<!-- TODO: contiguous vs paged allocation, block pool, block tables, on-demand allocation, immediate freeing, internal fragmentation, preemption/swapping -->

## Implementation

<!-- TODO: block manager, scheduler integration, scatter-gather reads, kernel support -->

## Trade-offs

<!-- TODO -->

## References

*Last updated: March 2026*
