---
title: "Tensor & Pipeline Parallelism"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["parallelism", "tensor-parallel", "pipeline-parallel", "all-reduce", "tp", "pp"]
description: "Splitting heads and weight matrices across GPUs — column/row parallel linear layers, all-reduce patterns, and pipeline stages."
weight: 1051
---

## TL;DR

<!-- TODO -->

## Motivation

<!-- TODO -->

## Design

<!-- TODO: TP (column-parallel, row-parallel, all-reduce, reduce-scatter/all-gather), PP (layer assignment, pipeline bubbles, micro-batching) -->

## Implementation

<!-- TODO: RowParallelLinear, ColumnParallelLinear, sequence parallelism, NCCL -->

## Trade-offs

<!-- TODO -->

## References

*Last updated: March 2026*
