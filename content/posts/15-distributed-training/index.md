---
title: "Distributed Training (ZeRO & FSDP)"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["training", "zero", "fsdp", "distributed", "sharding"]
description: "ZeRO stages 1/2/3, FSDP as PyTorch-native ZeRO-3, and how attention layers are sharded during training."
weight: 15
---

## TL;DR

<!-- TODO -->

## Motivation

<!-- TODO -->

## Design

<!-- TODO: ZeRO-1 (optimizer sharding), ZeRO-2 (+ gradient sharding), ZeRO-3 (+ parameter sharding), FSDP as PyTorch-native ZeRO-3, how attention weight matrices are sharded/gathered -->

## Implementation

<!-- TODO: FSDP wrapping policies, communication patterns, interaction with TP -->

## Trade-offs

<!-- TODO -->

## References

*Last updated: March 2026*
