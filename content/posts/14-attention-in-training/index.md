---
title: "Attention in Training"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["training", "gradient-checkpointing", "mixed-precision", "backward-pass"]
description: "Memory profile, gradient checkpointing, mixed precision, and the backward pass through FlashAttention."
weight: 1061
---

## TL;DR

<!-- TODO -->

## Motivation

<!-- TODO -->

## Design

<!-- TODO: activation memory vs weight memory, gradient checkpointing (recompute vs store), mixed precision (FP32 master weights, BF16 forward/backward), FA backward pass -->

## Implementation

<!-- TODO: torch.compile interaction, activation memory estimation, checkpointing strategies -->

## Trade-offs

<!-- TODO -->

## References

*Last updated: March 2026*
