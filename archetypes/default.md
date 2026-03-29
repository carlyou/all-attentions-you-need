---
title: "{{ replace .File.ContentBaseName "-" " " | title }}"
date: {{ .Date }}
draft: true
math: true
toc: true
tags: []
description: ""
---

## TL;DR

<!-- 2-3 sentence summary of what this topic is and why it matters -->

## Motivation

<!-- What problem does this solve? What came before? What bottleneck does it address? -->

<!--
Pick the middle sections that fit your topic.
Delete the ones that don't apply.

FOR MECHANISM POSTS (MHA, MQA/GQA, MLA, MoE, Sliding Window, Sparse, Linear Attention):
-->

## Mechanism

<!-- Architecture, equations, diagrams -->

## Training

<!-- Memory, compute, stability implications during training -->

## Inference

<!-- KV cache, serving behavior, kernel support -->

<!--
FOR SYSTEMS POSTS (FlashAttention, Parallelism, Paged Attention, Batching, Backends, etc.):

## Design

Key ideas, algorithms, data structures — the theory.

## Implementation

How it works on actual hardware or in actual code — the practice.
-->

## Trade-offs

<!-- Comparison table vs relevant alternatives -->

## References

*Last updated: {{ .Date | time.Format "January 2006" }}*
