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

<!-- 2-3 sentence summary of what this mechanism is and why it matters -->

## Motivation

<!-- What problem does this solve? What came before? -->

## Mechanism

<!-- Architecture, equations, diagram -->

## Training

<!-- Memory, compute, stability implications -->

## Inference

<!-- KV cache, kernel support, TP, quantization -->

## Trade-offs

<!-- Comparison table vs relevant alternatives -->

## Adoption

<!-- Which models and frameworks use it today -->

## References

*Last updated: {{ .Date | time.Format "January 2006" }}*
