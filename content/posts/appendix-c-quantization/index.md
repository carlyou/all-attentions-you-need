---
title: "Appendix C: Quantization for Attention"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["quantization", "fp8", "fp4", "nvfp4", "scale", "fusion"]
description: "FP8, FP4, scale granularity, and fusing quantization into attention kernels."
weight: 1093
---

## TL;DR

<!-- TODO -->

## Motivation

<!-- TODO -->

## Mechanism

<!-- TODO: weight vs activation quantization, scale factors (per-tensor, per-channel, per-group, per-token), FP8 formats (E4M3, E5M2), NVFP4, quantize-dequantize flow around GEMMs -->

## Inference

<!-- TODO: quant fusion (eliminate separate kernel launches), connection to issue #35792, where quant happens in MLA paths -->

## Trade-offs

<!-- TODO -->

## References

*Last updated: March 2026*
