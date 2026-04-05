---
title: "Appendix C: Quantization for Attention"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["quantization", "fp8", "fp4", "nvfp4", "scale", "fusion", "ptq", "qat", "int8", "smoothquant", "gptq", "awq"]
description: "From number formats and scaling strategies through PTQ, QAT, and fusing quantization into attention kernels — everything you need to understand quantized inference."
weight: 1093
---

## TL;DR

Quantization maps FP32/BF16 values to fewer bits (INT8/FP8/FP4), trading precision for throughput and memory savings. The key decisions are the **number format** (integer vs floating-point, how many exponent vs mantissa bits), the **scale granularity** (per-tensor, per-channel, per-group, per-token), and **when quantization happens** (post-training calibration vs quantization-aware training). For attention specifically, quantization intersects with the $QK^\top$ and $\text{score} \times V$ GEMMs, the KV cache, and kernel fusion — where quantize/dequantize steps are folded into the attention kernel to eliminate extra memory round-trips. Hopper introduced FP8 Tensor Cores enabling FlashAttention v3's FP8 path; Blackwell added FP4 for FlashAttention v4. Frontier models like DeepSeek-V3 now train end-to-end in FP8.

## Motivation

**The bandwidth wall.** Attention decode is memory-bandwidth-bound: the bottleneck is reading the KV cache from HBM. Recall the KV cache formula from earlier in this series:

$$
\text{KV cache} = 2 \times L \times H \times d_k \times s \times B \times \text{sizeof(dtype)}
$$

Halving $\text{sizeof(dtype)}$ from 2 bytes (BF16) to 1 byte (FP8) nearly halves both the memory footprint and the bandwidth cost. This is the most direct lever for improving decode throughput — no architectural changes, no approximations to the attention pattern, just storing and moving fewer bytes per value.

**Tensor Core throughput scaling.** Each GPU generation adds lower-precision datapaths that roughly double peak TFLOPS: FP16 on Ampere, FP8 on Hopper, FP4 on Blackwell. To actually use those faster paths, your operands must be in the corresponding format — you can't feed BF16 tensors to the FP8 Tensor Cores. So quantization isn't just a memory optimization; it's the gateway to the hardware's fastest compute. See [Appendix A]({{< relref "appendix-a-gpu-architecture" >}}) for the full Tensor Core progression.

**The challenge.** Attention has unique quantization sensitivity compared to a typical feed-forward GEMM. Softmax is an exponential — small errors in the $QK^\top$ scores get amplified into large errors in the attention weights. The KV cache persists quantization error across the full sequence length, so a rounding error introduced at position 0 still affects generation at position 10,000. Generic GEMM quantization recipes (calibrate, scale, round) don't account for these attention-specific concerns — which is exactly what this post addresses.

## Number Formats

<!-- Section content: Task 2 -->

## Scaling Strategies

<!-- Section content: Task 3 -->

## Post-Training Quantization (PTQ)

<!-- Section content: Task 4 -->

## Quantization-Aware Training (QAT)

<!-- Section content: Task 5 -->

## Hardware Support

<!-- Section content: Task 6 -->

## Where Quantization Hits Attention

<!-- Section content: Task 7 -->

## Kernel Fusion

<!-- Section content: Task 8 -->

## Quantization in MLA Paths

<!-- Section content: Task 9 -->

## Quantization in FFN / MoE

<!-- Section content: Task 10 -->

## Trade-offs

<!-- Section content: Task 11 -->

## Adoption

<!-- Section content: Task 11 -->

## References

*Last updated: April 2026*
