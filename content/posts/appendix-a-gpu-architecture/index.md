---
title: "Appendix A: GPU Architecture Primer"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["gpu", "cuda", "sm", "hbm", "sram", "tensor-cores", "tma", "nccl"]
description: "The GPU hardware that attention runs on — compute hierarchy (threads to SMs), memory hierarchy (registers to HBM), data movement (TMA, NCCL), and what each GPU generation added."
weight: 20
---

## TL;DR

Understanding attention kernels requires understanding the hardware they run on. This appendix covers the GPU compute hierarchy (thread → warp → thread block → SM → grid), memory hierarchy (registers → SRAM → L2 → HBM), data movement mechanisms (TMA, NCCL, NVLink), and what each NVIDIA GPU generation (Ampere, Hopper, Blackwell) added that matters for attention.

## Compute Hierarchy

<!-- TODO:
- Thread: single execution unit
- Warp: 32 threads executing in lockstep (SIMT)
- Warp Group: 4 warps (128 threads, Hopper+) — unit for WGMMA
- Thread Block (CTA): scheduled on one SM, shares SRAM
- Grid: collection of thread blocks launched by one kernel
- SM (Streaming Multiprocessor): the compute unit
  - CUDA cores (FP32/FP64)
  - Tensor Cores (matrix multiply)
  - SFU (special function units: exp, log, sin)
  - Occupancy: how many thread blocks fit on one SM
-->

## Memory Hierarchy

<!-- TODO:
- Registers: per-thread, fastest, ~256 KB per SM
- Shared Memory / SRAM: per-SM, ~192 KB (configurable with L1), ~19 TB/s
- L2 Cache: shared across all SMs, ~40 MB (A100)
- HBM: off-chip, 80 GB (A100) / 80 GB (H100) / 192 GB (B200), ~2-3 TB/s
- TMEM: Blackwell-only, 256 KB per SM, for tensor core accumulators
- Arithmetic intensity / roofline model: FLOPs per byte determines whether you're compute-bound or memory-bound
-->

## Tensor Cores

<!-- TODO:
- Matrix multiply-accumulate units
- Ampere: FP16/BF16/TF32/INT8, wmma instructions
- Hopper: + FP8, WGMMA (warp-group level)
- Blackwell: + FP4, UMMA/tcgen05 (single-thread launch, async, accumulate to TMEM)
- Throughput scaling: ~312 TFLOPS (A100 FP16) → ~990 (H100) → ~2250 (B200)
-->

## Data Movement

<!-- TODO:
### Within a GPU
- TMA (Tensor Memory Accelerator, Hopper+): async HBM↔SRAM, hardware-managed, frees compute warps
- TMA multicast (Blackwell): one HBM load → multiple SMs' SRAM
- DSMEM (Distributed Shared Memory): SM-to-SM within a cluster

### Across GPUs
- NVLink: high-bandwidth GPU-to-GPU (~900 GB/s per GPU, Blackwell)
- PCIe: lower bandwidth (~64 GB/s Gen5)
- NCCL: NVIDIA's collective communication library
- Collective operations: all-reduce, all-gather, reduce-scatter, all-to-all
-->

## GPU Generations: What Matters for Attention

<!-- TODO:
| Generation | GPU | Key additions for attention |
|---|---|---|
| Ampere (SM 80) | A100 | Large HBM (80 GB), TF32 Tensor Cores, FlashAttention v1/v2 target |
| Hopper (SM 90) | H100 | TMA, WGMMA, FP8 Tensor Cores, thread block clusters — FlashAttention v3 target |
| Blackwell (SM 100) | B200 | TMEM, UMMA, TMA multicast, FP4, ping-pong scheduling — FlashAttention v4 target |
-->

## References

*Last updated: March 2026*
