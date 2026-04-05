---
title: "Appendix A: GPU Architecture Primer"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["gpu", "cuda", "sm", "hbm", "sram", "tensor-cores", "tma", "nccl"]
description: "The GPU hardware that attention runs on — compute hierarchy (threads to SMs), memory hierarchy (registers to HBM), data movement (TMA, NCCL), and what each GPU generation added."
weight: 1091
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

**Tensor Cores** are fixed-function matrix multiply-accumulate (MMA) units inside each SM. They operate on small matrix tiles (e.g., $16 \times 16$) and compute $D = A \times B + C$ in a single operation, delivering far higher throughput than scalar CUDA cores for matrix workloads. Every GEMM in a transformer — projections, attention scores, FFN layers — runs on Tensor Cores.

### Throughput by precision

Each GPU generation added lower-precision datapaths, roughly doubling peak throughput at each step:

| Precision | A100 (Ampere) | H100 (Hopper) | B200 (Blackwell) |
|---|---|---|---|
| FP64 | ~19.5 TFLOPS | ~34 TFLOPS | ~45 TFLOPS |
| TF32 | ~156 TFLOPS | ~495 TFLOPS | ~1125 TFLOPS |
| FP16 / BF16 | ~312 TFLOPS | ~990 TFLOPS | ~2250 TFLOPS |
| FP8 | — | ~1979 TFLOPS | ~4500 TFLOPS |
| FP4 | — | — | ~9000 TFLOPS |
| INT8 | ~624 TOPS | ~1979 TOPS | ~4500 TOPS |

*Note: values are approximate peak dense (non-sparsity). Actual throughput depends on tile utilization, memory-boundedness, and kernel efficiency.*

### Accumulator precision

All reduced-precision Tensor Core paths (FP16, BF16, FP8, FP4, INT8) accumulate partial products into an **FP32 accumulator**. The operands may be low-precision, but the running sum is kept at full precision. This is why FP8 GEMMs can match BF16 quality — individual multiply errors stay small and the FP32 accumulation prevents them from compounding. See [Appendix C]({{< relref "appendix-c-quantization" >}}) for a detailed walkthrough of the quantized GEMM dataflow.

### Evolution of the MMA interface

- **Ampere (SM 80):** introduced `wmma` (warp-level matrix multiply-accumulate) instructions. A single warp (32 threads) cooperatively loads a tile from registers and executes the MMA. Supported formats: FP16, BF16, TF32, INT8.
- **Hopper (SM 90):** introduced **WGMMA** (warp-group MMA). A **warp group** (4 warps, 128 threads) issues a single MMA instruction, operating on larger tiles and feeding **FP8 operands directly from shared memory** — no register staging needed. This reduces register pressure and instruction overhead, enabling higher sustained throughput. WGMMA is the instruction that FlashAttention v3 targets for its FP8 attention path.
- **Blackwell (SM 100):** introduced **UMMA** (unified MMA) via the `tcgen05` instruction family. UMMA can be launched by a **single thread**, is fully asynchronous, and supports **FP4 operands**. Results accumulate into **TMEM** (Tensor Memory), a dedicated 256 KB per-SM scratchpad for accumulator state — freeing registers and shared memory for other use. UMMA with TMEM is what enables FlashAttention v4's FP4 path and the ping-pong scheduling pattern on Blackwell.

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
