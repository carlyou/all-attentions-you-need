---
title: "Appendix C: Quantization for Attention"
date: 2026-03-26
draft: true
math: true
toc: true
tocDepth: 2
tags: ["quantization", "fp8", "fp4", "nvfp4", "scale", "fusion", "ptq", "qat", "int8", "smoothquant", "gptq", "awq"]
description: "From number formats and scaling strategies through PTQ, QAT, and fusing quantization into attention kernels — everything you need to understand quantized inference."
weight: 1093
---

## TL;DR

Quantization maps FP32/BF16 values to fewer bits (INT8/FP8/FP4), trading precision for throughput and memory savings. The key decisions are the **number format** (integer vs floating-point, how many exponent vs mantissa bits), the **scale granularity** (per-tensor, per-channel, per-group, per-token), and **when quantization happens** (post-training calibration vs quantization-aware training). For attention specifically, quantization intersects with the $QK^\top$ and $\text{score} \times V$ GEMMs, the KV cache, and kernel fusion — where quantize/dequantize steps are folded into the attention kernel to eliminate extra memory round-trips. Hopper introduced FP8 Tensor Cores enabling FlashAttention v3's FP8 path; Blackwell added FP4 for FlashAttention v4. Frontier models like DeepSeek-V3 now train end-to-end in FP8.

## Motivation

**The bandwidth wall.** Attention decode is memory-bandwidth-bound: the bottleneck is reading the KV cache from HBM. Recall the KV cache formula from earlier in this series:

$$
\text{KV cache} = 2 \times L \times H \times d_k \times n \times B \times \text{sizeof(dtype)}
$$

Halving $\text{sizeof(dtype)}$ from 2 bytes (BF16) to 1 byte (FP8) nearly halves both the memory footprint and the bandwidth cost. This is the most direct lever for improving decode throughput — no architectural changes, no approximations to the attention pattern, just storing and moving fewer bytes per value.

**Tensor Core throughput scaling.** Each GPU generation adds lower-precision datapaths that roughly double peak TFLOPS: FP16 on Ampere, FP8 on Hopper, FP4 on Blackwell. To actually use those faster paths, your operands must be in the corresponding format — you can't feed BF16 tensors to the FP8 Tensor Cores. So quantization isn't just a memory optimization; it's the gateway to the hardware's fastest compute. See [Appendix A]({{< relref "appendix-a-gpu-architecture" >}}) for the full Tensor Core progression.

**The challenge.** Attention has unique quantization sensitivity compared to a typical feed-forward GEMM. Softmax is an exponential — small errors in the $QK^\top$ scores get amplified into large errors in the attention weights. The KV cache persists quantization error across the full sequence length, so a rounding error introduced at position 0 still affects generation at position 10,000. Generic GEMM quantization recipes (calibrate, scale, round) don't account for these attention-specific concerns — which is exactly what this post addresses.

## Number Formats

Quantization's first decision is the **target format**. **Integer** formats (INT8) offer 256 evenly spaced levels — simple, but wasteful for neural network weights that cluster near zero with long tails. **Floating-point** formats (FP8, FP4) use an exponent field for variable step sizes — small steps near zero where most values live, larger steps for outliers — a natural match for transformer value distributions [[6]](#ref-6).

| Format | Bits | Layout | Range | Precision | Primary use |
|--------|------|--------|-------|-----------|-------------|
| FP32 | 32 | 1S + 8E + 23M | $\pm 3.4 \times 10^{38}$ | ~7 decimal digits | Master weights |
| BF16 | 16 | 1S + 8E + 7M | $\pm 3.4 \times 10^{38}$ | ~2 decimal digits | Training/inference baseline |
| FP16 | 16 | 1S + 5E + 10M | $\pm 6.5 \times 10^{4}$ | ~3 decimal digits | Legacy |
| INT8 | 8 | 1S + 7 magnitude | $[-128, 127]$ | 1 unit step | PTQ weight quant |
| FP8 E4M3 | 8 | 1S + 4E + 3M | $\pm 448$ | ~1 decimal digit | Weights + activations |
| FP8 E5M2 | 8 | 1S + 5E + 2M | $\pm 57344$ | coarser | Gradients |
| NVFP4 (MX) | 4 | 1S + 2E + 1M + shared 8-bit scale | per-block | very coarse | Blackwell TC |

<!-- DIAGRAM: bit-layouts.svg — side-by-side bit field diagrams for FP32, BF16, FP8 E4M3, FP8 E5M2, NVFP4. Each shows sign/exponent/mantissa fields with bit counts labeled. -->

**FP8: E4M3 vs E5M2.** The 8-bit budget splits two ways. **E4M3** (range $\pm 448$, finer precision) is preferred for **weights and forward activations**. **E5M2** (range $\pm 57344$, coarser) is preferred for **gradients**, which span many orders of magnitude during training.

**NVFP4 and microscaling (MX).** At 4 bits (E2M1 format), you have only 16 representable values. **Microscaling** makes this viable: a block of $B$ values (typically $B = 16$ for NVFP4) shares a single FP8 scale factor that shifts the block's range. Effective cost: $4 + 8/B = 4.5$ bits per value [[7]](#ref-7). Blackwell Tensor Cores consume NVFP4 natively — dequantization happens inside the Tensor Core, no separate kernel needed.

## Scaling Strategies

The **scale factor** maps actual value ranges into the quantized format's representable range. The quantize-dequantize round-trip: $X_q = \text{clamp}(\lfloor X/s \rceil, q_{\min}, q_{\max})$, then $\hat{X} = s \cdot X_q$. The key choice is **granularity** — how many values share one scale:

| Granularity | Scale count | Use case | Trade-off |
|---|---|---|---|
| **Per-tensor** | 1 for whole tensor | Quick-and-dirty 8-bit | One outlier ruins resolution for all values |
| **Per-channel** | $d_{\text{out}}$ (one per weight row) | Standard weight quant | Each output channel fits its own range |
| **Per-token** | $n$ (one per activation row) | Dynamic activation quant | Handles wildly varying token magnitudes; computed at runtime |
| **Per-group** | $\lceil d/B \rceil$ (blocks of $B$) | MX/NVFP4 ($B=16$) | Finest granularity, highest overhead ($8/B$ extra bits) |

<!-- DIAGRAM: scaling-granularity.svg — a matrix with colored overlays showing: per-tensor (one color for whole matrix), per-channel (one color per column of the weight matrix, i.e., per output channel), per-token (one color per row of the activation matrix, i.e., per token), per-group (small blocks within each row). Each has its scale factor labeled. -->

**Why per-token × per-channel is the standard.** For a GEMM $Y = X W^\top$: per-token scaling on $X$ (rows) and per-channel scaling on $W$ (rows) gives a combined scale $s_i^X \cdot s_j^W$ per output element — a simple outer product applied *after* the quantized GEMM, with no per-element lookups inside the inner loop. This is why FP8 GEMM libraries default to this configuration [[9]](#ref-9).

## Post-Training Quantization (PTQ)

PTQ quantizes a pre-trained model without retraining — you have model weights and a small calibration dataset, no training loop. The methods below progress from naive rounding to increasingly clever strategies for allocating precision.

**Round-to-Nearest (RTN)** is the baseline: $\hat{w} = s \cdot \lfloor w/s \rceil$. Works at 8 bits; breaks at 4 bits because rounding errors compound across the dot product and across 80+ layers.

**GPTQ** [[2]](#ref-2) quantizes weights one at a time and *compensates* remaining weights for each error introduced, using the Hessian of the layer's reconstruction loss to find optimal adjustments. By processing in column order with lazy Hessian updates, it reduces cost to $O(d^2)$ amortized per weight. The result: accurate INT4 quantization in minutes. Widely used for weight-only INT4 models.

**AWQ** [[3]](#ref-3) observes that ~1% of weight channels correspond to large activations and carry disproportionate signal. It applies per-channel scaling to protect these **salient channels** before quantization — simpler than GPTQ (no Hessian), comparable quality at INT4.

**SmoothQuant** [[4]](#ref-4) enables **W8A8** (both weights and activations quantized). The problem: activations develop outlier channels [[1]](#ref-1) with values 10--100$\times$ larger than the rest, making activation quantization terrible. SmoothQuant migrates the difficulty from activations to weights via a diagonal scaling matrix:

$$
Y = (X \cdot \text{diag}(\mathbf{s})^{-1}) \cdot (\text{diag}(\mathbf{s}) \cdot W^\top) = \hat{X} \hat{W}^\top
$$

The smoothed activations $\hat{X}$ have flattened outliers (easy to quantize); the adjusted weights $\hat{W}$ absorb them (but weights tolerate per-channel variation). The smoothing factor $s_j = \max(|X_{:,j}|)^\alpha / \max(|W_{:,j}|)^{1-\alpha}$ balances the difficulty, with $\alpha \approx 0.5$ working well in practice.

## Quantization-Aware Training (QAT)

PTQ treats the model as frozen. QAT lets the model *adapt* to quantization during training, learning weights robust to reduced precision. The cost is a training run; the payoff is near-zero quality loss.

### The straight-through estimator (STE)

Quantization is piecewise-constant — its gradient is zero almost everywhere, so you can't backpropagate through it. The **STE** sidesteps this: apply quantization in the forward pass, but pass gradients through unchanged in the backward pass:

$$
\text{Forward:} \quad \hat{w} = \text{quant}(w) \qquad \text{Backward:} \quad \frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial \hat{w}}
$$

A biased estimator, but it works — over many steps, the optimizer learns to place weights near quantization grid points where rounding error is minimal.

<!-- DIAGRAM: ste.svg — two side-by-side plots. Left: "Forward" showing staircase quantization function. Right: "Backward" showing identity function (straight line) with annotation "gradient passes through unchanged". -->

### FP8 mixed-precision training

Traditional mixed precision [[5]](#ref-5) keeps master weights in FP32, runs GEMMs in FP16/BF16. **FP8 mixed precision** (DeepSeek-V3 [[8]](#ref-8)) pushes further: forward GEMMs use **E4M3** (precision), backward GEMMs use **E5M2** (range), both accumulate in FP32, master weights stay in FP32. Per-tile scaling ($128 \times 128$ blocks) keeps error local. Result: ~$2\times$ throughput over BF16 with negligible quality loss — FP8 is the new baseline for large-scale training.

## Hardware Support

The number format you quantize to must match what the hardware's fast path can consume. Each NVIDIA GPU generation added Tensor Core support for a lower-precision datapath — and the throughput gains are substantial. See [Appendix A]({{< relref "appendix-a-gpu-architecture" >}}) for the full GPU architecture story; here we focus on what matters for quantized computation.

| Generation | GPU | Quantized formats | Approx. peak TFLOPS |
|---|---|---|---|
| Ampere (SM 80) | A100 | FP16, BF16, TF32, INT8 | ~312 (FP16) |
| Hopper (SM 90) | H100 | + FP8 (E4M3, E5M2) | ~990 (FP16), ~1979 (FP8) |
| Blackwell (SM 100) | B200 | + FP4 (NVFP4/MX) | ~2250 (FP16), ~4500 (FP8), ~9000 (FP4) |

*Note: TFLOPS are approximate peak dense. Real workloads achieve a fraction depending on memory-boundedness and kernel efficiency.*

### Tensor Core GEMM dataflow

Key insight: **quantized operands ≠ quantized arithmetic.**

The Tensor Core pipeline processes a GEMM in four stages, and understanding the precision at each stage is critical:

1. **Load**: operands read from memory in reduced precision (FP8, FP4) — this is where bandwidth savings come from.
2. **Multiply**: element-wise products computed at operand precision (or a slightly wider intermediate format).
3. **Accumulate**: partial products summed in an **FP32 accumulator** — this prevents rounding error from compounding across the dot product.
4. **Store**: the FP32 result is either kept at full precision or re-quantized back to a reduced format for the next operation.

<!-- DIAGRAM: gemm-dataflow.svg — pipeline diagram. Two input arrows labeled "FP8 A" and "FP8 B" feed into a "Multiply (FP8)" box, which feeds into an "Accumulate (FP32)" box with feedback loop, which outputs to "FP32 result" with optional branch to "re-quantize → FP8". The FP32 accumulator box is highlighted. -->

Practical implications:

- **FP32 accumulator is why FP8 GEMMs match BF16 quality** — individual multiply errors don't snowball. Each FP8 product has small rounding error, but the FP32 accumulator sums thousands of these without additional precision loss.
- **FP4 still accumulates in FP32**, but coarser operands mean more per-element error — this is why FP4 needs finer-grained scaling (per-group/MX) to maintain acceptable output quality.
- **The re-quantize decision between GEMMs**: keep FP32 (accurate, but $4\times$ the memory) or re-quantize to FP8 (compact, but adds error at every stage boundary). In attention kernels, this decision happens per-tile in SRAM — this is the bridge to the kernel fusion section below, where we'll see how FlashAttention folds the re-quantize step into the attention loop to avoid extra memory round-trips.

For details on the Tensor Core microarchitecture — including WGMMA (Hopper's warp-group level MMA), UMMA (Blackwell's single-thread launch), and TMEM (Blackwell's accumulator memory) — see [Appendix A]({{< relref "appendix-a-gpu-architecture" >}}).

## Where Quantization Hits Attention

The foundations above apply to any GEMM — the formats, scaling strategies, and PTQ/QAT techniques work the same whether you're quantizing an FFN projection or a convolution. But attention has **four distinct quantization points**, each with different operand characteristics and error sensitivity. Understanding these differences is what separates "quantize everything to FP8" from a quantization strategy that actually works.

### 1. QKV projection GEMMs

The first quantization point is the most straightforward: the linear projections that produce Q, K, and V from the input.

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

These are standard weight $\times$ activation GEMMs. Weight quantization uses **per-channel, static** scaling — the scale factors are computed once from calibration data and baked in. Activation quantization uses **per-token, dynamic** scaling — each token's scale is computed on the fly based on its actual values.

There is nothing attention-specific here. The QKV projections have the same weight and activation distributions as FFN GEMMs, the same outlier patterns, and the same per-token $\times$ per-channel scaling recipe described in the Scaling Strategies section above. All the PTQ and QAT techniques (GPTQ, AWQ, SmoothQuant) apply identically.

### 2. $QK^\top$ score computation

The second point is where things get interesting. The attention score computation $S = QK^\top / \sqrt{d_k}$ is a GEMM between two **activations** — there are no static weights. Both Q and K are dynamic tensors whose distributions change with every input. Both require **per-token, dynamic scaling**: you cannot precompute scale factors offline.

The critical issue is what happens *after* this GEMM. The raw scores pass through softmax:

$$
\alpha_{ij} = \frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}
$$

where $s_{ij} = q_i \cdot k_j / \sqrt{d_k}$. Softmax is an exponential — and exponentials amplify errors. If the true score is $s$ and quantization introduces an error $\epsilon$ so the computed score is $s + \epsilon$, the effect on the attention weight is:

$$
\frac{\text{softmax}(s + \epsilon)}{\text{softmax}(s)} \approx e^{\epsilon} \cdot \frac{\sum_k e^{s_k}}{\sum_k e^{s_k + \epsilon_k}}
$$

The error doesn't simply add — it **redistributes attention mass**. A quantization error of $\epsilon = 0.5$ in a single score creates a multiplicative factor of $e^{0.5} \approx 1.65$ — that token receives 65% more attention weight than it should, with the excess stolen from every other token. At lower precision (FP4), where quantization errors can easily reach this magnitude, the attention pattern can be substantially distorted.

This makes $QK^\top$ the **most error-sensitive** of the four quantization points. It's the primary reason that naive FP8 quantization of attention can degrade quality even when the same format works fine for FFN layers.

### 3. Score $\times$ V value aggregation

The third point is the weighted sum of value vectors: $O = \alpha V$, where $\alpha$ is the post-softmax attention weight matrix and $V$ is the value matrix.

This GEMM has a favorable asymmetry. The attention weights $\alpha$ are **well-bounded**: every element lies in $[0, 1]$ and each row sums to 1. There are no outliers, the range is known a priori, and the distribution is friendly to quantization. You could use a fixed scale factor and waste very little resolution.

The V matrix, on the other hand, has the same dynamic range challenges as K — it's an activation tensor with potential outliers, requiring per-token dynamic scaling.

The overall sensitivity is **lower than $QK^\top$** for a simple reason: the bounded attention weights act as a dampener. Even if V has quantization errors, the weighted sum averages across many value vectors, and the softmax-clamped weights prevent any single error from dominating. There is no exponential amplification step downstream.

### 4. KV cache storage

The fourth point is unique to inference: K and V tensors are quantized when written to the KV cache and dequantized every time a decode step reads them. For long-context inference, the KV cache dominates memory — compressing it from BF16 to FP8 or FP4 is one of the highest-impact quantization decisions.

The error characteristics differ from the GEMM quantization points above:

- **Single quantize-dequantize cycle**: each cached token undergoes one round of quantization error at write time. That error persists for the token's entire lifetime in the cache, but it doesn't accumulate — there's no repeated requantization.
- **Per-channel or per-head scaling** is typical. Since each head has a consistent value distribution across positions, a single scale per head works well.
- **Aggregate effect at long sequences**: across $s$ cached tokens, the per-token quantization errors form a distribution. No individual error is catastrophic, but the aggregate effect matters — when the model attends over thousands of tokens, the errors across all cached K vectors combine in the $QK^\top$ dot products. At very long sequences, this aggregate noise can shift the attention distribution enough to matter.

Methods like KIVI [[11]](#ref-11) and KVQuant [[12]](#ref-12) explore aggressive KV cache quantization (down to 2-bit keys with higher-precision values). For a deeper treatment of KV cache compression strategies — including mixed-precision caching, adaptive bit-width, and token eviction — see [post 18]({{< relref "18-kv-cache-compression" >}}).

### Sensitivity summary

| Point | Operands | Scaling | Sensitivity | Why |
|---|---|---|---|---|
| QKV projections | weight $\times$ activation | per-channel $\times$ per-token | Low | Standard GEMM, no amplification |
| $QK^\top$ | activation $\times$ activation | per-token $\times$ per-token | **High** | Softmax amplifies errors exponentially |
| Score $\times$ V | softmax output $\times$ activation | (bounded) $\times$ per-token | Medium | Bounded weights dampen errors |
| KV cache | stored activations | per-head / per-channel | Medium | Error persists across full sequence |

<!-- DIAGRAM: attention-quant-points.svg — an attention layer diagram (Q,K,V projections → QK^T → softmax → score*V → output projection) with the four quantization points highlighted in different colors, each labeled with sensitivity level. -->

## Kernel Fusion

Without fusion, each quantize/dequantize step is a separate kernel launch with an HBM round-trip — the memory traffic can erase the throughput gains. **Kernel fusion** folds quantization into the attention kernel itself, keeping intermediates in SRAM. A naively quantized attention layer needs 6 kernel launches; a fused version needs 2. The intermediate score matrix never materializes in HBM.

<!-- DIAGRAM: fused-vs-unfused.svg — two-column comparison. Left: "Unfused" showing 6 kernel boxes with HBM arrows between each. Right: "Fused" showing 2 kernel boxes (QKV projection, fused attention) with only input/output HBM arrows. Eliminated HBM traffic is crossed out or grayed. -->

**FlashAttention v3 (Hopper, FP8)** [[9]](#ref-9) targets WGMMA instructions: both GEMMs use FP8 E4M3 operands with FP32 accumulation, online softmax stays in FP32, and quant/dequant happens in registers with zero HBM traffic. Per-token scaling is folded into tile loading. See [FlashAttention v3/v4]({{< relref "10-flash-attention-v3-v4" >}}).

**FlashAttention v4 (Blackwell, FP4)** [[10]](#ref-10) targets UMMA instructions with NVFP4 operands and MX block scaling. The kernel manages per-block scale bookkeeping within each tile. FP32 accumulation and online softmax unchanged from v3 — precision-sensitive parts stay full-precision regardless of operand format. See [FlashAttention v3/v4]({{< relref "10-flash-attention-v3-v4" >}}).

**KV cache dequant fusion.** In serving frameworks (vLLM, TensorRT-LLM), the KV cache is often stored in FP8. Without fusion, a separate dequant kernel reads FP8 from paged blocks and writes to a staging buffer — two full HBM traversals. With fusion, the attention kernel reads FP8 directly and dequantizes in-register, eliminating one HBM read-write cycle per attention layer per decode step.

## Quantization in MLA Paths

[MLA]({{< relref "04-mla" >}}) introduced two distinct attention code paths: prefill (decompress and run standard MHA) and decode (absorb decompression into query, attend in latent space). Quantization interacts differently with each.

### Prefill: standard FP8 attention

After decompression, K and V are standard-shaped: $(n, H, d_k)$. Standard FP8 FlashAttention applies directly — same as any MHA model. The latent $\mathbf{c}$ and decompression happen in BF16 (the projection GEMMs can use FP8). This path is straightforward — no MLA-specific concerns.

### Decode: quantized latent cache

The absorbed decode path operates in **latent space**: attention over cached $\mathbf{c}$ vectors of dimension $d_c$. Storing $\mathbf{c}$ in FP8 costs $576 \times 1 = 576$ bytes per token (vs 1152 bytes in BF16) — a further 2$\times$ reduction on top of MLA's architectural compression.

**Stacked compression.** MLA gives ~57$\times$ over MHA, FP8 gives another 2$\times$, for a total ~114$\times$ KV cache reduction.

**Calibration caveat.** The absorbed query $\mathbf{q}' = \mathbf{q} \cdot {W^{\text{up}}_K}^\top$ has a different distribution than a standard query — it includes the decompression matrix. Scale factors calibrated on standard attention may not transfer; MLA decode may need its own calibration pass.

One quantization-friendly property of this path: the $\mathbf{c} \cdot {\mathbf{q}'}^\top$ dot product is in a higher-dimensional space ($d_c = 512$ vs $d_k = 128$), which helps quantization. More elements in the dot product means the FP32 accumulator averages out per-element errors better.

### Sparse attention indexer (FP8 keys)

DeepSeek-V3.2's NSA/DSA indexer ([post 07]({{< relref "07-sparse-attention" >}})) caches indexer keys in FP8. The indexer makes a coarse decision — "which tokens are worth attending to?" — not exact scores. FP8 is sufficient because the indexer only needs to rank tokens roughly, not compute precise attention weights. Memory saving: the indexer cache costs ~132 bytes per token in FP8 vs 264 in BF16.

## Quantization in FFN / MoE

Attention gets the nuanced quantization story, but the FFN layers are where quantization delivers the most straightforward wins — large weight matrices, standard GEMM patterns, no softmax to worry about.

### FFN weight quantization

Typical SwiGLU FFN has three weight matrices per layer: $W_{\text{gate}}$, $W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$. For $d = 7168$, $d_{\text{ff}} = 18432$ (DeepSeek-V3 dense dimensions): each FFN layer stores $3 \times 7168 \times 18432 = 396\text{M}$ parameters.

The memory arithmetic is simple:

- In **BF16**: ~792 MB per layer
- In **FP8**: ~396 MB per layer
- In **FP4**: ~198 MB per layer

No softmax sensitivity, no cache accumulation — **activation quantization is simpler than in attention**. The FFN path is a sequence of GEMMs with element-wise nonlinearities in between, which is exactly what quantized matmul kernels are optimized for.

### MoE: the quantization multiplier

MoE models ([post 05]({{< relref "05-moe" >}})) have $N$ expert FFNs (e.g., 256 in DeepSeek-V3), but only $k$ are active per token. Total expert parameters: $N \times 3 \times d \times d_{\text{ff}}^{\text{expert}}$. For DeepSeek-V3: $256 \times 3 \times 7168 \times 2048 \approx 11.3$ GB per MoE layer in BF16.

- **Memory savings scale with $N$**: quantizing BF16 to FP8 saves ~5.6 GB *per MoE layer*
- **Compute savings scale with $k$**: only $k$ experts run per token, so throughput gain applies to $k$ GEMMs, not $N$. But at memory level, all $N$ experts must be resident.

This asymmetry makes quantization especially valuable for MoE — the memory cost is dominated by the full expert count $N$, while compute cost depends only on the active count $k$. Quantization attacks the bigger problem.

### DeepSeek-V3 FP8 MoE training

DeepSeek-V3 [[8]](#ref-8) trained all 256 experts in FP8 from the start (quantization-aware training). Per-tile scaling (128$\times$128 tiles) for expert GEMMs keeps outlier impact local — each tile gets its own scale factor, preventing a single outlier from crushing the dynamic range of a large block.

**Gate precision**: the routing gate stays in BF16/FP32. Small errors in gate logits can flip top-$k$ selection, sending a token to entirely different experts — a catastrophic error that doesn't heal. The gate GEMM is small ($d \times N$), so high precision costs negligible throughput.

## Trade-offs

| Method | Bits | Quality impact | Throughput gain | Complexity | When to use |
|---|---|---|---|---|---|
| BF16 (baseline) | 16 | None | 1× | None | Training, quality-critical serving |
| INT8 PTQ (GPTQ/AWQ) | 8 | Minimal | ~1.5-2× | Offline calibration | Weight-dominated serving |
| SmoothQuant (W8A8) | 8 | Minimal | ~1.5-2× | Calibration + smoothing | Activation-bottlenecked serving |
| FP8 PTQ | 8 | Minimal | ~2× | Scale calibration | Hopper+ serving |
| FP8 QAT | 8 | Near-zero | ~2× | Training recipe | Frontier model training |
| FP4 / NVFP4 | 4 | Small-moderate | ~4× | Block scaling (MX) | Blackwell, throughput-critical |
| INT4 PTQ (GPTQ/AWQ) | 4 | Moderate | ~2-3× | Careful calibration | Memory-constrained deployment |

**Precision vs throughput.** The jump from 16-bit to 8-bit is nearly free in terms of quality — modern calibration techniques keep perplexity degradation within noise for most models. The jump from 8-bit to 4-bit is a different story: you're cutting representable values from 256 to 16, and even with microscaling the precision loss is measurable. The gap between 8-bit and 4-bit quality impact is much larger than the gap between 16-bit and 8-bit, which means the decision to go sub-8-bit should always be driven by a concrete throughput or memory constraint, not just "lower is better."

**Static vs dynamic quantization.** Static scales (computed once during calibration, fixed at inference) are fast — no per-token overhead. But they miss outliers that weren't present in the calibration set, and they can't adapt to distribution shifts across different inputs. Dynamic quantization (computing scales per-token or per-block at runtime) adds overhead but handles the full range of real inputs. FP8 per-token dynamic quantization hits the sweet spot: the scale computation is a single reduction per token, which is negligible relative to the GEMM it feeds. This is why FP8 dynamic quantization has become the default for Hopper deployments.

**Where error compounds.** Not all quantization targets are equally sensitive. Weights are the most forgiving — they're the same values every forward pass, so calibration has full opportunity to minimize error. Activations are harder because they vary per input, making static calibration a compromise. The KV cache is hardest of all: quantized key and value vectors persist for the full sequence length, and attention aggregates across thousands of cached positions. A rounding error introduced at position 0 still affects generation at position 10,000. In practice, this means you can afford coarser quantization for weights (INT4 is viable) than for the KV cache (FP8 or careful INT4 with per-channel scaling).

**Fusion or it doesn't count.** Throughput gains from quantization assume fused kernels — quantize/dequantize steps folded into the GEMM or attention kernel so there are no extra memory round-trips. Unfused FP8 attention that launches separate dequant and quant kernels can actually be *slower* than BF16 FlashAttention, because the extra kernel launches and memory traffic outweigh the faster arithmetic. This is why kernel maturity is the practical bottleneck for quantization adoption: the format is useless until someone writes a fused kernel that exploits it. FlashAttention v3's FP8 path and TensorRT-LLM's fused FP8 attention were the inflection points that made FP8 attention practical, not the Hopper hardware itself.

## Adoption

- **Llama 3 / 3.1** (Meta, 2024): official INT8 and FP8 quantized checkpoints. Trained in BF16, PTQ for inference.
- **DeepSeek-V3** (DeepSeek, 2024): first major model trained end-to-end in FP8 (QAT). Per-tile scaling across all 256 MoE experts.
- **Mistral / Mixtral** (Mistral AI, 2023-2024): community GPTQ and AWQ INT4 quantizations widely used for local serving on consumer GPUs.
- **vLLM / TensorRT-LLM** (2024-2025): FP8 KV cache support, fused FP8 attention kernels for Hopper+. FP4 support in development.
- **Blackwell (B200)** (NVIDIA, 2025): FP4 Tensor Cores shipping. FlashAttention v4 first attention kernel targeting native FP4. Adoption still early as of early 2026.

## References

<span id="ref-1">[1]</span> Dettmers, T., et al. (2022). [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339). *NeurIPS 2022*.

<span id="ref-2">[2]</span> Frantar, E., et al. (2022). [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323). *ICLR 2023*.

<span id="ref-3">[3]</span> Lin, J., et al. (2023). [AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978). *MLSys 2024*.

<span id="ref-4">[4]</span> Xiao, G., et al. (2023). [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438). *ICML 2023*.

<span id="ref-5">[5]</span> Micikevicius, P., et al. (2018). [Mixed Precision Training](https://arxiv.org/abs/1710.03740). *ICLR 2018*.

<span id="ref-6">[6]</span> NVIDIA. (2022). [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433). *arXiv preprint*.

<span id="ref-7">[7]</span> OCP. (2023). [Open Compute Project Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). *OCP*.

<span id="ref-8">[8]</span> DeepSeek-AI. (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). *arXiv preprint*.

<span id="ref-9">[9]</span> Shah, J., et al. (2024). [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08691). *arXiv preprint*.

<span id="ref-10">[10]</span> Shah, J., et al. (2025). FlashAttention-4: Hardware-Friendly Attention on Blackwell.

<span id="ref-11">[11]</span> Liu, Z., et al. (2024). [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750). *arXiv preprint*.

<span id="ref-12">[12]</span> Hooper, C., et al. (2024). [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079). *arXiv preprint*.

*Last updated: April 2026*
