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

Quantization's first decision is the **target format**. There are two families: **integer** formats (fixed step size between representable values, simple hardware) and **floating-point** formats (variable step size via an exponent field, better for values spanning a wide dynamic range). Modern LLM quantization has shifted decisively from integer to floating-point — the value distributions inside transformers are too peaky and heavy-tailed for uniform grids to handle well.

| Format | Bits | Layout | Range | Precision | Primary use |
|--------|------|--------|-------|-----------|-------------|
| FP32 | 32 | 1S + 8E + 23M | $\pm 3.4 \times 10^{38}$ | ~7 decimal digits | Master weights |
| BF16 | 16 | 1S + 8E + 7M | $\pm 3.4 \times 10^{38}$ | ~2 decimal digits | Training/inference baseline |
| FP16 | 16 | 1S + 5E + 10M | $\pm 6.5 \times 10^{4}$ | ~3 decimal digits | Legacy |
| INT8 | 8 | 1S + 7 magnitude | $[-128, 127]$ | 1 unit step | PTQ weight quant |
| FP8 E4M3 | 8 | 1S + 4E + 3M | $\pm 448$ | ~1 decimal digit | Weights + activations |
| FP8 E5M2 | 8 | 1S + 5E + 2M | $\pm 57344$ | coarser | Gradients |
| NVFP4 (MX) | 4 | 1E + 2M + shared 8-bit scale | per-block | very coarse | Blackwell TC |

### Integer vs floating-point

**INT8** gives you 256 evenly spaced levels across whatever range you map. If your values are roughly uniform — spread evenly between min and max — this works fine. But neural network weights and activations are emphatically *not* uniform. Weights typically follow a Gaussian-like distribution: most values cluster near zero with a long tail of outliers. With a uniform grid, most of your 256 levels end up in the tails where almost no values live, while the crowded region near zero gets only coarse coverage.

**FP8** solves this with an exponent field that gives *variable* step sizes — small steps near zero (where most weights live) and larger steps for outliers. This is a natural match for neural network value distributions. The step size doubles each time the exponent increments by 1, so the density of representable values is highest near zero and drops off logarithmically. You don't need to "waste" levels on empty regions of the number line.

### FP8: E4M3 vs E5M2

The 8-bit floating-point budget can be split two ways, and the choice matters:

- **E4M3** — 4 exponent bits, 3 mantissa bits. Range $\pm 448$, with 15 distinct exponent values. The extra mantissa bit buys finer precision near zero, making E4M3 the preferred format for **weights and forward activations** where precision matters more than range.
- **E5M2** — 5 exponent bits, 2 mantissa bits. Range $\pm 57344$, with 31 exponent values. The extra exponent bit doubles the dynamic range at the cost of precision — preferred for **gradients** during training, which can span many orders of magnitude.

<!-- DIAGRAM: bit-layouts.svg — side-by-side bit field diagrams for FP32, BF16, FP8 E4M3, FP8 E5M2, NVFP4. Each shows sign/exponent/mantissa fields with bit counts labeled. -->

**Concrete example: representing 0.1875.** Let's trace the value $0.1875 = 3/16$ through three formats.

First, express it in binary: $0.1875 = 0.0011_2 = 1.1_2 \times 2^{-3}$.

**FP32** (1S + 8E + 23M, bias = 127):

$$
\text{sign} = 0, \quad \text{exponent} = -3 + 127 = 124 = 01111100_2, \quad \text{mantissa} = 10000000000000000000000_2
$$

The stored 32 bits: `0 01111100 10000000000000000000000`. Exact representation — no rounding.

**BF16** (1S + 8E + 7M, bias = 127):

$$
\text{sign} = 0, \quad \text{exponent} = 124 = 01111100_2, \quad \text{mantissa} = 1000000_2
$$

The stored 16 bits: `0 01111100 1000000`. Still exact — the mantissa `1.1` only needs 1 bit beyond the implicit leading 1, and BF16 has 7 mantissa bits to spare.

**FP8 E4M3** (1S + 4E + 3M, bias = 7):

$$
\text{sign} = 0, \quad \text{exponent} = -3 + 7 = 4 = 0100_2, \quad \text{mantissa} = 100_2
$$

The stored 8 bits: `0 0100 100`. Still exact in this case — $0.1875$ is friendly to binary. But you can see how quickly you'd lose precision: with only 3 mantissa bits, any value requiring more than 3 fractional binary digits will be rounded.

### NVFP4 and microscaling (MX)

At 4 bits you have only 16 representable values — far too few for any useful dynamic range on their own. The trick is **microscaling**: a block of $B$ values (typically $B = 32$) shares a single FP8 scale factor. Each individual value uses 4 bits (1 exponent + 2 mantissa), and the shared scale "shifts" the block's representable range to wherever it needs to be.

The effective storage cost per value:

$$
\text{effective bits} = 4 + \frac{8}{B}
$$

For $B = 32$, that's $4.25$ bits per value — close to the 4-bit ideal with only a small overhead for the shared scale.

This design is formalized in the **OCP Microscaling (MX) specification** [[7]](#ref-7). NVIDIA's Blackwell Tensor Cores consume NVFP4 natively, meaning the dequantization from 4-bit block format to wider types happens inside the Tensor Core itself — no separate dequantize kernel needed. This is what enables FlashAttention v4's FP4 path to actually hit the hardware's peak throughput.

## Scaling Strategies

Even with the right number format, you need to map the actual value range into the representable range. The **scale factor** does this mapping. The key design choice: how many values share a single scale factor?

The quantize-dequantize round-trip looks like this:

$$
X_q = \text{clamp}\!\left(\left\lfloor \frac{X}{s} \right\rceil,\; q_{\min},\; q_{\max}\right)
$$

$$
\hat{X} = s \cdot X_q
$$

where $s$ is the scale factor, $\lfloor \cdot \rceil$ is round-to-nearest, and $[q_{\min}, q_{\max}]$ is the format's representable range. The **quantization error** is $X - \hat{X}$.

The scale factor controls the trade-off between *clipping* (values outside the representable range get clamped, introducing large errors) and *rounding* (values inside the range get rounded to the nearest representable value, introducing small errors). A well-chosen scale minimizes the sum of both. The granularity at which you assign scale factors — per-tensor, per-channel, per-token, or per-group — determines how tightly each region of the tensor can be fit.

### Per-tensor scaling

One scale for the entire tensor:

$$
s = \frac{\max(|X|)}{q_{\max}}
$$

This is the cheapest option: one scale to store and one scale to apply. But a single outlier stretches the range, wasting resolution for the majority of values. If one weight in a matrix is $10\times$ larger than the rest, the step size between representable values is set by that outlier, and the 99.9% of weights clustered near zero get only coarse coverage.

Per-tensor scaling is acceptable for 8-bit weights (which tend to have narrow, well-behaved distributions), but poor for activations (which are outlier-prone, especially in large models).

### Per-channel scaling

One scale per row or column of a weight matrix. For $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, you compute one scale per output channel (row), giving $d_{\text{out}}$ scales total:

$$
s_j = \frac{\max(|W_{j,:}|)}{q_{\max}}, \quad j = 1, \ldots, d_{\text{out}}
$$

This is the standard approach for weight quantization. Each output channel can have a different range — if channel 42 has weights spanning $[-0.5, 0.5]$ while channel 17 spans $[-2.0, 2.0]$, each gets its own scale that covers its range tightly. The overhead is small: $d_{\text{out}}$ extra FP16/FP32 values per weight matrix, which is negligible compared to $d_{\text{out}} \times d_{\text{in}}$ quantized weights.

### Per-token scaling

One scale per row of the activation matrix $X \in \mathbb{R}^{n \times d}$: one scale per token.

$$
s_i = \frac{\max(|X_{i,:}|)}{q_{\max}}, \quad i = 1, \ldots, n
$$

Activations vary wildly across tokens — a token for "the" vs one for a rare technical term may have $10\times$ different magnitude. Per-token scaling lets each token use its own range. The catch: unlike weight scales (which can be precomputed offline), per-token scales must be computed **dynamically at runtime**, since activations depend on the input. This adds a reduction operation (find the max of each row) before every quantized GEMM, but on modern hardware the cost is small relative to the GEMM itself.

### Per-group / block scaling (MX)

One scale per contiguous block of $B$ values. For a vector of length $d$, that's $\lceil d / B \rceil$ scales:

$$
s_k = \frac{\max(|X_{kB:(k+1)B}|)}{q_{\max}}, \quad k = 0, 1, \ldots, \lceil d/B \rceil - 1
$$

This is the finest practical granularity — it approaches per-element accuracy at the cost of $\text{sizeof(scale)} / B$ overhead per value. This is exactly the MX/NVFP4 approach described in the Number Formats section above: $B = 32$, each shared scale is FP8 (8 bits), giving $8/32 = 0.25$ bits overhead per value. The total effective cost is $4.25$ bits per value rather than the raw 4 bits.

Per-group scaling is particularly valuable for FP4 quantization, where 4 bits alone provide only 16 representable values — the shared scale "shifts" each block to the right region of the number line, dramatically improving effective resolution.

<!-- DIAGRAM: scaling-granularity.svg — a matrix with colored overlays showing: per-tensor (one color for whole matrix), per-channel (one color per row), per-token (one color per column), per-group (small blocks within the matrix). Each has its scale factor labeled. -->

### Why per-token × per-channel is the standard

For a GEMM $Y = X W^\top$ where $X \in \mathbb{R}^{n \times d_{\text{in}}}$ and $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:

- Quantize $X$ with **per-token** scales: one scale $s_i^X$ per row of $X$ (per token).
- Quantize $W$ with **per-channel** scales: one scale $s_j^W$ per row of $W$ (per output channel).

Each output element $Y_{i,j}$ is the dot product of row $i$ of $X$ with row $j$ of $W$. The combined scale is simply:

$$
Y_{i,j} = s_i^X \cdot s_j^W \cdot (X_q \cdot W_q^\top)_{i,j}
$$

The scale matrix $s_i^X \cdot s_j^W$ is an outer product of two vectors — $n$ token scales and $d_{\text{out}}$ channel scales — which can be applied as a cheap post-processing step after the integer/FP8 GEMM finishes. No per-element scale lookups inside the inner loop, no complex indexing. This is why FP8 GEMM libraries (cuBLAS, CUTLASS) default to the per-token $\times$ per-channel configuration, and why the FP8 attention kernels in FlashAttention v3 [[4]](#ref-4) adopt the same pattern for the $QK^\top$ and $\text{score} \times V$ GEMMs.

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
