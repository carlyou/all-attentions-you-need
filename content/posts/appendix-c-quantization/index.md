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

The scale matrix $s_i^X \cdot s_j^W$ is an outer product of two vectors — $n$ token scales and $d_{\text{out}}$ channel scales — which can be applied as a cheap post-processing step after the integer/FP8 GEMM finishes. No per-element scale lookups inside the inner loop, no complex indexing. This is why FP8 GEMM libraries (cuBLAS, CUTLASS) default to the per-token $\times$ per-channel configuration, and why the FP8 attention kernels in FlashAttention v3 [[9]](#ref-9) adopt the same pattern for the $QK^\top$ and $\text{score} \times V$ GEMMs.

## Post-Training Quantization (PTQ)

PTQ quantizes a pre-trained model without retraining. You have the model weights and a small calibration dataset — no training loop, no gradient updates. The challenge: find a quantized representation that preserves the model's output quality despite reducing precision. The methods below represent a progression from naive rounding to increasingly clever strategies for deciding *where* to spend your limited precision budget.

### Round-to-Nearest (RTN)

The simplest approach: quantize each weight independently to the nearest representable value. Given a scale factor $s$ (determined by whichever scaling strategy you choose from the section above), each weight is mapped as:

$$
w_q = s \cdot \left\lfloor \frac{w}{s} \right\rceil
$$

where $\lfloor \cdot \rceil$ denotes rounding to the nearest integer. No calibration data needed, no inter-weight dependencies considered — just round and move on.

RTN works surprisingly well at 8 bits, where 256 levels provide enough resolution that individual rounding errors stay small and don't accumulate destructively across a matrix multiply. But at 4 bits you have only 16 representable levels. Each weight gets rounded by up to half a step size, and these errors aren't independent — they accumulate across the dot product. For a $d$-dimensional dot product, the expected squared error grows proportionally with $d$. In a model with 80+ layers, each feeding into the next, the compounding effect makes naive 4-bit RTN unusable for most architectures. You need smarter methods.

### GPTQ

**GPTQ** [[2]](#ref-2) takes a fundamentally different approach: quantize weights one at a time, and *compensate* the remaining unquantized weights for the error you just introduced. If rounding weight $w_{13}$ down introduced an error in the layer's output, you can slightly adjust weights $w_{14}, w_{15}, \ldots$ to cancel that error — before you quantize them too.

This builds on **Optimal Brain Quantization (OBQ)**, which frames quantization as minimizing the layer's reconstruction loss $\| WX - \hat{W}X \|^2$. The optimal compensation for remaining weights $\mathbf{w}_F$ when quantizing weight $w_q$ is:

$$
\delta_F = -\frac{w_q - \text{quant}(w_q)}{[\mathbf{H}_F^{-1}]_{qq}} \cdot (\mathbf{H}_F^{-1})_{:,q}
$$

where $\mathbf{H}_F = 2 X X^\top$ is the Hessian of the reconstruction loss (computed from calibration data), and $[\mathbf{H}_F^{-1}]_{qq}$ is the diagonal element corresponding to the weight being quantized. Intuitively, the Hessian tells you how sensitive the output is to each weight — weights with high curvature need careful compensation, while low-curvature weights can absorb more adjustment.

OBQ's problem is cost: it requires updating the Hessian inverse after every single weight, giving $O(d^3)$ complexity per weight. GPTQ's key innovation is processing weights in **column order** with a lazy batch update to the Hessian. Instead of recomputing the inverse after each weight, GPTQ quantizes an entire column, accumulates the compensation, and updates the Hessian once per column. This reduces the amortized cost to $O(d^2)$ per weight, making it practical for billion-parameter models. The result: accurate 4-bit weight quantization in minutes rather than hours. GPTQ is now one of the most widely used methods for producing INT4 weight-only models.

### AWQ (Activation-Aware Weight Quantization)

**AWQ** [[3]](#ref-3) starts from an empirical observation: not all weight channels are equally important. Roughly 1% of weight channels correspond to consistently large activations across calibration inputs — these are **salient channels** that carry a disproportionate share of the signal. Quantizing them carelessly causes outsized damage.

Rather than the Hessian-based compensation of GPTQ, AWQ applies a simple per-channel scaling before quantization:

$$
\hat{W} = \text{quant}(W \cdot \text{diag}(\mathbf{s}))
$$

where the scale vector $\mathbf{s}$ is chosen to *protect* salient channels. The idea: multiplying a salient channel's weights by a scale $s_j > 1$ before quantization effectively gives that channel finer quantization resolution (since the step size relative to the scaled values is smaller). The corresponding activation channel is divided by $s_j$ at runtime to preserve the mathematical result.

The scale factors are determined by activation magnitudes from the calibration set — channels with consistently large activations get higher scales. This is conceptually much simpler than GPTQ: no Hessian computation, no sequential weight updates, no column-order processing. Calibration is faster and the implementation is straightforward. Despite this simplicity, AWQ achieves comparable quality to GPTQ at INT4, making it a popular choice when calibration speed matters.

### SmoothQuant

The methods above focus on **weight-only quantization**: weights are INT4/INT8 while activations remain in FP16 (a W4A16 or W8A16 scheme). This helps with memory bandwidth — fewer bytes to load — but the actual matrix multiply still uses FP16 arithmetic for the activations. To get the full speedup from INT8 Tensor Cores, you want **W8A8**: both weights *and* activations quantized to 8 bits.

The problem is that activations are much harder to quantize than weights. Weights have well-behaved, roughly Gaussian distributions. Activations, especially in large language models, develop **outlier channels**: specific feature dimensions where values are 10--100$\times$ larger than the rest. These outliers appear at fixed channel positions across tokens and layers. A per-tensor or even per-token scale is dominated by these outliers, crushing the resolution for the 99% of channels that have normal magnitudes.

**SmoothQuant** [[4]](#ref-4) solves this with a mathematical trick: migrate the quantization difficulty from activations to weights. For a linear layer $Y = X W^\top$, insert a diagonal scaling matrix:

$$
Y = (X \cdot \text{diag}(\mathbf{s})^{-1}) \cdot (\text{diag}(\mathbf{s}) \cdot W^\top) = \hat{X} \hat{W}^\top
$$

The smoothed activation $\hat{X} = X \cdot \text{diag}(\mathbf{s})^{-1}$ has its outlier channels divided down — easier to quantize. The adjusted weight $\hat{W}^\top = \text{diag}(\mathbf{s}) \cdot W^\top$ absorbs the outlier magnitudes — but weights are inherently more tolerant of per-channel variation because we can use per-channel scaling.

The smoothing factor balances the difficulty between activations and weights:

$$
s_j = \frac{\max(|X_{:,j}|)^\alpha}{\max(|W_{:,j}|)^{1-\alpha}}, \quad \alpha \in [0, 1]
$$

When $\alpha = 1$, all difficulty stays on the activations (no smoothing). When $\alpha = 0$, all difficulty moves to the weights. In practice, $\alpha \approx 0.5$ works well for most models. The smoothing factors are computed offline from calibration data and folded into the weights, so there is zero runtime overhead — just a modified weight matrix that makes W8A8 quantization viable.

## Quantization-Aware Training (QAT)

PTQ treats the model as frozen — it can only rearrange the quantization to minimize damage. QAT goes further: it lets the model *adapt* to quantization during training, learning weight values that are robust to reduced precision. The cost is a training run; the payoff is near-zero quality loss even at aggressive bit-widths.

### The straight-through estimator (STE)

The fundamental problem: quantization (rounding + clamping) is piecewise-constant, so its gradient is zero almost everywhere. Between any two adjacent quantization levels the output is flat — the derivative is zero. At the transition points the function jumps discontinuously — the derivative is undefined. You can't backpropagate through a function with no useful gradient.

The **straight-through estimator (STE)** sidesteps this with a deliberately inconsistent pair of definitions for the forward and backward pass. During the forward pass, apply quantization normally — the network sees the actual quantized values. During the backward pass, *pretend quantization didn't happen* and pass the gradient through unchanged:

$$
\text{Forward:} \quad \hat{w} = \text{quant}(w)
$$

$$
\text{Backward:} \quad \frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial \hat{w}}
$$

This is a **biased gradient estimator** — the gradient the optimizer sees doesn't correspond to the actual forward computation. Yet it works remarkably well in practice. The intuition: the STE gradient points in roughly the right direction, and over many steps the optimizer learns to place weights near quantization grid points where rounding error is minimal. Weights that start between two levels get nudged toward the nearest one; weights that are already close to a grid point stay put.

Think of the forward pass as following a staircase function (flat steps with jumps), while the backward pass follows the identity function (a straight diagonal line). The "straight-through" name comes from this backward path: the gradient passes straight through the quantization operation as if it were the identity.

<!-- DIAGRAM: ste.svg — two side-by-side plots. Left: "Forward" showing staircase quantization function. Right: "Backward" showing identity function (straight line) with annotation "gradient passes through unchanged". -->

### FP8 mixed-precision training

**Traditional mixed-precision training** (Micikevicius et al. [[5]](#ref-5)) established the pattern: keep **master weights** in FP32 for the optimizer update, but run the forward and backward GEMM operations in FP16 or BF16. The reduced-precision GEMMs are $2\times$ faster on Tensor Cores, while the FP32 master copy prevents the small optimizer updates from vanishing due to limited mantissa bits.

**FP8 mixed-precision training** extends this idea one step further. DeepSeek-V3 [[8]](#ref-8) demonstrated end-to-end FP8 training at the 671B-parameter scale with negligible quality loss. The recipe splits the two FP8 sub-formats across their natural roles:

- **Forward GEMMs:** weights and activations are cast to **E4M3** (higher precision, narrower range). The Tensor Core accumulates partial sums in FP32 to avoid rounding error in the reduction.
- **Backward GEMMs:** gradients are cast to **E5M2** (wider dynamic range, coarser precision). Gradients span many orders of magnitude — the extra exponent bit in E5M2 prevents overflow/underflow. Again, accumulation in FP32.
- **Master weights:** kept in FP32 for the optimizer (Adam) update. The optimizer's momentum and variance terms also stay in FP32. Only the GEMM operands are in FP8.

**Per-tile scaling** is critical to making this work. DeepSeek-V3 uses **128 x 128 tile granularity** for scale factors — each $128 \times 128$ block of a GEMM operand gets its own FP8 scale. This is finer than per-tensor scaling (which would lose too much resolution for a 671B model) but coarser than per-element (which would be prohibitively expensive). The tile size aligns with the Tensor Core's native tile dimensions, so the scaling overhead is minimal.

The result: training FLOPs are dominated by FP8 Tensor Core operations, which deliver roughly $2\times$ throughput over BF16 on Hopper GPUs. DeepSeek-V3 reported that FP8 training matched BF16 training quality while substantially reducing compute cost — making FP8 the new practical baseline for large-scale training.

### Loss scaling

When training in reduced precision, small gradient values can **underflow** — round to zero in the limited-precision format. A gradient of $10^{-6}$ is well within FP32's range but may vanish in FP16 (smallest normal: $\sim 6 \times 10^{-5}$). Once a gradient underflows, the corresponding weight stops learning.

**Static loss scaling** is the simplest fix: multiply the loss by a large constant (e.g., 1024) *before* the backward pass. By the chain rule, every gradient in the network is scaled by the same factor, shifting the entire gradient distribution into the representable range. After the backward pass, divide all gradients by the same constant before the optimizer step:

$$
\mathcal{L}_{\text{scaled}} = s \cdot \mathcal{L}, \quad \nabla w = \frac{1}{s} \cdot \frac{\partial \mathcal{L}_{\text{scaled}}}{\partial w}
$$

The problem: if the scale is too large, gradients *overflow* (become Inf/NaN). If too small, they still underflow. A fixed constant can't adapt to the gradient distribution as it evolves during training.

**Dynamic loss scaling** solves this by adjusting the scale at runtime. Start with a large scale (e.g., $2^{16}$). If a NaN or Inf appears in the gradients, the step is skipped and the scale is halved. If $N$ consecutive steps are stable (no NaN/Inf), the scale is doubled. This finds and tracks the largest safe scale automatically. Dynamic loss scaling is the default in most mixed-precision frameworks (PyTorch's `GradScaler`, NVIDIA Apex).

For FP8 training, loss scaling is less critical than it was for FP16. **E5M2** (used for gradients) has 5 exponent bits — the same dynamic range as FP16 — so it faces similar underflow risks, and loss scaling is still applied. **BF16**, with its 8 exponent bits matching FP32's range, rarely needs loss scaling at all: its $\sim 10^{-38}$ minimum normal is small enough to represent virtually any gradient encountered in practice.

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

For a deeper treatment of KV cache compression strategies — including mixed-precision caching, adaptive bit-width, and token eviction — see [post 18]({{< relref "18-kv-cache-compression" >}}).

### Sensitivity summary

| Point | Operands | Scaling | Sensitivity | Why |
|---|---|---|---|---|
| QKV projections | weight $\times$ activation | per-channel $\times$ per-token | Low | Standard GEMM, no amplification |
| $QK^\top$ | activation $\times$ activation | per-token $\times$ per-token | **High** | Softmax amplifies errors exponentially |
| Score $\times$ V | softmax output $\times$ activation | (bounded) $\times$ per-token | Medium | Bounded weights dampen errors |
| KV cache | stored activations | per-head / per-channel | Medium | Error persists across full sequence |

<!-- DIAGRAM: attention-quant-points.svg — an attention layer diagram (Q,K,V projections → QK^T → softmax → score*V → output projection) with the four quantization points highlighted in different colors, each labeled with sensitivity level. -->

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
