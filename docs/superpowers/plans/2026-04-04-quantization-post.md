# Appendix C: Quantization Post — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draft the full content for Appendix C: Quantization for Attention — a two-act blog post covering quantization foundations and attention-specific fusion.

**Architecture:** Two-act structure. Act 1 (sections 3-7) is a self-contained quantization primer: number formats, scaling, PTQ, QAT, hardware. Act 2 (sections 8-11) applies those concepts to attention: quantization points, kernel fusion, MLA paths, FFN/MoE. Each task writes 1-2 sections, verifies with Hugo build, and commits.

**Tech Stack:** Hugo static site (theme: sans), Markdown with KaTeX math (`$$..$$` display, `$..$` inline), Excalidraw SVG diagrams

---

## Conventions (read before starting any task)

**File being edited:** `content/posts/appendix-c-quantization/index.md`

**Frontmatter** (already exists, update tags/description as needed):
```yaml
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
```

**Math conventions:**
- Display math: `$$...$$` (on own lines)
- Inline math: `$...$`
- Dimension variables: consistent with series ($n$ = seq_len, $d$ = d_model, $d_k$ = d_model/h)

**Cross-references:** Use Hugo `relref` shortcode:
```
[MLA post]({{< relref "04-mla" >}})
```

**Reference anchors:** Use `<span id="ref-N">[N]</span>` format (see MLA post for example).

**Section style (from MLA post exemplar):**
- Each section starts with the *why* before the *what*
- Tables for comparisons
- Bold for key terms on first introduction
- Math blocks for formulas, with prose explanation following each formula
- Code blocks with shape comments for any PyTorch snippets

**Diagrams:** Each diagram is an SVG file in the same directory (`content/posts/appendix-c-quantization/`). Referenced as `![alt text](filename.svg)`. Diagrams will be described in placeholder `<!-- DIAGRAM: ... -->` comments for later Excalidraw creation.

**Hugo build verification:**
```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```
The build should complete without errors. Warnings about missing pages are acceptable for `relref` links to posts that are still skeletons.

---

### Task 1: Frontmatter, TL;DR, and Motivation

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace entire file)

- [ ] **Step 1: Replace the skeleton file with frontmatter + TL;DR + Motivation**

Replace the full contents of `content/posts/appendix-c-quantization/index.md` with the updated frontmatter (expanded tags and description from Conventions above) followed by:

**TL;DR** (~5 sentences): Quantization maps FP32/BF16 to fewer bits. Key decisions: number format (INT8/FP8/FP4), scale granularity (per-tensor through per-group), and timing (PTQ vs QAT). For attention: quantization intersects QK^T and score*V GEMMs, KV cache, and kernel fusion. Hopper added FP8 (FA v3), Blackwell added FP4 (FA v4). DeepSeek-V3 trains end-to-end in FP8.

**Motivation** (3 paragraphs with subsection headers):

1. **The bandwidth wall** — Attention decode is memory-bandwidth-bound. Cite the KV cache formula from post 03:
$$\text{KV cache} = 2 \times L \times H \times d_k \times s \times B \times \text{sizeof(dtype)}$$
Halving sizeof(dtype) from 2 bytes (BF16) to 1 byte (FP8) nearly halves the memory and bandwidth cost.

2. **Tensor Core throughput scaling** — Each generation adds lower-precision datapaths that roughly double peak TFLOPS. Using them requires quantized operands. Link to [Appendix A]({{< relref "appendix-a-gpu-architecture" >}}).

3. **The challenge** — Attention has unique quantization sensitivity. Softmax is an exponential — small errors in $QK^\top$ scores get amplified. The KV cache persists quantization error across the full sequence. Generic GEMM quantization recipes don't account for this.

End the file with placeholder section headers for the rest of the post (so Hugo builds cleanly):
```markdown
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
```

- [ ] **Step 2: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

Expected: Build succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add TL;DR and motivation sections"
```

---

### Task 2: Number Formats (Act 1, Section 3)

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Number Formats` placeholder)

- [ ] **Step 1: Write the Number Formats section**

Replace the `## Number Formats` placeholder with full content:

**Opening paragraph:** Explain that quantization's first decision is the target format. Two families: integer (fixed step size, simple hardware) and floating-point (variable step size via exponent, better for values with wide dynamic range). Modern LLM quantization has shifted from integer to floating-point formats.

**Comparison table:**

| Format | Bits | Layout | Range | Precision | Primary use |
|--------|------|--------|-------|-----------|-------------|
| FP32 | 32 | 1S + 8E + 23M | $\pm 3.4 \times 10^{38}$ | ~7 decimal digits | Master weights |
| BF16 | 16 | 1S + 8E + 7M | $\pm 3.4 \times 10^{38}$ | ~2 decimal digits | Training/inference baseline |
| FP16 | 16 | 1S + 5E + 10M | $\pm 6.5 \times 10^{4}$ | ~3 decimal digits | Legacy |
| INT8 | 8 | 1S + 7 magnitude | $[-128, 127]$ | 1 unit step | PTQ weight quant |
| FP8 E4M3 | 8 | 1S + 4E + 3M | $\pm 448$ | ~1 decimal digit | Weights + activations |
| FP8 E5M2 | 8 | 1S + 5E + 2M | $\pm 57344$ | coarser | Gradients |
| NVFP4 (MX) | 4 | 1E + 2M + shared 8-bit scale | per-block | very coarse | Blackwell TC |

**Subsections within Number Formats:**

**### Integer vs floating-point**
- INT8: 256 evenly spaced levels. Good when values are roughly uniform. For weights with a Gaussian-like distribution, many levels are "wasted" in the tails.
- FP8: the exponent gives variable step sizes — small steps near zero (where most weights live), larger steps for outliers. This matches neural network value distributions better.

**### FP8: E4M3 vs E5M2**
- E4M3: 4 exponent bits, 3 mantissa bits. Range $\pm 448$, 15 distinct exponent values. Better precision near zero — preferred for weights and forward activations.
- E5M2: 5 exponent bits, 2 mantissa bits. Range $\pm 57344$, 31 exponent values. Wider dynamic range at the cost of precision — preferred for gradients (which span a wide range during training).
- Show the bit layout of each with a diagram placeholder:
```markdown
<!-- DIAGRAM: bit-layouts.svg — side-by-side bit field diagrams for FP32, BF16, FP8 E4M3, FP8 E5M2, NVFP4. Each shows sign/exponent/mantissa fields with bit counts labeled. -->
```
- Concrete example: represent the value $0.1875$ in FP32, BF16, FP8 E4M3. Show the bit pattern for each.

**### NVFP4 and microscaling (MX)**
- At 4 bits, you have only 16 representable values — far too few for any useful dynamic range on their own.
- Microscaling: a block of $B$ values (typically $B = 32$) shares a single FP8 scale factor. Each value is 4 bits (1 exponent + 2 mantissa), and the shared scale "shifts" the entire block's representable range.
- Effective bits per value: $4 + 8/B$. For $B = 32$: $4.25$ bits per value.
- The OCP MX specification [[7]](#ref-7) standardizes this format. NVIDIA's Blackwell Tensor Cores consume NVFP4 natively.

- [ ] **Step 2: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add number formats section"
```

---

### Task 3: Scaling Strategies (Act 1, Section 4)

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Scaling Strategies` placeholder)

- [ ] **Step 1: Write the Scaling Strategies section**

**Opening paragraph:** Even with the right number format, you need to map the actual value range into the representable range. The **scale factor** does this mapping. The key design choice: how many values share a single scale factor?

**The quantize-dequantize round-trip** (display math):

$$X_q = \text{clamp}\!\left(\left\lfloor \frac{X}{s} \right\rceil,\; q_{\min},\; q_{\max}\right)$$

$$\hat{X} = s \cdot X_q$$

where $s$ is the scale factor, $\lfloor \cdot \rceil$ is round-to-nearest, and $[q_{\min}, q_{\max}]$ is the format's representable range. The **quantization error** is $X - \hat{X}$.

**### Per-tensor scaling**
- One scale for the entire tensor: $s = \max(|X|) / q_{\max}$
- Cheapest: one scale to store and apply. But a single outlier stretches the range, wasting resolution for the majority of values.
- Acceptable for 8-bit weights (narrow distribution), poor for activations (outlier-prone).

**### Per-channel scaling**
- One scale per row or column of a weight matrix.
- For a weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$: one scale per output channel (row), so $d_{\text{out}}$ scales.
- Standard for weight quantization. Each output channel can have a different range without affecting others.

**### Per-token scaling**
- One scale per row of an activation matrix $X \in \mathbb{R}^{n \times d}$: one scale per token.
- Activations vary wildly across tokens — a token representing "the" vs one representing a rare technical term may have 10× different magnitude. Per-token scaling captures this.
- Must be computed dynamically at runtime (the activations aren't known ahead of time).

**### Per-group / block scaling (MX)**
- One scale per contiguous block of $B$ values.
- Finest granularity: approaches per-element accuracy at the cost of $\text{sizeof(scale)} / B$ overhead per value.
- This is the MX/NVFP4 approach: $B = 32$, scale is FP8, giving 0.25 bits overhead per value.
- Diagram placeholder:
```markdown
<!-- DIAGRAM: scaling-granularity.svg — a matrix with colored overlays showing: per-tensor (one color for whole matrix), per-channel (one color per row), per-token (one color per column), per-group (small blocks within the matrix). Each has its scale factor labeled. -->
```

**### Why per-token × per-channel is the standard**
- For a GEMM $Y = X W^\top$: $X$ has shape $(n, d_{\text{in}})$, $W$ has shape $(d_{\text{out}}, d_{\text{in}})$.
- Per-token scaling on $X$ (rows) and per-channel scaling on $W$ (rows) means each output element $Y_{i,j}$ has a combined scale $s_i^X \cdot s_j^W$ — a simple outer product of scales, no expensive per-element work.
- This is why FP8 GEMM libraries default to per-token × per-channel: it's the finest granularity that doesn't break the GEMM structure.

- [ ] **Step 2: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add scaling strategies section"
```

---

### Task 4: Post-Training Quantization (Act 1, Section 5)

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Post-Training Quantization (PTQ)` placeholder)

- [ ] **Step 1: Write the PTQ section**

**Opening paragraph:** PTQ quantizes a pre-trained model without retraining. You have the model weights and a small calibration dataset — no training loop, no gradient updates. The challenge: find a quantized representation that preserves the model's output quality despite reducing precision.

**### Round-to-nearest (RTN)**
- The simplest approach: quantize each weight independently to the nearest representable value.
- Formula: $w_q = s \cdot \lfloor w / s \rceil$ where $s$ is determined by the scale strategy.
- Works well at 8 bits (the rounding error is small relative to the value range).
- Breaks at 4 bits: with only 16 levels, rounding errors accumulate across the dot product. A single layer's degradation is manageable, but compounded across 80+ layers, output quality collapses.

**### GPTQ**
- Key idea: quantize weights one at a time, and *compensate* the remaining (not-yet-quantized) weights for the error introduced.
- Builds on Optimal Brain Quantization (OBQ): uses the Hessian of the layer's reconstruction loss to find the optimal compensation.
- For weight $w_q$ being quantized, the update to remaining weights $\mathbf{w}_F$ is:

$$\delta_F = -\frac{w_q - \text{quant}(w_q)}{[H_F^{-1}]_{qq}} \cdot (H_F^{-1})_{:,q}$$

where $H_F$ is the Hessian of the remaining weights and $[H_F^{-1}]_{qq}$ is the diagonal element for the quantized weight.

- GPTQ's innovation: process weights in a fixed order (columns of $W$) and use a lazy Hessian update, reducing complexity from $O(d^3)$ per weight to $O(d^2)$ amortized.
- Enables accurate 4-bit weight quantization. Widely used for INT4 models (Llama, Mistral community quantizations).

**### AWQ (Activation-Aware Weight Quantization)**
- Observation: not all weight channels are equally important. ~1% of channels correspond to consistently large activations — these "salient" channels carry disproportionate signal.
- Rather than quantize all channels the same way, AWQ applies a per-channel scaling factor $\mathbf{s}$ *before* quantization:

$$\hat{W} = \text{quant}(W \cdot \text{diag}(\mathbf{s}))$$

where $\mathbf{s}$ is chosen to equalize the quantization difficulty across channels, protecting the salient ones.

- The scale $\mathbf{s}$ is determined by the activation magnitudes from a calibration set: channels with large activations get larger scales (effectively more quantization resolution).
- Simpler than GPTQ (no Hessian), faster calibration, comparable quality at INT4.

**### SmoothQuant**
- Problem: weight-only quantization (W8A16 or W4A16) doesn't help with activation-bottlenecked scenarios (large batch inference). You want W8A8 — both weights and activations quantized.
- Challenge: activations have outlier channels (individual dimensions with values 10-100× larger than the rest), making per-tensor activation quantization terrible.
- SmoothQuant's insight: mathematically migrate the quantization difficulty from activations to weights. Apply a per-channel scaling:

$$Y = (X \cdot \text{diag}(\mathbf{s})^{-1}) \cdot (\text{diag}(\mathbf{s}) \cdot W^\top) = \hat{X} \hat{W}^\top$$

- $\hat{X}$ has smoothed-out outliers (easier to quantize), $\hat{W}$ has absorbed them (but weights are more tolerant of quantization).
- The smoothing factor $\mathbf{s}$ balances the difficulty: $s_j = \max(|X_{:,j}|)^\alpha / \max(|W_{:,j}|)^{1-\alpha}$ with $\alpha \in [0, 1]$ as a hyperparameter.

- [ ] **Step 2: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add PTQ section (RTN, GPTQ, AWQ, SmoothQuant)"
```

---

### Task 5: Quantization-Aware Training (Act 1, Section 6)

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Quantization-Aware Training (QAT)` placeholder)

- [ ] **Step 1: Write the QAT section**

**Opening paragraph:** PTQ treats the model as frozen — it can only rearrange the quantization to minimize damage. QAT goes further: it lets the model *adapt* to quantization during training, learning weight values that are robust to reduced precision. The cost is a training run; the payoff is near-zero quality loss even at aggressive bit-widths.

**### The straight-through estimator (STE)**
- The fundamental problem: quantization (rounding + clamping) is piecewise-constant, so its gradient is zero almost everywhere. You can't backpropagate through it.
- STE's solution: during the *forward* pass, apply quantization. During the *backward* pass, pretend it didn't happen — pass the gradient straight through:

$$\text{Forward:} \quad \hat{w} = \text{quant}(w)$$
$$\text{Backward:} \quad \frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial \hat{w}}$$

- This is a biased gradient estimator — it ignores the rounding error — but it works remarkably well in practice. The model learns to place weights near quantization grid points, minimizing the rounding error the STE ignores.
- Visualize: the forward pass follows the staircase quantization function, the backward pass follows the identity (a straight diagonal line). Diagram placeholder:
```markdown
<!-- DIAGRAM: ste.svg — two side-by-side plots. Left: "Forward" showing the staircase quantization function. Right: "Backward" showing the identity function (straight line) with annotation "gradient passes through unchanged". -->
```

**### FP8 mixed-precision training**
- Traditional mixed precision (Micikevicius et al. [[5]](#ref-5)): master weights in FP32, forward/backward GEMMs in FP16/BF16.
- FP8 mixed precision (DeepSeek-V3 [[8]](#ref-8)): extends this to FP8 for the GEMM operands:
  - **Forward GEMMs**: weights and activations cast to E4M3, accumulated in FP32
  - **Backward GEMMs**: gradients cast to E5M2 (wider range for gradient magnitudes), accumulated in FP32
  - **Master weights**: kept in FP32 for the optimizer update
- **Per-tile scaling**: DeepSeek-V3 uses a fine-grained scaling approach — each $128 \times 128$ tile of the GEMM operands gets its own scale factor. This is finer than per-tensor but coarser than per-element, balancing accuracy and overhead.
- The result: training FLOPs dominated by FP8 Tensor Cores (~2× throughput over BF16) with negligible quality loss.

**### Loss scaling**
- When training in reduced precision, small gradient values can underflow (round to zero), stalling learning for parameters with small gradients.
- **Static loss scaling**: multiply the loss by a constant (e.g., 1024) before backprop, then divide gradients by the same constant after. Shifts the gradient distribution into the representable range.
- **Dynamic loss scaling**: start with a large scale, halve it if NaN/Inf gradients appear, double it every $N$ steps if training is stable. This is the default in most mixed-precision frameworks.
- For FP8 training: E5M2's wider dynamic range (exponent 5 bits) reduces the need for aggressive loss scaling compared to FP16 (exponent 5 bits, but narrower total range). BF16 (exponent 8 bits) rarely needs loss scaling at all.

- [ ] **Step 2: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add QAT section (STE, FP8 training, loss scaling)"
```

---

### Task 6: Hardware Support (Act 1, Section 7) + Appendix A update

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Hardware Support` placeholder)
- Modify: `content/posts/appendix-a-gpu-architecture/index.md` (expand Tensor Cores section with TFLOPS and dataflow details)

- [ ] **Step 1: Write the Hardware Support section in the quantization post**

**Opening paragraph:** The number format you quantize to must match what the hardware's fast path can consume. Each NVIDIA GPU generation added Tensor Core support for a lower-precision datapath — and the throughput gains are substantial. See [Appendix A]({{< relref "appendix-a-gpu-architecture" >}}) for the full GPU architecture story; here we focus on what matters for quantized computation.

**Format support table:**

| Generation | GPU | Quantized formats | Approx. peak TFLOPS |
|---|---|---|---|
| Ampere (SM 80) | A100 | FP16, BF16, TF32, INT8 | ~312 (FP16) |
| Hopper (SM 90) | H100 | + FP8 (E4M3, E5M2) | ~990 (FP16), ~1979 (FP8) |
| Blackwell (SM 100) | B200 | + FP4 (NVFP4/MX) | ~2250 (FP16), ~4500 (FP8), ~9000 (FP4) |

Note: TFLOPS numbers are approximate peak dense performance. Real workloads achieve a fraction depending on memory-boundedness and kernel efficiency.

**### Tensor Core GEMM dataflow**
- Key insight: **quantized operands do not mean quantized arithmetic.** The Tensor Core pipeline:

1. **Load**: operands read from memory in reduced precision (FP8, FP4) — this is where bandwidth savings come from
2. **Multiply**: the element-wise products are computed at operand precision (or a slightly wider intermediate)
3. **Accumulate**: partial products are summed in an **FP32 accumulator** — this prevents quantization errors from compounding across the dot product
4. **Store**: the FP32 result is either kept for the next operation or re-quantized back to reduced precision

- Diagram placeholder:
```markdown
<!-- DIAGRAM: gemm-dataflow.svg — pipeline diagram. Two input arrows labeled "FP8 A" and "FP8 B" feed into a "Multiply (FP8)" box, which feeds into an "Accumulate (FP32)" box with a feedback loop, which outputs to "FP32 result" with an optional branch to "re-quantize → FP8". The FP32 accumulator box is highlighted as the key element. -->
```

- **Why this matters**: the FP32 accumulator is why FP8 GEMMs can match BF16 output quality. Each individual FP8 multiply has ~0.5% relative error, but the FP32 sum of 4096 such products has far less aggregate error than if you accumulated in FP8.
- **FP4 implication**: coarser operands (only 8 distinct magnitude levels per value) mean each multiply has larger error. The FP32 accumulator still contains the damage, but you need finer-grained scaling (per-group MX, not per-tensor) to keep the input error small enough.
- **The re-quantize decision**: between successive GEMMs, you can keep the FP32 result (accurate but 4× the memory) or re-quantize to FP8 (compact but adds another round of error). In attention kernels, this decision is made per-tile in SRAM — which brings us to kernel fusion (Act 2).

- [ ] **Step 2: Update Appendix A with Tensor Core TFLOPS and dataflow details**

In `content/posts/appendix-a-gpu-architecture/index.md`, expand the `## Tensor Cores` section's TODO comment to include the TFLOPS table and dataflow description. Keep it as the authoritative detailed reference — the quantization post will link here.

Add to the Tensor Cores TODO:
- The TFLOPS table (same as above but with all precision levels: FP64, TF32, FP16, BF16, FP8, FP4, INT8)
- The accumulator precision note (FP32 for all reduced-precision paths)
- WGMMA (Hopper): warp-group level MMA that feeds FP8 operands from shared memory
- UMMA (Blackwell): single-thread launch, feeds FP4 operands, accumulates to TMEM

This is expanding the existing TODO, not writing the full section — just add enough detail that the `relref` link from the quantization post is useful.

- [ ] **Step 3: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md content/posts/appendix-a-gpu-architecture/index.md && git commit -m "draft(appendix-c): add hardware support section, expand appendix-a tensor cores"
```

---

### Task 7: Where Quantization Hits Attention (Act 2, Section 8)

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Where Quantization Hits Attention` placeholder)

- [ ] **Step 1: Write the section**

**Opening paragraph:** Bridge from Act 1 to Act 2. The foundations above apply to any GEMM. Attention has four distinct quantization points, each with different operand characteristics and error sensitivity. Understanding these differences is what separates "quantize everything to FP8" from a quantization strategy that actually works.

**### The four quantization points**

Walk through each in a numbered list with its own subsection:

**### 1. QKV projection GEMMs**
- $Q = XW^Q$, $K = XW^K$, $V = XW^V$ — standard weight × activation GEMMs
- Weight quantization: per-channel, static (weights don't change)
- Activation quantization: per-token, dynamic (computed at runtime)
- Nothing attention-specific here — same treatment as FFN GEMMs

**### 2. $QK^\top$ (score computation)**
- Both $Q$ and $K$ are activations — no static weights. Both operands need dynamic per-token scaling.
- **Softmax sensitivity**: the attention score $s_{ij} = q_i \cdot k_j / \sqrt{d_k}$ passes through softmax: $\alpha_{ij} = e^{s_{ij}} / \sum_k e^{s_{ik}}$. A quantization error $\epsilon$ in $s_{ij}$ becomes a multiplicative factor $e^\epsilon$ in the attention weight — errors are *amplified exponentially*.
- Show math: if the true score is $s$ and the quantized score is $s + \epsilon$:
$$\frac{\text{softmax}(s + \epsilon)}{\text{softmax}(s)} \approx e^{\epsilon} \cdot \frac{\sum e^{s_k}}{\sum e^{s_k + \epsilon_k}}$$
The error doesn't simply add — it redistributes attention mass.
- This is the most error-sensitive of the four points.

**### 3. Score $\times$ V (value aggregation)**
- Post-softmax attention weights (range $[0, 1]$, sum to 1) times $V$.
- The attention weights are well-bounded — friendly to quantize (no outliers, known range).
- $V$ has the same dynamic range challenges as $K$. Per-token scaling on $V$.
- Less sensitive than $QK^\top$ because softmax clamped the weights to $[0, 1]$.

**### 4. KV cache storage**
- $K$ and $V$ quantized to FP8/FP4 at write time, dequantized at every decode step.
- Error persists for the token's entire lifetime in the sequence — but it's a single quantize-dequantize cycle, not cumulative.
- Per-channel or per-head scaling is typical (the cached tensors have shape $(s, d_k)$ per head).
- The aggregate effect: across $s$ cached tokens, the per-token quantization errors form a distribution. For long sequences ($s > 4096$), this distribution matters more than any individual error.
- Link to [KV Cache Compression (post 18)]({{< relref "17-batching-scheduling" >}}) for KIVI and KV-Quant details.

**### Sensitivity summary table:**

| Point | Operands | Scaling | Sensitivity | Why |
|---|---|---|---|---|
| QKV projections | weight × activation | per-channel × per-token | Low | Standard GEMM |
| $QK^\top$ | activation × activation | per-token × per-token | **High** | Softmax amplifies |
| Score × V | softmax output × activation | (bounded) × per-token | Medium | Bounded weights help |
| KV cache | stored activations | per-head / per-channel | Medium | Persists full sequence |

- Diagram placeholder:
```markdown
<!-- DIAGRAM: attention-quant-points.svg — an attention layer diagram (Q, K, V projections → QK^T → softmax → score*V → output projection) with the four quantization points highlighted in different colors, each labeled with its sensitivity level. -->
```

- [ ] **Step 2: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add attention quantization points section"
```

---

### Task 8: Kernel Fusion (Act 2, Section 9)

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Kernel Fusion` placeholder)

- [ ] **Step 1: Write the Kernel Fusion section**

**Opening paragraph:** Having the right number format and scaling strategy is necessary but not sufficient. If each quantize/dequantize step is a separate CUDA kernel launch — reading from HBM, processing, writing back to HBM — the memory round-trips can erase the throughput gains. **Kernel fusion** folds the quantization operations into the attention kernel itself, keeping intermediate values in SRAM.

**### The unfused baseline**

Show the kernel launch sequence for a naively quantized attention layer:

```
1. Launch: QKV projection (FP8 GEMM) → write Q, K, V to HBM in FP8
2. Launch: dequant Q, K to FP32 → write to HBM
3. Launch: QK^T GEMM (FP32) → write scores to HBM
4. Launch: softmax → write attention weights to HBM
5. Launch: quant attention weights to FP8 → write to HBM
6. Launch: score*V GEMM (FP8) → write output to HBM
```

Each "→ write to HBM" is a round-trip: ~2-3 TB/s bandwidth shared across all SMs. For a sequence of length 4096 with $d_k = 128$, the score matrix alone is $4096 \times 4096 \times 4 = 64$ MB — written and read multiple times.

**### The fused approach**

Fused attention (FlashAttention-style) keeps the entire score → softmax → value aggregation pipeline in SRAM:

```
1. Launch: QKV projection (FP8 GEMM) → Q, K, V in HBM (FP8)
2. Launch: fused attention kernel
   - Load Q tile, K tile from HBM (FP8)
   - QK^T in FP8, accumulate in FP32 (in registers)
   - Online softmax in FP32 (in registers)
   - Score * V in FP8, accumulate in FP32 (in registers)
   - Write final output to HBM
```

The intermediate score matrix never materializes in HBM — it lives in registers/SRAM for the duration of the tile computation.

**### FlashAttention v3 (Hopper, FP8)**
- FA v3 [[9]](#ref-9) targets Hopper's FP8 Tensor Cores (WGMMA instructions).
- Both GEMMs ($QK^\top$ and $\text{score} \times V$) use FP8 E4M3 operands with FP32 accumulation.
- Online softmax (the log-sum-exp running computation from FA v1/v2) stays in FP32 — no quantization on the softmax itself.
- The quantize/dequantize for intermediate values happens in registers, zero HBM traffic.
- Per-token scaling on Q, K, V is folded into the tile loading step.
- See [FlashAttention v3/v4]({{< relref "10-flash-attention-v3-v4" >}}) for the full kernel design.

**### FlashAttention v4 (Blackwell, FP4)**
- FA v4 [[10]](#ref-10) targets Blackwell's FP4 Tensor Cores (UMMA instructions).
- NVFP4 operands with MX block scaling: each $32$-element block has its own FP8 scale.
- The kernel manages the block scale bookkeeping within each tile — loading scales alongside operands from HBM, applying them during the GEMM.
- FP32 accumulation and online softmax remain unchanged from FA v3.
- See [FlashAttention v3/v4]({{< relref "10-flash-attention-v3-v4" >}}) for details.

**### KV cache dequant fusion**
- In serving frameworks (vLLM, TensorRT-LLM), the KV cache may be stored in FP8.
- Without fusion: a separate dequant kernel reads FP8 values from paged KV cache blocks, writes FP32/BF16 to a staging buffer, then the attention kernel reads from the buffer.
- With fusion: the attention kernel reads FP8 directly from the paged blocks and dequantizes in-register during the tile computation.
- This eliminates one full HBM read-write cycle per attention layer per decode step — significant at high batch sizes where KV cache bandwidth is the bottleneck.

- Diagram placeholder:
```markdown
<!-- DIAGRAM: fused-vs-unfused.svg — two-column comparison. Left: "Unfused" showing 6 kernel boxes with HBM arrows between each. Right: "Fused" showing 2 kernel boxes (QKV projection, fused attention) with only input/output HBM arrows. The eliminated HBM traffic is crossed out or grayed. -->
```

- [ ] **Step 2: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add kernel fusion section"
```

---

### Task 9: Quantization in MLA Paths (Act 2, Section 10)

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Quantization in MLA Paths` placeholder)

- [ ] **Step 1: Write the section**

**Opening paragraph:** [MLA]({{< relref "04-mla" >}}) introduced two distinct attention code paths: prefill (decompress and run standard MHA) and decode (absorb decompression into query, attend in latent space). Quantization interacts differently with each.

**### Prefill: standard FP8 attention**
- After decompression, $K$ and $V$ are standard-shaped: $(n, H, d_k)$
- Standard FP8 FlashAttention applies directly — same as any MHA model
- The latent $\mathbf{c}$ and decompression happen in BF16 (the projection GEMMs can use FP8)
- Quantization here is straightforward — no MLA-specific concerns

**### Decode: quantized latent cache**
- The absorbed decode path operates in latent space: attention over cached $\mathbf{c}$ vectors of dimension $d_c$
- Storing $\mathbf{c}$ in FP8: $576 \times 1 = 576$ bytes per token (vs 1152 bytes in BF16) — a further 2× reduction on top of MLA's architectural compression
- **Stacked compression**: MLA gives ~57× over MHA, FP8 gives another 2×, total ~114× KV cache reduction
- **Calibration caveat**: the absorbed query $\mathbf{q}' = \mathbf{q} \cdot {W^{\text{up}}_K}^\top$ has a different value distribution than a standard query (it includes the decompression matrix). Scale factors calibrated on standard attention may not transfer — MLA decode may need its own calibration pass
- The $\mathbf{c} \cdot {\mathbf{q}'}^\top$ dot product is in a higher-dimensional space ($d_c = 512$ vs $d_k = 128$), which actually *helps* quantization: more elements in the dot product means the FP32 accumulator averages out per-element errors better

**### Sparse attention indexer (FP8 keys)**
- DeepSeek-V3.2's NSA/DSA indexer ([post 07]({{< relref "07-sparse-attention" >}})) caches indexer keys in FP8
- The indexer makes a coarse-grained decision: "which tokens are worth attending to?" — not computing exact attention scores
- FP8 is sufficient because the indexer only needs to rank tokens roughly, not compute precise scores
- Memory saving: indexer cache is $(s, d_{\text{idx}})$ per head in FP8 (~132 bytes per token vs 264 in BF16)

- [ ] **Step 2: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add MLA quantization paths section"
```

---

### Task 10: Quantization in FFN / MoE (Act 2, Section 11)

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Quantization in FFN / MoE` placeholder)

- [ ] **Step 1: Write the section**

**Opening paragraph:** Attention gets the nuanced quantization story, but the FFN layers are where quantization delivers the most straightforward wins — large weight matrices, standard GEMM patterns, no softmax to worry about.

**### FFN weight quantization**
- A typical SwiGLU FFN has three weight matrices per layer: $W_{\text{gate}}$, $W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$
- For a model with $d = 7168$, $d_{\text{ff}} = 18432$ (DeepSeek-V3 dense dimensions): each FFN layer stores $3 \times 7168 \times 18432 = 396M$ parameters
- In BF16: ~792 MB per layer. In FP8: ~396 MB. In FP4: ~198 MB.
- No softmax sensitivity, no cache accumulation — activation quantization is simpler than in attention

**### MoE: the quantization multiplier**
- MoE models ([post 05]({{< relref "05-moe" >}})) have $N$ expert FFNs (e.g., 256 in DeepSeek-V3), but only $k$ experts activate per token
- Total expert parameters: $N \times 3 \times d \times d_{\text{ff}}^{\text{expert}}$. For DeepSeek-V3: $256 \times 3 \times 7168 \times 2048 \approx 11.3$ GB per MoE layer in BF16
- **Memory savings scale with $N$**: quantizing from BF16 to FP8 saves ~5.6 GB per MoE layer. Across all MoE layers, this determines whether the model fits on a given number of GPUs.
- **Compute savings scale with $k$**: only $k$ experts run per token, so the throughput gain applies to $k$ GEMMs, not $N$. But at the memory level, all $N$ experts' weights must be resident.
- This asymmetry makes quantization especially valuable for MoE: the memory pressure is from $N$ experts, but the compute is only $k$.

**### DeepSeek-V3 FP8 MoE training**
- DeepSeek-V3 [[8]](#ref-8) trained all 256 experts in FP8 from the start (QAT)
- Per-tile scaling ($128 \times 128$ tiles) for expert GEMMs, keeping fine-grained error control across the diverse expert weight distributions
- **Gate precision**: the routing gate (which selects which $k$ experts activate for each token) stays in BF16/FP32. Small errors in gate logits can flip the top-$k$ selection, sending a token to entirely different experts — a catastrophic error that doesn't heal. The gate's GEMM is small ($d \times N$), so keeping it in high precision costs negligible throughput.

- [ ] **Step 2: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add FFN/MoE quantization section"
```

---

### Task 11: Trade-offs, Adoption, and References

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (replace `## Trade-offs`, `## Adoption`, and `## References` placeholders)

- [ ] **Step 1: Write Trade-offs section**

**Comparison table:**

| Method | Bits | Quality impact | Throughput gain | Complexity | When to use |
|---|---|---|---|---|---|
| BF16 (baseline) | 16 | None | 1× | None | Training, quality-critical serving |
| INT8 PTQ (GPTQ/AWQ) | 8 | Minimal | ~1.5-2× | Offline calibration | Weight-dominated serving |
| SmoothQuant (W8A8) | 8 | Minimal | ~1.5-2× | Calibration + smoothing | Activation-bottlenecked serving |
| FP8 PTQ | 8 | Minimal | ~2× | Scale calibration | Hopper+ serving |
| FP8 QAT | 8 | Near-zero | ~2× | Training recipe | Frontier model training |
| FP4 / NVFP4 | 4 | Small-moderate | ~4× | Block scaling (MX) | Blackwell, throughput-critical |
| INT4 PTQ (GPTQ/AWQ) | 4 | Moderate | ~2-3× | Careful calibration | Memory-constrained deployment |

**Key trade-off axes** (prose, ~1 paragraph each):

1. **Precision vs throughput**: 16→8 bits is nearly free quality-wise; 8→4 requires fine-grained scaling to maintain quality. The gap between 8-bit and 4-bit is much larger than between 16-bit and 8-bit.

2. **Static vs dynamic quantization**: Static scales (calibrated offline) are faster but miss input-dependent outliers. Dynamic scales (computed per-token at runtime) add overhead but handle distribution shifts. FP8 with per-token dynamic scaling has become the sweet spot — the overhead of computing a per-token max is negligible relative to the GEMM.

3. **Where error compounds**: Weights are the most forgiving (same values every forward pass, easy to calibrate statically). Activations are harder (vary per input, need dynamic scaling). The KV cache is hardest (error persists for the full sequence, aggregates across thousands of tokens). Rule of thumb: you can afford coarser quantization for weights than for the KV cache.

4. **Fusion or it doesn't count**: Throughput gains assume fused kernels. An unfused FP8 attention path with separate dequant/quant kernel launches can be *slower* than BF16 FlashAttention because the extra HBM round-trips dominate. Kernel maturity — not just number format support — determines whether quantization helps in practice.

- [ ] **Step 2: Write Adoption section**

~5 bullet points:

- **Llama 3 / 3.1** (Meta, 2024): released with official INT8 and FP8 quantized checkpoints. Trained in BF16, PTQ for inference.
- **DeepSeek-V3** (DeepSeek, 2024): first major model trained end-to-end in FP8 (QAT). Per-tile scaling across all 256 MoE experts.
- **Mistral / Mixtral** (Mistral AI, 2023-2024): community GPTQ and AWQ INT4 quantizations widely used for local serving on consumer GPUs.
- **vLLM / TensorRT-LLM** (2024-2025): FP8 KV cache support, fused FP8 attention kernels for Hopper+. FP4 support in development for Blackwell.
- **Blackwell (B200)** (NVIDIA, 2025): FP4 Tensor Cores shipping. FlashAttention v4 is the first attention kernel targeting native FP4. Adoption still early as of early 2026.

- [ ] **Step 3: Write References section**

Use the series convention with `<span id="ref-N">` anchors:

```markdown
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
```

End with `*Last updated: April 2026*`.

- [ ] **Step 4: Verify Hugo build**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): add trade-offs, adoption, and references"
```

---

### Task 12: Full post review and final commit

**Files:**
- Modify: `content/posts/appendix-c-quantization/index.md` (any fixes from review)

- [ ] **Step 1: Read the full post end-to-end**

Read the complete file and check for:
- Math rendering: all `$$` blocks properly delimited, no unclosed dollar signs
- Internal consistency: variable names consistent ($d_k$, $d_c$, $s$, etc.)
- Reference anchors: all `[[N]](#ref-N)` links have matching `<span id="ref-N">` targets
- Cross-references: all `relref` links point to existing post directories
- No remaining `<!-- TODO -->` or `<!-- Section content: Task N -->` placeholders
- Diagram placeholders are descriptive enough for later Excalidraw creation
- Transitions between sections flow naturally (especially the Act 1 → Act 2 bridge)

- [ ] **Step 2: Fix any issues found**

Make targeted edits. Common issues:
- Missing `$$` closing tags
- Inconsistent variable names across sections
- Broken reference links
- Awkward transitions

- [ ] **Step 3: Verify Hugo build one final time**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && hugo --buildDrafts 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
cd /Users/yiqiyou/projects/all-attentions-you-need && git add content/posts/appendix-c-quantization/index.md && git commit -m "draft(appendix-c): review pass and fixes"
```
