# Appendix C: Quantization for Attention — Post Design

**Date:** 2026-04-04
**Post:** `content/posts/appendix-c-quantization/index.md`
**Scope:** Full quantization primer (foundations through attention-specific fusion)
**Structure:** Two-act hybrid — Act 1 (reference primer) + Act 2 (attention-specific)

---

## Design Decisions

- **Scope:** Broad — full primer from INT8/PTQ/QAT basics through FP8/FP4 and attention kernel fusion. Assumes no prior quantization knowledge.
- **Depth:** Balanced treatment of classic methods (PTQ, QAT) and attention-specific material. Each gets full math and examples.
- **Hardware:** Light recap with `relref` to Appendix A. New hardware details (FP8/FP4 TFLOPS, GEMM dataflow) get added to Appendix A and linked from here.
- **FFN/MoE coverage:** Brief section connecting quantization to FFN and MoE layers, bridging to post 05.

---

## Post Structure

### 1. TL;DR (~5 sentences)

Quantization maps high-precision values (FP32/BF16) to fewer bits (INT8/FP8/FP4), trading precision for throughput and memory savings. The key decisions are the number format (integer vs floating-point, exponent vs mantissa bit allocation), the scale granularity (per-tensor, per-channel, per-group, per-token), and when quantization happens (post-training calibration vs quantization-aware training). For attention specifically, quantization intersects with the QK^T and score*V GEMMs, the KV cache, and kernel fusion — where quantize/dequantize steps are folded into the attention kernel to eliminate extra memory round-trips. Hopper introduced FP8 Tensor Cores enabling FlashAttention v3's FP8 path; Blackwell added FP4 for FlashAttention v4. Frontier models like DeepSeek-V3 now train end-to-end in FP8.

### 2. Motivation (~3 paragraphs)

1. **The bandwidth wall** — Attention decode is memory-bandwidth-bound (reading KV cache). Halving the bytes per value nearly doubles decode throughput — quantization is the most direct lever.
2. **Tensor Core throughput scaling** — Each GPU generation roughly doubles Tensor Core TFLOPS by adding lower-precision datapaths (FP16 → FP8 → FP4). Using these requires quantized operands.
3. **The challenge** — Attention has unique sensitivity: softmax amplifies quantization errors in the score matrix, and the KV cache accumulates quantization drift across the full sequence length. Generic quantization recipes don't account for this.

---

## Act 1: Foundations

### 3. Number Formats

Comparison table of formats relevant to modern inference:

| Format | Bits | Layout | Dynamic Range | Use case |
|--------|------|--------|---------------|----------|
| FP32 | 32 | 8E+23M | huge | Training master weights |
| BF16 | 16 | 8E+7M | same as FP32 | Training/inference baseline |
| FP16 | 16 | 5E+10M | narrower | Legacy, mostly replaced by BF16 |
| INT8 | 8 | signed integer | 256 levels | PTQ weight quant (GPTQ, AWQ) |
| FP8 E4M3 | 8 | 4E+3M | wider mantissa | Weights and activations (Hopper+) |
| FP8 E5M2 | 8 | 5E+2M | more range | Gradients (training) |
| NVFP4 (MX) | 4 | 1E+2M + shared scale | very narrow | Blackwell Tensor Cores |

For each format:
- Bit layout diagram
- Representable range formula
- Concrete example (e.g., how 0.1875 is represented)
- E4M3 vs E5M2 trade-off (precision vs dynamic range) — why training uses E5M2 for gradients but E4M3 for forward activations
- NVFP4/MX: microscaling concept — groups of values share a scale factor at higher precision

### 4. Scaling Strategies

The core problem: mapping wide-range tensors into narrow-range formats. Four granularities with math:

- **Per-tensor**: one scale s = max(|X|) / qmax for the whole tensor. Cheapest, worst accuracy.
- **Per-channel**: one scale per output channel (weight rows). Standard for weight quantization.
- **Per-token**: one scale per token (activation rows). Handles different tokens having wildly different activation magnitudes.
- **Per-group / block**: one scale per block of B values (e.g., B=128). The MX/NVFP4 approach. Best accuracy, highest overhead.

Key math — the quantize-dequantize round-trip:
$$\hat{X} = s \cdot \text{clamp}(\text{round}(X / s), q_{min}, q_{max})$$

Explain why per-token × per-channel scaling is the standard choice for GEMM quantization (rows of activations × columns of weights).

### 5. Post-Training Quantization (PTQ)

~2-3 paragraphs each with key equations:

- **Naive round-to-nearest (RTN)** — baseline, works for 8-bit, breaks at 4-bit.
- **GPTQ** — layer-wise quantization using approximate second-order information (Hessian). Quantize one weight at a time, compensate remaining weights. Key equation: optimal update to remaining weights after quantizing weight w_q.
- **AWQ (Activation-Aware)** — ~1% of weight channels carry disproportionate signal (corresponding to large activations). Protect these with per-channel scaling before quantization.
- **SmoothQuant** — migrate quantization difficulty from activations to weights by balancing their ranges with a diagonal scaling matrix.

For each: what problem it solves, the core idea, and where it's used in practice.

### 6. Quantization-Aware Training (QAT)

- **Straight-through estimator (STE)** — forward pass: quantize; backward pass: pass gradients through as if quantization didn't happen. Show the math.
- **FP8 mixed-precision training** — the DeepSeek-V3 approach: forward GEMMs in FP8 (E4M3 for weights/activations), gradient communication in FP8 (E5M2), master weights in FP32/BF16. Per-tile scaling for fine-grained error control.
- **Loss scaling** — why reduced precision training needs loss scaling to keep small gradients from underflowing, and how it interacts with FP8's dynamic range.

### 7. Hardware Support

**Format support table** — which GPU generations support which formats, with approximate peak TFLOPS:

| Generation | Added | Approx. peak TFLOPS |
|---|---|---|
| Ampere (A100) | FP16/BF16/INT8 | ~312 (FP16) |
| Hopper (H100) | FP8 (E4M3/E5M2) | ~990 (FP16), ~1979 (FP8) |
| Blackwell (B200) | FP4 (NVFP4/MX) | ~2250 (FP16), ~4500 (FP8), ~9000 (FP4) |

**Tensor Core GEMM dataflow** — the key insight that quantized operands ≠ quantized math:

1. **Operands** loaded in reduced precision (FP8/FP4) — bandwidth/storage savings
2. **Multiply** at operand precision (or slightly higher)
3. **Accumulate** at FP32 — partial products summed in full-precision accumulator
4. **Output** kept in FP32 or re-quantized back to reduced precision

Pipeline diagram:
```
FP8 weights ──┐
              ├─ FP8 multiply ─→ FP32 accumulate ─→ FP32 output (or re-quantize)
FP8 activations┘
```

Practical implications:
- FP32 accumulation is why FP8 GEMMs match BF16 quality — quantization error doesn't snowball
- FP4 still accumulates in FP32, but coarser operands mean more per-element error — requires finer-grained scaling (per-group/MX) to compensate
- Dequantize placement matters: round-trip to HBM between GEMMs is wasteful → bridge to Act 2

Link to Appendix A for Tensor Core microarchitecture (WGMMA, UMMA, TMEM). New TFLOPS details get added to Appendix A.

---

## Act 2: Quantization in Attention

### 8. Where Quantization Hits Attention

Map out the four quantization points in an attention layer:

1. **QKV projection GEMMs** — weight quantization (INT8/FP8), activation quantization (per-token FP8). Standard GEMM quantization, nothing attention-specific.
2. **QK^T score GEMM** — both operands are activations (no static weights). Per-token scaling on both Q and K. Softmax amplifies small quantization errors in scores.
3. **Score × V GEMM** — attention weights (post-softmax, range [0,1]) times V. Softmax output is well-bounded (friendlier to quantize), but V has same dynamic range challenges as K.
4. **KV cache storage** — cached K and V stored in reduced precision (FP8/FP4) to cut memory. Quantized once at write time, dequantized at every decode step. Error doesn't compound (one round-trip), but aggregate error distribution across the full sequence matters.

For each point: which scaling strategy works best and why, plus sensitivity analysis (QK^T GEMM is most error-sensitive due to softmax amplification).

### 9. Kernel Fusion

**The problem** — without fusion, a quantized attention layer has multiple HBM round-trips:
```
GEMM(Q,K) → write FP32 to HBM → quant kernel → write FP8 to HBM → softmax kernel → ...
```

**Fused approach** — fold quantize/dequantize into the attention kernel:

- **FlashAttention v3 (Hopper, FP8)** — FP8 operands for both GEMMs, FP32 accumulation, online softmax in FP32, all in one fused kernel. Quant/dequant in SRAM, never touches HBM. Link to post 10.
- **FlashAttention v4 (Blackwell, FP4)** — NVFP4 operands with UMMA instructions. Block-wise MX scaling managed within tiles. Link to post 10.
- **vLLM quant fusion** — the dequant kernel for KV cache entries was a separate launch; fusing into the attention kernel eliminated overhead.

Before/after diagram: unfused (multiple kernel launches + HBM round-trips) vs fused (single kernel, all in SRAM).

### 10. Quantization in MLA Paths

Connect to the MLA post (04) — two code paths interact differently:

- **Prefill path**: decompressed K, V are standard-shaped → standard FP8 FlashAttention applies. Latent c computed in BF16, decompressed in BF16, attention GEMMs in FP8.
- **Decode path (absorbed)**: attention in latent space. Cached c stored in FP8 (576 bytes vs 1152 in BF16 — additional 2× on top of MLA's architectural compression). Absorbed query q' has different distribution than standard Q (includes W^up), may need different calibration.
- **Sparse attention indexer (post 07)**: FP8 cached keys in NSA/DSA indexer — FP8 is sufficient for coarse "which tokens matter" decisions.

### 11. Quantization in FFN / MoE

- **FFN GEMMs** — the two (or three, for SwiGLU) weight matrices per layer are the largest parameter blocks. Weight quantization gives the biggest absolute memory savings. Activation quantization simpler than attention — no softmax sensitivity, no cache accumulation.
- **MoE multiplier** — N expert FFNs (e.g., 256 in DeepSeek-V3), but only k active per token. Quantization's memory savings are critical for fitting the model; compute savings apply only to active experts. Link to post 05.
- **DeepSeek-V3 FP8 MoE training** — all 256 experts trained in FP8 with per-tile scaling. Expert routing (gate) stays in higher precision since small errors have outsized effects on which expert activates.

---

## Trade-offs

Comparison table:

| Method | Bits | Quality impact | Throughput gain | Complexity | When to use |
|---|---|---|---|---|---|
| BF16 (baseline) | 16 | None | 1× | None | Training, quality-critical |
| INT8 PTQ (GPTQ/AWQ) | 8 | Minimal | ~1.5-2× | Offline calibration | Serving weight-dominated models |
| SmoothQuant (W8A8) | 8 | Minimal | ~1.5-2× | Calibration + weight transform | Activation bottleneck |
| FP8 PTQ | 8 | Minimal | ~2× | Scale calibration | Hopper+ serving |
| FP8 QAT | 8 | Near-zero | ~2× | Training recipe change | Frontier training |
| FP4 / NVFP4 | 4 | Small-moderate | ~4× | Block scaling (MX) | Blackwell, throughput-critical |
| INT4 PTQ (GPTQ/AWQ) | 4 | Moderate | ~2-3× | Careful calibration | Memory-constrained |

Key trade-off axes (prose):
- **Precision vs throughput** — 16→8 is nearly free quality-wise; 8→4 requires careful scaling to maintain quality
- **Static vs dynamic quantization** — static scales faster but can't handle outliers; dynamic (per-token) adds overhead but handles distribution shifts. FP8 per-token dynamic is the sweet spot.
- **Where error compounds** — weights (forgiving) → activations (harder) → KV cache (hardest, persists full sequence). Appropriate bit-width decreases in that order.
- **Fusion or it doesn't count** — throughput gains assume fused kernels. Unfused FP8 attention can be slower than BF16 FlashAttention.

## Adoption

- **Llama 3 / 3.1** — official INT8 and FP8 quantized checkpoints, BF16 training
- **DeepSeek-V3** — trained end-to-end in FP8 (QAT), first major model to do so at scale
- **Mistral / Mixtral** — community GPTQ/AWQ INT4 widely used for local serving
- **vLLM / TensorRT-LLM** — FP8 KV cache support, fused FP8 attention kernels for Hopper+
- **Blackwell (B200)** — FP4 Tensor Cores shipping, FA v4 is first attention kernel targeting FP4. Adoption still early (early 2026).

## References

Key papers (numbered, with `<span id="ref-N">` anchors):
1. Dettmers et al. — LLM.int8()
2. Frantar et al. — GPTQ
3. Lin et al. — AWQ
4. Xiao et al. — SmoothQuant
5. Micikevicius et al. — Mixed Precision Training
6. NVIDIA — FP8 Formats for Deep Learning (whitepaper)
7. OCP — Microscaling (MX) Data Formats Specification
8. DeepSeek-AI — DeepSeek-V3 Technical Report
9. Shah et al. — FlashAttention-3
10. Shah et al. — FlashAttention-4
11. Liu et al. — KIVI: KV Cache Quantization
12. Hooper et al. — KV-Quant

## Cross-references to other posts

- Post 04 (MLA): MLA quantization paths (section 10)
- Post 05 (MoE): MoE quantization (section 11)
- Post 07 (Sparse Attention): FP8 indexer keys
- Post 09-10 (FlashAttention): FP8/FP4 fused kernels
- Post 18 (KV Cache Compression): quantized KV cache (KIVI, KV-Quant)
- Appendix A (GPU Architecture): Tensor Core details, TFLOPS numbers (expanded)

## Diagrams (Excalidraw)

1. **Bit layout diagrams** — side-by-side bit fields for FP32, BF16, FP8 E4M3, FP8 E5M2, NVFP4
2. **Scaling granularity visual** — a matrix with per-tensor, per-channel, per-token, per-group scale factors highlighted
3. **Tensor Core GEMM dataflow** — pipeline showing FP8 operands → multiply → FP32 accumulate → output
4. **Attention quantization map** — the four quantization points in an attention layer
5. **Fused vs unfused kernel** — before/after showing HBM round-trips eliminated by fusion
