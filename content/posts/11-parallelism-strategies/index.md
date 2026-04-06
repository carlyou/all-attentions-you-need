---
title: "Parallelism Strategies"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["parallelism", "data-parallel", "tensor-parallel", "pipeline-parallel", "sequence-parallel", "all-reduce", "dp", "tp", "sp", "pp"]
description: "The four building blocks for training and serving large models — data, tensor, sequence, and pipeline parallelism — and how they combine into 3D parallelism."
weight: 1051
---

## TL;DR

Data parallelism (DP) replicates the model and shards the data — it scales compute but not memory. Tensor parallelism (TP) splits weight matrices within a layer across GPUs, using all-reduce to recombine. Sequence parallelism (SP) extends TP to the non-matmul ops (LayerNorm, dropout) by decomposing all-reduce into reduce-scatter + all-gather — same communication, less memory. Pipeline parallelism (PP) assigns different layers to different GPUs and pipelines micro-batches to hide the bubble. Combined, they form 3D parallelism: TP within a node, PP across nodes, DP across replicas.

## Motivation

A 70B-parameter model in BF16 occupies 140 GB just for weights. During training, Adam optimizer states add another ~420 GB (two FP32 moments per parameter), and activations scale with batch size and sequence length. A single 80 GB GPU cannot hold even the weights alone.

Data parallelism — the default distributed strategy — doesn't help here. Each GPU holds a full model replica and processes a different data shard. It scales throughput linearly but does nothing for per-GPU memory.

We need **model parallelism**: splitting the model itself across GPUs. There are two orthogonal axes:

- **Within a layer** — tensor parallelism splits the weight matrices so each GPU computes a portion of each layer
- **Across layers** — pipeline parallelism assigns different layers to different GPUs

| Strategy | What it splits | Scales | Bottleneck |
|---|---|---|---|
| Data (DP) | Training data | Throughput | Per-GPU memory (full model) |
| Tensor (TP) | Weight matrices within a layer | Per-GPU memory | All-reduce communication |
| Pipeline (PP) | Layers across stages | Per-GPU memory | Pipeline bubble (idle time) |

## Data Parallelism

The simplest distributed strategy. Each of $N$ GPUs holds a full copy of the model and processes $1/N$ of the mini-batch:

1. **Forward**: each GPU computes loss on its local micro-batch independently
2. **All-reduce**: average gradients across all GPUs
3. **Step**: each GPU applies the identical update, keeping replicas in sync

This is easy to implement (`torch.nn.parallel.DistributedDataParallel`) and scales to thousands of GPUs. The only communication is the gradient all-reduce — one pass over the model parameters per step.

**The problem**: every GPU stores the full model weights, full optimizer states, and full gradients. For a 70B model with Adam in mixed precision:

| Component | Per-GPU memory |
|---|---|
| Weights (BF16) | 140 GB |
| Adam moments (2 × FP32) | 560 GB |
| Gradients (BF16) | 140 GB |
| **Total** | **~840 GB** |

This doesn't fit on any single GPU. The insight behind [FSDP/ZeRO]({{< relref "15-distributed-training" >}}) is that you can shard these redundant copies across the DP group — but that's a story for post 15.

For now, the takeaway: DP alone can't train large models. We need to split the model itself.

## Tensor Parallelism

Tensor parallelism (TP) partitions the weight matrices of each layer across $T$ GPUs. Each GPU stores $1/T$ of the weights and computes $1/T$ of each matmul. The key building blocks are two ways to split a linear layer.

### Column-parallel linear

Partition the weight matrix $W \in \mathbb{R}^{d \times k}$ along columns into $T$ shards:

$$
W = [W_1 \mid W_2 \mid \cdots \mid W_T], \quad W_i \in \mathbb{R}^{d \times k/T}
$$

Each GPU $i$ holds $W_i$ and computes $Y_i = X W_i \in \mathbb{R}^{b \times k/T}$. The full output is the concatenation $Y = [Y_1 \mid Y_2 \mid \cdots \mid Y_T]$ — but we don't need to gather it yet if the next operation can consume the sharded output directly.

```
Input X ─────────┬──────────┬──────────┐
(b, d)           │          │          │
              [X·W₁]    [X·W₂]    [X·W₃]     ← each GPU
              (b,k/T)   (b,k/T)   (b,k/T)
```

### Row-parallel linear

Partition the weight matrix along rows:

$$
W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_T \end{bmatrix}, \quad W_i \in \mathbb{R}^{k/T \times d}
$$

Each GPU $i$ holds $W_i$ and takes the corresponding shard of the input: $Y_i = X_i W_i \in \mathbb{R}^{b \times d}$. The full output is the **sum** $Y = \sum_i Y_i$ — an all-reduce.

```
Input X₁ ──→ [X₁·W₁] ─┐
(b, k/T)     (b, d)    │
                        ├──→ all-reduce ──→ Y (b, d)
Input X₂ ──→ [X₂·W₂] ─┘
(b, k/T)     (b, d)
```

### The Megatron trick

The key insight from Megatron-LM [[1]](#ref-1): pair column-parallel with row-parallel in sequence, so the column-parallel output (sharded along $k$) feeds directly into the row-parallel input (which expects shards along $k$). No communication between them — only one all-reduce at the end.

For the MLP (SwiGLU variant):

```
         column-parallel           row-parallel
X ──→ [SiLU(X·Wᵍᵃᵗᵉᵢ) ⊙ X·Wᵘᵖᵢ] ──→ [·Wᵈᵒʷⁿᵢ] ──→ all-reduce ──→ Y
         each GPU has 1/T              each GPU has 1/T
         of gate & up columns          of down rows
```

Each GPU computes the full gated activation for its shard of the hidden dimension, then the row-parallel down-projection produces partial sums that the all-reduce combines. **One all-reduce per MLP block.**

### Applying to attention

Multi-head attention is naturally parallelizable — the heads are independent. With $h$ heads and $T$ GPUs, assign $h/T$ heads per GPU:

- **QKV projections**: column-parallel — each GPU computes Q, K, V for its $h/T$ heads
- **Attention**: each GPU runs attention on its local heads (no communication)
- **Output projection $W^O$**: row-parallel — each GPU's head outputs are partial contributions to the final result, combined by all-reduce

**One all-reduce per attention block.**

### Communication per layer

A full Transformer layer with TP has **two all-reduces**: one in attention, one in MLP. Each all-reduce moves $O(b \cdot n \cdot d)$ data across the TP group, where $b$ = batch size, $n$ = sequence length, $d$ = model dimension.

```
                     TP Region                              TP Region
                  ┌─────────────┐                       ┌─────────────┐
  LayerNorm ──→   │  Attention   │ ──all-reduce──→  LN ──→ │     MLP     │ ──all-reduce──→  + residual
                  │  (h/T heads) │                       │ (dff/T hidden)|
                  └─────────────┘                       └─────────────┘
```

TP requires fast interconnect (NVLink, ~900 GB/s per GPU on H100) because communication happens twice per layer, every layer, every micro-batch. This is why TP is typically used **within a single node** (8 GPUs connected by NVLink) and not across nodes.

## Sequence Parallelism

### The redundancy problem

Look at the diagram above: LayerNorm, dropout, and residual additions sit **between** the TP regions. In standard TP, these ops are replicated on every GPU — each GPU independently runs LayerNorm on the full $(b, n, d)$ activation tensor, producing identical results.

This is wasted memory. With TP degree $T = 8$, you have 8 copies of these activations doing the same thing.

### The fix

Sequence parallelism [[2]](#ref-2) replaces the all-reduce with its two constituent operations:

- **Reduce-scatter**: after a TP region, instead of giving every GPU the full result, give each GPU $1/T$ of the sequence — activations are now partitioned along the sequence dimension
- **All-gather**: before the next TP region, reconstruct the full sequence from the partitions

Between reduce-scatter and all-gather, each GPU holds only $(b, n/T, d)$ activations for LayerNorm, dropout, and residual add. These are element-wise ops, so splitting along the sequence dimension is trivial.

```
TP region              SP region (n/T per GPU)           TP region
[Attention] ──reduce-scatter──→ [LN + dropout + residual] ──all-gather──→ [MLP]
```

### Why it's free

All-reduce is literally reduce-scatter followed by all-gather. Standard TP fuses them into one all-reduce; SP just keeps the data distributed in between. **Same bytes transferred, same number of messages.** The only difference is that activations between TP regions are $T\times$ smaller.

| | Without SP | With SP |
|---|---|---|
| Non-TP activation memory per GPU | $(b, n, d)$ | $(b, n/T, d)$ |
| Communication per TP boundary | 1 all-reduce | 1 reduce-scatter + 1 all-gather |
| Total bytes transferred | Same | Same |

SP inherits its GPU group and degree $T$ from TP — it's not an independent parallelism dimension, just a memory optimization on the TP setup. In Megatron-LM, it's a single flag: `--sequence-parallel`.

This same "decompose all-reduce and stay sharded" pattern appears in [FSDP/ZeRO]({{< relref "15-distributed-training" >}}), applied to parameters and gradients instead of activations.

## Pipeline Parallelism

TP splits within a layer but requires fast interconnect. Pipeline parallelism (PP) takes the other axis: split **across layers**, assigning different layers to different GPUs. Communication is only the activations passed between adjacent stages — much less frequent than TP's per-layer all-reduces.

### Layer assignment

With $L$ layers and $S$ pipeline stages, assign $L/S$ consecutive layers per stage:

```
Stage 0 (GPU 0): layers 0–7
Stage 1 (GPU 1): layers 8–15
Stage 2 (GPU 2): layers 16–23
Stage 3 (GPU 3): layers 24–31
```

Each stage communicates only with its neighbors — sending activations forward and gradients backward. The communication volume per boundary is just one activation tensor $(b, n, d)$, and it only happens once per stage transition.

### The bubble problem

Naive execution is sequential — only one stage is active at a time:

```
Time ──→
GPU 0: [  F  ][     ][     ][     ][     ][     ][  B  ][     ]
GPU 1: [     ][  F  ][     ][     ][     ][  B  ][     ][     ]
GPU 2: [     ][     ][  F  ][     ][  B  ][     ][     ][     ]
GPU 3: [     ][     ][     ][  F  ][  B  ][     ][     ][     ]
                              ↑ bubble = (S-1)/S of total time
```

With $S = 4$ stages, 75% of GPU time is idle. Unacceptable.

### Micro-batching (GPipe)

GPipe [[3]](#ref-3) splits the mini-batch into $M$ micro-batches and pipelines them:

```
Time ──→
GPU 0: [F₁][F₂][F₃][F₄][        ][B₄][B₃][B₂][B₁]
GPU 1: [  ][F₁][F₂][F₃][F₄][  B₄][B₃][B₂][B₁][  ]
GPU 2: [  ][  ][F₁][F₂][F₃][F₄B₄][B₃][B₂][B₁][  ]
GPU 3: [  ][  ][  ][F₁][F₂][F₃F₄ ][B₃][B₂][B₁][  ]
```

The bubble shrinks to $(S-1)/(S-1+M)$ of total time. With $M = 32$ micro-batches and $S = 4$ stages, the bubble is only ~9%.

**The cost**: GPipe must store activations for all $M$ micro-batches simultaneously (needed for backward), so peak memory scales as $O(M)$.

### 1F1B schedule

The one-forward-one-backward (1F1B) schedule [[4]](#ref-4) interleaves forward and backward passes of different micro-batches:

```
Time ──→
GPU 0: [F₁][F₂][F₃][F₄][B₁][F₅][B₂][F₆][B₃]···[B_M]
GPU 1: [  ][F₁][F₂][F₃][F₄][B₁][F₅][B₂][F₆]···
GPU 2: [  ][  ][F₁][F₂][F₃][F₄][B₁][F₅][B₂]···
GPU 3: [  ][  ][  ][F₁][F₂][F₃][F₄][B₁][F₅]···
```

After the warmup phase (filling the pipeline), each GPU alternates: one forward, one backward. This means each GPU only stores activations for at most $S$ in-flight micro-batches at a time, reducing peak memory from $O(M)$ to $O(S)$. The bubble ratio is the same as GPipe, but the memory savings are significant — $S$ is typically 4–8, while $M$ can be 32+.

### Interleaved schedule

Megatron-LM [[5]](#ref-5) assigns **non-contiguous** layers to each stage. With $V$ virtual stages per GPU:

```
V = 2:
Stage 0 (GPU 0): layers {0–3, 16–19}
Stage 1 (GPU 1): layers {4–7, 20–23}
Stage 2 (GPU 2): layers {8–11, 24–27}
Stage 3 (GPU 3): layers {12–15, 28–31}
```

A micro-batch visits each GPU $V$ times instead of once, making the pipeline deeper with shorter stages. The bubble shrinks by a factor of $V$:

$$
\text{bubble ratio} = \frac{S - 1}{V \cdot M + S - 1}
$$

The trade-off: $V\times$ more point-to-point communications (activations now bounce between GPUs multiple times), so this requires fast inter-node bandwidth.

### Zero-bubble pipeline parallelism

Zero-bubble PP [[6]](#ref-6) exploits the fact that the backward pass has two independent parts:

- **B** (activation gradient): $\partial L / \partial X = \partial L / \partial Y \cdot W^T$ — needed by the previous stage, on the critical path
- **W** (weight gradient): $\partial L / \partial W = X^T \cdot \partial L / \partial Y$ — only needed locally for the optimizer step, not time-critical

By computing B first (to unblock the previous stage) and deferring W to fill the bubble:

```
Time ──→
GPU 0: [F₁][F₂][F₃][F₄][B₁][F₅][B₂]···[B_M][W₁][W₂]···[W_M]
                                                  ↑ fills the bubble
```

The W computations slot into what would otherwise be idle time, approaching zero bubble overhead.

## 3D Parallelism

In practice, large-scale training combines all three dimensions:

| Dimension | Splits | Communication | Typical placement |
|---|---|---|---|
| TP ($T$) | Weights within a layer | 2 all-reduces/layer | Within a node (NVLink) |
| PP ($S$) | Layers across stages | Point-to-point activations between stages | Across nodes |
| DP ($N$) | Data across replicas | All-reduce gradients once per step | Across replicas |

For a cluster of 512 GPUs training a 175B model:

```
TP = 8  (within each 8-GPU node, NVLink)
PP = 8  (8 stages across 8 nodes)
DP = 8  (8 replicas, each spanning 64 GPUs)
Total: 8 × 8 × 8 = 512 GPUs
```

Each GPU holds $1/8$ of a layer's weights (TP) for $1/8$ of all layers (PP), and processes $1/8$ of the data (DP). The key design principle: **use fast interconnect for frequent communication (TP), slower interconnect for infrequent communication (PP, DP).**

[Expert parallelism]({{< relref "12-expert-parallelism" >}}) and [context parallelism]({{< relref "13-context-ring-parallelism" >}}) add further dimensions for MoE models and long-context training respectively.

## Trade-offs

| | DP | TP | PP |
|---|---|---|---|
| **Memory savings** | None (full model per GPU) | $T\times$ less weights per GPU | $S\times$ less layers per GPU |
| **Communication** | 1 all-reduce/step (gradients) | 2 all-reduces/layer (activations) | Point-to-point between stages |
| **Bandwidth need** | Moderate | Very high (NVLink) | Low–moderate |
| **Idle time** | None | None | Pipeline bubble |
| **Complexity** | Low | Medium | High (scheduling) |

**When to use what:**

- **DP only**: model fits on one GPU — just scale throughput
- **TP + DP**: model doesn't fit on one GPU but fits on one node (TP = 8) — most common setup for 7B–70B models
- **TP + PP + DP**: model doesn't fit on one node — add PP stages across nodes, required for 100B+ models
- **+ SP**: always enable with TP — it's free memory savings

## References

<span id="ref-1">[1]</span> Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). *arXiv preprint*.

<span id="ref-2">[2]</span> Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., & Catanzaro, B. (2022). [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198). *MLSys 2023*.

<span id="ref-3">[3]</span> Huang, Y., Cheng, Y., Bapna, A., Firat, O., Chen, D., Chen, M., ... & Wu, Y. (2019). [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965). *NeurIPS 2019*.

<span id="ref-4">[4]</span> Narayanan, D., Harlap, A., Phanishayee, A., Seshadri, V., Devanur, N. R., Ganger, G. R., ... & Zaharia, M. (2019). [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377). *SOSP 2019*.

<span id="ref-5">[5]</span> Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., ... & Catanzaro, B. (2021). [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473). *SC 2021*.

<span id="ref-6">[6]</span> Qi, Z., Wan, X., Huang, Y., & Yang, Y. (2023). [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241). *ICLR 2024*.

*Last updated: April 2026*
