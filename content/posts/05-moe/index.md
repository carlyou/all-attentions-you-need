---
title: "Mixture of Experts (MoE)"
date: 2026-03-26
draft: true
math: true
toc: true
tags: ["moe", "deepseek", "sparse", "ffn", "routing"]
description: "Mixture of Experts replaces the dense FFN with many specialized expert networks, routing each token to only a few — dramatically increasing model capacity without proportional compute cost."
weight: 5
---

## TL;DR

Mixture of Experts (MoE) replaces the Transformer's dense FFN with hundreds of smaller "expert" FFNs and a learned router that selects which experts each token uses. This decouples model capacity (total parameters) from per-token compute — a 256-expert model stores 256× more knowledge but each token only activates 8 experts. The cost: irregular computation patterns, load imbalancing, and all-to-all communication when experts are distributed across GPUs.

## Motivation

In a standard Transformer, the [FFN]({{< relref "02-the-rest-of-the-transformer" >}}) is the parameter-heavy component — typically 2/3 of total parameters. To make a model "smarter," you make the FFN larger: more hidden units, more parameters, more compute per token. This scales linearly: $2\times$ parameters = $2\times$ compute.

But not all knowledge is relevant to every token. A token about French cuisine doesn't need the parameters that encode quantum physics. What if the model could dynamically select which parameters to use?

The idea dates back to Jacobs et al. [[1]](#ref-1) in 1991. In the Transformer era, Shazeer et al. [[2]](#ref-2) revived it with the Sparsely-Gated Mixture-of-Experts layer, and Fedus et al. [[3]](#ref-3) scaled it further with the Switch Transformer. DeepSeek-V2/V3 [[4]](#ref-4) pushed it to 256 fine-grained experts with shared experts.

## Mechanism

### The standard dense FFN

The baseline Transformer FFN (gated variant, a.k.a. SwiGLU):

$$
\text{FFN}(\mathbf{x}) = W^{\text{down}} \left( \text{SiLU}(W^{\text{gate}} \mathbf{x}) \odot W^{\text{up}} \mathbf{x} \right)
$$

where $W^{\text{gate}}, W^{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ and $W^{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$. The gate branch controls how much of the value branch passes through, element-wise.

Every token uses the same $W^{\text{gate}}$, $W^{\text{up}}$, $W^{\text{down}}$.

### MoE: many experts, sparse activation

Replace the single FFN with $E$ expert FFNs, each with its own weights:

$$
\text{Expert}_i(\mathbf{x}) = W^{\text{down}}_i \left( \text{SiLU}(W^{\text{gate}}_i \mathbf{x}) \odot W^{\text{up}}_i \mathbf{x} \right)
$$

A **router** (also called a gate) selects which experts each token uses:

$$
\mathbf{r} = \mathbf{x} W^{\text{router}} \quad \in \mathbb{R}^{E}
$$

$$
\text{selected} = \text{top-}k(\mathbf{r})
$$

$$
\mathbf{w} = \text{softmax}(\mathbf{r}[\text{selected}]) \quad \in \mathbb{R}^{k}
$$

The MoE output is a weighted sum of the selected experts:

$$
\text{MoE}(\mathbf{x}) = \sum_{i \in \text{selected}} w_i \cdot \text{Expert}_i(\mathbf{x})
$$

The router is a simple learned linear projection — a single matrix multiply produces scores for all experts, from which the top-$k$ are selected.

### Shared experts

DeepSeek-V2 [[4]](#ref-4) adds **shared experts** that process every token unconditionally:

$$
\text{MoE}(\mathbf{x}) = \text{SharedExpert}(\mathbf{x}) + \sum_{i \in \text{selected}} w_i \cdot \text{Expert}_i(\mathbf{x})
$$

The shared expert captures common patterns that every token needs (syntactic structure, common vocabulary), while routed experts specialize in domain-specific knowledge. This stabilizes training and improves quality.

### Concrete numbers (DeepSeek-V3)

| Parameter | Value |
|---|---|
| Routed experts ($E$) | 256 |
| Experts per token ($k$) | 8 |
| Shared experts | 1 |
| Expert FFN hidden dim ($d_{\text{ff}}$) | 2,048 |
| Hidden dim ($d$) | 7,168 |

Total expert parameters: $256 \times 3 \times 7168 \times 2048 \approx 11.3\text{B}$ per MoE layer. But per-token compute: only $8 \times 3 \times 7168 \times 2048$ — 32× less than the total.

### The permute-compute-unpermute pattern

Different tokens route to different experts, creating an irregular computation pattern. The standard implementation:

**Step 1: Permute** — group tokens by expert assignment.

```
Input:  [tok0, tok1, tok2, tok3, tok4, tok5]
Routes: [E2,   E0,   E2,   E1,   E0,   E2]

Permuted:
  Expert 0 batch: [tok1, tok4]
  Expert 1 batch: [tok3]
  Expert 2 batch: [tok0, tok2, tok5]
```

**Step 2: Compute** — run each expert's FFN on its token batch as a batched GEMM.

**Step 3: Unpermute** — scatter results back to original token order and apply router weights.

The permute/unpermute is overhead, but it enables efficient batched matrix multiplies per expert. Without it, you'd process one token at a time per expert — far too slow.

### Fused MoE kernels

Running 256 separate small GEMMs (one per expert) is inefficient — too many kernel launches with small batch sizes. Fused MoE kernels handle the entire permute-compute-unpermute in a single kernel launch:

- **Triton fused MoE**: vLLM's default, handles routing + expert dispatch + weighted sum
- **CUTLASS grouped GEMM**: batches all expert GEMMs into one kernel call

The kernel receives the full token batch plus routing indices and internally handles the expert dispatch.

## Training

### Load balancing

The router can collapse — sending all tokens to a few popular experts while others go unused. This wastes capacity and creates compute imbalance.

**Auxiliary loss**: an extra loss term penalizes uneven expert utilization:

$$
\mathcal{L}_{\text{aux}} = \alpha \sum_{i=1}^{E} f_i \cdot p_i
$$

where $f_i$ is the fraction of tokens routed to expert $i$, and $p_i$ is the average router probability for expert $i$. Minimizing this product encourages uniform distribution.

### Training stability

MoE models are harder to train than dense models:

- Router gradients are sparse (only selected experts receive gradients per token)
- Expert utilization can oscillate during training
- The auxiliary loss weight $\alpha$ requires careful tuning — too high forces uniform routing regardless of token content, too low allows collapse

### Expert specialization

Despite being initialized identically, experts naturally specialize during training. Analysis of trained MoE models shows:

- Some experts specialize in languages, domains, or syntactic roles
- Specialization is soft — most experts handle a range of inputs
- The shared expert acts as a generalist backbone

## Inference

### Memory

All expert weights must reside in GPU memory, even though each token uses only $k$ of $E$:

$$
\text{MoE layer weights} = E \times 3 \times d \times d_{\text{ff}} \times \text{sizeof(dtype)}
$$

For DeepSeek-V3 in FP8: $256 \times 3 \times 7168 \times 2048 \times 1 \approx 11.3\text{ GB}$ per MoE layer. This is why large MoE models typically require multi-GPU serving.

### Compute efficiency

During decode, each token activates $k$ experts. With a batch of $B$ tokens:

- Best case (all tokens route to the same experts): $k$ large GEMMs with batch size $B$
- Worst case (all tokens route to different experts): up to $B \cdot k$ tiny GEMMs

Real workloads fall somewhere between. The load imbalance is the main efficiency challenge — some experts get many tokens (good utilization), others get few or none (wasted GPU cycles).

### Expert parallelism

With $E = 256$ experts and 8 GPUs, split experts across GPUs:

```
GPU 0: experts 0–31
GPU 1: experts 32–63
...
GPU 7: experts 224–255
```

Each GPU stores 1/8 of the expert weights. But tokens routed to remote experts need to be sent there:

1. **All-to-all**: each GPU sends tokens to the GPU that owns their expert
2. **Compute**: each GPU runs its local experts
3. **All-to-all**: send results back to originating GPU

The two all-to-all communications are the main overhead. They require fast interconnect (NVLink) and careful overlap with computation to hide latency.

### Interaction with tensor parallelism

Expert parallelism (EP) and tensor parallelism (TP) are orthogonal:

- **TP**: splits each expert's weight matrices across GPUs (all-reduce per expert)
- **EP**: puts different experts on different GPUs (all-to-all between experts)

A typical deployment combines both: TP within a node for the attention layers and shared experts, EP across the full GPU pool for routed experts.

## Trade-offs

| | Parameters | Compute per token | Memory | Communication |
|---|---|---|---|---|
| **Dense FFN** | $3 d \cdot d_{\text{ff}}$ | $3 d \cdot d_{\text{ff}}$ | Low | None (TP all-reduce) |
| **MoE** ($E$ experts, top-$k$) | $E \times 3 d \cdot d_{\text{ff}}$ | $k \times 3 d \cdot d_{\text{ff}}$ | High (all experts) | All-to-all (EP) |

**Strengths**:
- Massive capacity increase without proportional compute: 256 experts with top-8 gives 32× more parameters for only 8× compute
- Natural fit for expert parallelism — different experts on different GPUs
- Shared experts provide a stable baseline while routed experts specialize

**Weaknesses**:
- All expert weights must be in GPU memory — high memory cost even though most are idle per token
- Irregular computation: load imbalance across experts reduces GPU utilization
- All-to-all communication for expert parallelism requires fast interconnect
- Training is harder: auxiliary loss tuning, risk of expert collapse

**MoE vs wider dense FFN**: for the same compute budget, a MoE model with many small experts typically outperforms a dense model with one large FFN, because the conditional computation lets the model allocate capacity where it's needed.

## Adoption

MoE has been adopted across multiple model families:

- **Switch Transformer** (Google, 2021) [[3]](#ref-3): popularized sparse MoE for language models
- **Mixtral 8x7B / 8x22B** (Mistral, 2023): made MoE mainstream in open models, 8 experts with top-2 routing
- **DeepSeek-V2/V3** (2024): 256 fine-grained experts with top-8 routing and shared experts
- **Qwen 2 MoE**: 60 experts with top-6 routing
- **DBRX** (Databricks): 16 experts with top-4 routing
- **Grok-1** (xAI): 8 experts with top-2 routing

The trend is toward more fine-grained experts (many small experts rather than few large ones), following DeepSeek's approach. Shared experts have also gained traction as a way to stabilize training.

## References

<span id="ref-1">[1]</span> Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). [Adaptive Mixtures of Local Experts](https://doi.org/10.1162/neco.1991.3.1.79). *Neural Computation, 3*(1), 79–87.

<span id="ref-2">[2]</span> Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538). *ICLR 2017*.

<span id="ref-3">[3]</span> Fedus, W., Zoph, B., & Shazeer, N. (2022). [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961). *JMLR, 23*(120), 1–39.

<span id="ref-4">[4]</span> DeepSeek-AI. (2024). [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434). *arXiv preprint*.

*Last updated: March 2026*
