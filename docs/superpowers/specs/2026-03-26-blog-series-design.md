# All Attentions You Need — Series Design

## Overview

A blog series covering attention mechanisms in modern AI, targeting readers at all levels: learners (Mechanism section), practitioners (Training), and systems engineers (Inference). Each post follows a consistent template. The series is ordered by narrative arc, not chronology.

## Per-Post Template

| # | Section | Purpose |
|---|---------|---------|
| 1 | TL;DR | 2-3 sentence summary |
| 2 | Motivation | What problem does this solve? What came before? |
| 3 | Mechanism | Architecture, equations, diagram |
| 4 | Training | Memory, compute, stability |
| 5 | Inference | KV cache, kernel support, TP, quantization |
| 6 | Trade-offs | Comparison table vs relevant alternatives |
| 7 | Adoption | Models and frameworks using it today |
| 8 | References | Papers with [1]-style inline citations |

## Topic Roadmap

**Prologue**
- 0: Prologue — The Attention Landscape (taxonomy, overview, reading guide)

**Part 1: The Foundations**
- 1: Multi-Head Attention (MHA)
- 2: Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)

**Part 2: The KV Cache Problem**
- 3: Multi-head Latent Attention (MLA)
- 4: PagedAttention

**Part 3: The Quadratic Problem**
- 5: Sliding Window Attention
- 6: FlashAttention (v1/v2/v3)
- 7: Sparse Attention (Longformer, BigBird, DeepSeek NSA/DSA)

**Part 4: Beyond Softmax**
- 8: Linear Attention & Gated DeltaNet (GDN)
- 9: Differential Attention
- 10: Residual Attention (Moonshot AI)

**Part 5: Scaling Out**
- 11: Ring Attention

**Appendix**
- A: Position Encodings (RoPE, ALiBi, NoPE)

## Conventions

- Math via MathJax ($ inline, $$ display)
- SVG diagrams co-located as page bundles
- Academic-style references with [n] inline links
- Each post tagged by mechanism name
- Hugo theme: SANS
- Prologue contains the roadmap (not README)
- README is dev/contributor docs only
