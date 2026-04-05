# All Attentions You Need

A Hugo blog series covering attention mechanisms in transformers — from foundational Multi-Head Attention through modern optimizations, parallelism strategies, and inference techniques.

## Project Structure

- `content/posts/XX-topic/index.md` — Post content (Hugo markdown with frontmatter)
- `content/posts/XX-topic/*.svg` — Diagrams (Excalidraw-exported SVGs)
- Excalidraw live diagrams at `excalidraw.com` for collaborative editing

## Blog Series Organization

- Posts 00-02: Foundations (MHA, Transformer architecture)
- Posts 03-05: Attention variants (MQA/GQA, MLA, MoE)
- Posts 06-10: Efficient attention (Sparse, Linear, FlashAttention)
- Posts 11-15: Parallelism & distributed training
- Posts 16-20: Inference optimizations (Paged attention, batching, speculative decoding)
- Appendices: GPU architecture, position encodings, quantization

## Team Workflow

This project uses a 4-role team for both creating and reviewing content. Each role can be invoked as a slash command:

| Role | Command | Creates | Reviews |
|------|---------|---------|---------|
| Editorial Manager | `/editorial` | Post structure, narrative arc, transitions, openings | Flow, readability, publishing readiness |
| Researcher | `/researcher` | Mechanism explanations, equations, citations, context | Theoretical accuracy, math rigor |
| Engineer | `/engineer` | Code examples, performance analysis, practical guidance | Implementation correctness, claims |
| Designer | `/designer` | Excalidraw diagrams, visual layouts, SVG exports | Diagram clarity, style consistency |

- Invoke a role to **create**: `/researcher` "draft the mechanism section for post 04-mla"
- Invoke a role to **review**: `/engineer` "review post 09-flash-attention"
- Run all four reviews: `/team-review` on any post

## Writing Conventions

- Use `$$..$$` for display math blocks, `$..$` for inline math
- Diagrams: neutral gray palette, monospace labels, dimension annotations
- Code examples: PyTorch preferred, with shape comments
- Each post should be self-contained but link to related posts
- Keep dimension annotations consistent: n=seq_len, d=d_model, d_k=d_model/h
