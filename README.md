# All Attentions You Need

A blog series covering attention mechanisms in modern AI — from math to metal.

Read the series at: https://carlyou.github.io/all-attentions-you-need/

## Local Development

### Prerequisites
- [Hugo Extended](https://gohugo.io/installation/) (v0.146+)
- Git

### Setup

```bash
git clone https://github.com/carlyou/all-attentions-you-need.git
cd all-attentions-you-need
git submodule update --init --recursive
hugo server -D
```

Site will be available at `http://localhost:1313/all-attentions-you-need/`.

### Creating a New Post

```bash
hugo new posts/02-mqa-gqa.md
```

Uses the archetype template. Every post has these fixed sections:
TL;DR → Motivation → ... → Trade-offs → References

The middle sections depend on the topic:
- **Mechanism posts** (attention variants, MoE): Mechanism → Training → Inference
- **Systems posts** (FlashAttention, parallelism, serving): Design → Implementation

### Tech Stack

- **Hugo** — static site generator
- **SANS** — minimal theme with dark mode
- **MathJax** — LaTeX math rendering
- **GitHub Pages** — hosting via GitHub Actions

## License

MIT
