# All Attentions You Need

A blog series exploring attention mechanisms in modern AI.

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

Uses the archetype template with the standard section structure:
TL;DR → Motivation → Mechanism → Training → Inference → Trade-offs → Adoption → References

### Tech Stack

- **Hugo** — static site generator
- **SANS** — minimal theme with dark mode
- **MathJax** — LaTeX math rendering
- **GitHub Pages** — hosting via GitHub Actions

## License

MIT
