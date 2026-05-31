# Contributing to OpenLawAI

Thank you for your interest in contributing to OpenLawAI!

## How to Contribute

1. **Fork** the repository
2. **Create a branch** for your feature or fix
3. **Make your changes** with clear, descriptive commits
4. **Test** your changes locally
5. **Open a Pull Request** with a clear description of what you've done

## Development Setup

See [README.md](README.md) for setup instructions.

## Areas of Contribution

- **New jurisdictions** — Add extractors for other legal systems (UK legislation.gov.uk, US Code, EU EUR-Lex, etc.)
- **Retrieval quality** — Improve search, chunking, embedding, or reranking
- **GPU-free operation** — The system currently requires a local NVIDIA GPU for embedding and reranking. Contributions that enable running without a personal GPU are very welcome — for example, integrating hosted vector stores (OpenAI, Pinecone, etc.), cloud embedding APIs, or alternative lightweight models that run on CPU
- **Frontend** — UI/UX improvements, accessibility, mobile experience
- **Documentation** — Setup guides, API docs, translations
- **Testing** — Unit tests, integration tests, end-to-end tests
- **Infrastructure** — Docker improvements, CI/CD, deployment guides

## Contributor License Agreement (CLA)

By contributing to OpenLawAI, you agree that your contributions may be relicensed under alternative terms by the project maintainers. This enables dual-licensing flexibility while keeping the project open source under AGPL-3.0.

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

## Code of Conduct

Be respectful, constructive, and inclusive. We're building this together.
