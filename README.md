# OpenLawAI

You're welcome to try it out and contribute. See [CONTRIBUTING.md](CONTRIBUTING.md).

An open-source AI-powered legal assistant that helps users find relevant laws and regulations, analyze legal documents, and generate drafts of legal texts.

**Norwegian law is the first supported jurisdiction**, with data sourced from [Lovdata's free public datasets](https://api.lovdata.no/om-api-tjenesten/). The system is designed to be extended to other legal systems — contributions are very welcome.

## Features

- **Legal document search** — semantic and keyword search across laws, regulations, and legal documents
- **AI-powered chat** — conversational interface with retrieval-augmented generation (RAG)
- **Document analysis** — upload PDFs, Word documents, or images for AI-assisted analysis
- **Document generation** — generate drafts of contracts and other legal documents
- **Multi-model routing** — configurable quality modes (fast/thorough) using different LLM tiers
- **Token usage telemetry** — track operational token usage for monitoring

## Architecture

| Component | Technology |
|-----------|-----------|
| Backend | Django + Django REST Framework (async) |
| Frontend | React (Vite) |
| Vector DB | Weaviate |
| LLM | OpenAI API (configurable models) |
| Embedding | vLLM (Qwen3-Embedding-4B) — requires NVIDIA GPU |
| Reranking | vLLM (Qwen3-Reranker-4B) — requires NVIDIA GPU |
| OCR | Marker |
| Streaming | Server-Sent Events via Redis pub/sub |
| Database | SQLite (zero config) |
| Cache/Locks | Redis |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- NVIDIA GPU with CUDA (for embedding/reranking via vLLM)
- Docker & Docker Compose (for infrastructure services)

### 1. Start infrastructure services

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY

docker compose up
```

This starts Weaviate, Redis, vLLM (embed + rerank), and Marker OCR.

> **Note:** On the first run, vLLM downloads and loads the embedding/reranking models (~4B parameters each). This can take 10–20 minutes depending on your connection and GPU. The health check may time out before the models are ready — if you see `container openlawai-embed-1 is unhealthy`, run `docker compose down && docker compose up` again. Subsequent starts use cached weights and are much faster.

### 2. Install Python dependencies

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

### 3. Run migrations and ingest legal data

```bash
uv run python manage.py migrate
```

Then ingest legal data for your jurisdiction. For **Norwegian law** (the currently supported dataset):

```bash
./scripts/download_norwegian_law.sh
uv run python manage.py ingest_legal ./data/norwegian-law/
```

Other jurisdictions can be added by implementing a new extractor — see [Adding Support for New Jurisdictions](#adding-support-for-new-jurisdictions) below.

### 4. Build the frontend

```bash
cd frontend && npm install && npm run build && cd ..
uv run manage.py collectstatic --noinput
```

### 5. Run the server

Using uvicorn:

```bash
uv run uvicorn config.asgi:application --host 0.0.0.0 --port 8000
```

Open http://localhost:8000, register an account, and start chatting.

## Adding Support for New Jurisdictions

The ingestion pipeline uses a pluggable extractor protocol. To add a new legal data source:

1. Create a new extractor in `legal_pipeline/extractors/` implementing the `DocumentExtractor` protocol
2. Add a download script in `scripts/` for acquiring the source data
3. Register the extractor in the ingest management command

See `legal_pipeline/extractors/base.py` for the protocol definition and `legal_pipeline/extractors/lovdata.py` for a reference implementation.

## Help Wanted

We'd especially love contributions in these areas:

- **More jurisdictions** — Add extractors for UK (legislation.gov.uk), US (US Code), EU (EUR-Lex), or any other legal system
- **GPU-free operation** — Make the system runnable without a personal NVIDIA GPU (hosted embeddings, cloud vector stores, CPU-friendly models)
- **Retrieval quality** — Improve search, chunking, embedding, or reranking
- **Frontend & UX** — UI improvements, accessibility, mobile experience
- **Infrastructure** — Docker improvements, deployment guides, CI/CD

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

If you modify OpenLawAI and provide it as a service, you must make your modified source code available under the same license.

For commercial licensing inquiries (e.g., proprietary SaaS deployments), please contact the maintainers.
