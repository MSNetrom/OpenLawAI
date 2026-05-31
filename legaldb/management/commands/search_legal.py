from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from legal_pipeline.chunker import EmbeddingService
from legal_pipeline.reranker import RerankerClient
from legal_pipeline.retriever import (
    DjangoChunkRepository,
    DocumentRetriever,
    DocumentScorer,
)
from legal_pipeline.weaviate_client import LegalChunkStore


class Command(BaseCommand):
    help = "Search documents using Weaviate + Django metadata reranking."

    def add_arguments(self, parser):
        parser.add_argument("query", nargs="+", help="Natural language query.")
        parser.add_argument("--documents", type=int, default=5, help="Number of documents to return.")
        parser.add_argument("--chunks", type=int, default=64, help="Chunk hits to fetch from Weaviate.")
        parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid alpha for Weaviate search.")
        parser.add_argument(
            "--no-embeddings",
            action="store_true",
            help="Skip query embeddings (text-only hybrid).",
        )
        parser.add_argument(
            "--no-reranker",
            action="store_true",
            help="Disable the reranker.",
        )

    def handle(self, *args, **options):
        query = " ".join(options["query"]).strip()
        if not query:
            raise CommandError("Query must not be empty.")
        weaviate_client = LegalChunkStore()
        chunk_repo = DjangoChunkRepository()
        scorer = DocumentScorer()
        embedding_service = None
        if not options["no_embeddings"]:
            try:
                embedding_service = EmbeddingService()
            except Exception as exc:
                raise CommandError(f"vLLM embeddings unavailable: {exc}") from exc
        reranker = None
        if not options["no_reranker"]:
            try:
                reranker = RerankerClient()
            except Exception as exc:
                raise CommandError(f"Failed to load reranker: {exc}") from exc
        retriever = DocumentRetriever(
            weaviate_client=weaviate_client,
            chunk_repository=chunk_repo,
            embedding_service=embedding_service,
            scorer=scorer,
            reranker=reranker,
            enable_reranker=not options["no_reranker"],
        )
        results = retriever.retrieve(
            lexical_query=query,
            semantic_query=query,  # Use same query for CLI
            chunk_limit=options["chunks"],
            document_limit=options["documents"],
            alpha=options["alpha"],
        )
        if not results:
            self.stdout.write(self.style.WARNING("No documents found."))
            return
        for idx, result in enumerate(results, 1):
            self.stdout.write(
                self.style.SUCCESS(
                    f"{idx}. {result.work_ref_id} ({result.document_type}) score={result.score:.3f} chunks={result.chunk_count}"
                )
            )
            self.stdout.write(f"   Title: {result.title}")
            if result.version_label:
                self.stdout.write(f"   Version: {result.version_label}")
            for chunk in result.chunks[:3]:
                preview = chunk.text.strip().replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:197] + "..."
                self.stdout.write(f"      - [{chunk.score:.3f}] {preview}")
