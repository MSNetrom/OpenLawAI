from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from legal_pipeline.chunker import LangChainSectionChunker, EmbeddingService
from legal_pipeline.extractors.lovdata import LovdataExtractor
from legal_pipeline.ingestor import DjangoMetadataRepository, IngestionPipeline
from legal_pipeline.weaviate_client import LegalChunkStore


class Command(BaseCommand):
    help = "Ingest Lovdata XML files into Django/Postgres and Weaviate."

    def add_arguments(self, parser):
        parser.add_argument("paths", nargs="+", help="Files or directories containing Lovdata XML.")
        parser.add_argument("--chunk-size", type=int, default=1200, help="Maximum characters per chunk.")
        parser.add_argument("--overlap", type=int, default=200, help="Character overlap between chunks.")

    def handle(self, *args, **options):
        paths = [Path(p).expanduser().resolve() for p in options["paths"]]
        for path in paths:
            if not path.exists():
                raise CommandError(f"Path not found: {path}")

        extractor = LovdataExtractor()
        chunker = LangChainSectionChunker(
            chunk_size=options["chunk_size"],
            chunk_overlap=options["overlap"],
        )
        try:
            embedding_service = EmbeddingService()
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        metadata_repo = DjangoMetadataRepository()
        vector_store = LegalChunkStore()

        pipeline = IngestionPipeline(extractor, chunker, embedding_service, metadata_repo, vector_store)
        pipeline.ingest_paths(paths)

        self.stdout.write(self.style.SUCCESS("Ingestion completed successfully."))
