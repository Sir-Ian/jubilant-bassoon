"""Embedding utilities for the Beauty Advisor application."""

from .manager import EmbeddingManager
from .pipeline import build_embeddings_for_catalog, iter_product_text
from .service import EmbeddingService, create_embedding_service
from .store import EmbeddingRecord, EmbeddingStore

__all__ = [
    "EmbeddingManager",
    "EmbeddingRecord",
    "build_embeddings_for_catalog",
    "iter_product_text",
    "EmbeddingService",
    "EmbeddingStore",
    "create_embedding_service",
]
