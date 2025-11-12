"""Embedding service abstractions for OpenAI and Azure OpenAI."""

from __future__ import annotations

from typing import Iterable, List

from openai import AzureOpenAI, OpenAI

from ..config import EmbeddingConfig, EmbeddingProvider


class EmbeddingService:
    """Wrap the OpenAI/Azure embedding client with batching helpers."""

    def __init__(self, client: OpenAI | AzureOpenAI, model_name: str, batch_size: int = 32) -> None:
        self._client = client
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_text(self, text: str) -> List[float]:
        """Generate a single embedding."""

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Generate embeddings in batches to respect rate limits."""

        texts = list(texts)
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            # API returns data in same order
            embeddings.extend([item.embedding for item in response.data])
        return embeddings


def create_embedding_service(config: EmbeddingConfig, *, batch_size: int = 32) -> EmbeddingService:
    """Factory for constructing an embedding service from configuration."""

    if config.provider is EmbeddingProvider.OPENAI:
        if not config.openai_api_key:
            raise RuntimeError("OpenAI API key missing from configuration")
        client = OpenAI(api_key=config.openai_api_key)
        return EmbeddingService(client, model_name=config.model, batch_size=batch_size)

    # Azure OpenAI uses deployment name as model identifier
    if not (config.openai_api_key and config.azure_endpoint and config.azure_api_version):
        raise RuntimeError("Azure OpenAI configuration incomplete")
    client = AzureOpenAI(
        api_key=config.openai_api_key,
        api_version=config.azure_api_version,
        azure_endpoint=config.azure_endpoint,
    )
    return EmbeddingService(client, model_name=config.model, batch_size=batch_size)
