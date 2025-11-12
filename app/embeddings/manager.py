"""High-level interface for embedding reuse."""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

from .service import EmbeddingService
from .store import EmbeddingRecord, EmbeddingStore


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingManager:
    """Coordinate embedding generation with caching."""

    def __init__(self, store: EmbeddingStore, service: EmbeddingService, item_type: str = "product") -> None:
        self.store = store
        self.service = service
        self.item_type = item_type

    @property
    def model_name(self) -> str:
        return self.service.model_name

    def get(self, item_id: str) -> EmbeddingRecord | None:
        return self.store.get(self.item_type, item_id, self.model_name)

    def get_or_create(self, item_id: str, text: str) -> Sequence[float]:
        existing = self.get(item_id)
        text_hash = _hash_text(text)
        if existing and existing.text_hash == text_hash:
            return existing.vector

        vector = self.service.embed_text(text)
        self.store.upsert(
            item_type=self.item_type,
            item_id=item_id,
            text_hash=text_hash,
            model=self.model_name,
            vector=vector,
        )
        return vector

    def bulk_get_or_create(self, items: Iterable[tuple[str, str]]) -> dict[str, Sequence[float]]:
        results: dict[str, Sequence[float]] = {}
        missing: list[tuple[str, str]] = []
        hashes: dict[str, str] = {}

        for item_id, text in items:
            text_hash = _hash_text(text)
            hashes[item_id] = text_hash
            record = self.get(item_id)
            if record and record.text_hash == text_hash:
                results[item_id] = record.vector
            else:
                missing.append((item_id, text))

        if missing:
            texts = [text for _, text in missing]
            vectors = self.service.embed_texts(texts)
            for (item_id, _), vector in zip(missing, vectors):
                self.store.upsert(
                    item_type=self.item_type,
                    item_id=item_id,
                    text_hash=hashes[item_id],
                    model=self.model_name,
                    vector=vector,
                )
                results[item_id] = vector

        return results
