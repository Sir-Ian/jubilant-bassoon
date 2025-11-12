from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.embeddings.manager import EmbeddingManager
from app.embeddings.pipeline import iter_product_text
from app.embeddings.store import EmbeddingStore


class FakeService:
    def __init__(self) -> None:
        self.model_name = "test-embedding-model"
        self._calls: list[str] = []

    def embed_text(self, text: str):
        self._calls.append(text)
        return [float(len(text))]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


@pytest.fixture
def store(tmp_path: Path) -> EmbeddingStore:
    db_path = tmp_path / "embeddings.db"
    return EmbeddingStore(db_path)


def test_embedding_store_roundtrip(store: EmbeddingStore) -> None:
    vector = [0.1, 0.2, 0.3]
    store.upsert(
        item_type="product",
        item_id="sku-1",
        text_hash="hash",
        model="test",
        vector=vector,
    )

    record = store.get("product", "sku-1", "test")
    assert record is not None
    assert record.vector == vector


def test_manager_reuses_cached_embeddings(store: EmbeddingStore) -> None:
    service = FakeService()
    manager = EmbeddingManager(store=store, service=service, item_type="product")

    vector1 = manager.get_or_create("sku-1", "vitamin c serum")
    vector2 = manager.get_or_create("sku-1", "vitamin c serum")

    assert vector1 == vector2
    # Underlying API should only be called once thanks to caching
    assert service._calls == ["vitamin c serum"]


def test_bulk_get_or_create_handles_mixed_cache(store: EmbeddingStore) -> None:
    service = FakeService()
    manager = EmbeddingManager(store=store, service=service)

    inputs = [
        ("sku-1", "hydrating moisturizer"),
        ("sku-2", "matte sunscreen"),
    ]

    first_pass = manager.bulk_get_or_create(inputs)
    assert set(first_pass.keys()) == {"sku-1", "sku-2"}
    assert len(service._calls) == 2

    second_pass = manager.bulk_get_or_create(inputs)
    assert second_pass == first_pass
    # No additional API calls expected due to cache reuse
    assert len(service._calls) == 2


def test_iter_product_text_builds_payloads() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "product_id": ["1", "2"],
            "name": ["Serum", "Moisturizer"],
            "category": ["Treatment", "Hydration"],
        }
    )

    payloads = iter_product_text(df, id_column="product_id", text_columns=["name", "category"])
    assert payloads == [
        ("1", "Serum | Treatment"),
        ("2", "Moisturizer | Hydration"),
    ]
