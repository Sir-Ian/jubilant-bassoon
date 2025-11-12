"""Utilities to build and refresh embedding indexes from tabular data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .manager import EmbeddingManager
from .store import EmbeddingStore


def iter_product_text(df: pd.DataFrame, *, id_column: str, text_columns: Iterable[str]) -> list[tuple[str, str]]:
    """Prepare concatenated text payloads for embedding generation."""

    if id_column not in df.columns:
        raise KeyError(f"Dataframe missing id column: {id_column}")

    for column in text_columns:
        if column not in df.columns:
            raise KeyError(f"Dataframe missing required text column: {column}")

    payloads: list[tuple[str, str]] = []
    selected = df[[id_column, *text_columns]]
    for row in selected.itertuples(index=False, name="Row"):
        row_dict = row._asdict()
        identifier = str(row_dict[id_column])
        parts: list[str] = []
        for key, value in row_dict.items():
            if key == id_column or value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                parts.append(value_str)
        text = " | ".join(parts)
        payloads.append((identifier, text))
    return payloads


def build_embeddings_for_catalog(
    *,
    csv_path: Path,
    id_column: str,
    text_columns: Iterable[str],
    store: EmbeddingStore,
    manager: EmbeddingManager,
    chunksize: int = 512,
) -> None:
    """Stream through a CSV file and ensure embeddings exist for each row."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    reader = pd.read_csv(csv_path, chunksize=chunksize)
    for chunk in reader:
        payloads = iter_product_text(chunk, id_column=id_column, text_columns=text_columns)
        manager.bulk_get_or_create(payloads)
