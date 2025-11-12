"""SQLite-backed storage for embedding vectors."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass
class EmbeddingRecord:
    """Container representing a stored embedding."""

    item_type: str
    item_id: str
    text_hash: str
    model: str
    vector: List[float]
    created_at: float


class EmbeddingStore:
    """Persist embedding vectors in a lightweight SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialise(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    item_type TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (item_type, item_id, model)
                )
                """
            )
            conn.commit()

    @staticmethod
    def _serialise_vector(vector: Sequence[float]) -> str:
        return json.dumps([float(v) for v in vector])

    @staticmethod
    def _deserialise_vector(raw: str) -> List[float]:
        return [float(v) for v in json.loads(raw)]

    def upsert(
        self,
        *,
        item_type: str,
        item_id: str,
        text_hash: str,
        model: str,
        vector: Sequence[float],
        created_at: Optional[float] = None,
    ) -> None:
        """Insert or update an embedding record."""

        created_at = created_at or time.time()
        payload = (
            item_type,
            item_id,
            text_hash,
            model,
            self._serialise_vector(vector),
            created_at,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO embeddings (item_type, item_id, text_hash, model, vector, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_type, item_id, model)
                DO UPDATE SET text_hash=excluded.text_hash,
                              vector=excluded.vector,
                              created_at=excluded.created_at
                """,
                payload,
            )
            conn.commit()

    def get(self, item_type: str, item_id: str, model: str) -> Optional[EmbeddingRecord]:
        """Fetch an embedding if available."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT item_type, item_id, text_hash, model, vector, created_at\n"
                "FROM embeddings\n"
                "WHERE item_type=? AND item_id=? AND model=?",
                (item_type, item_id, model),
            ).fetchone()
        if not row:
            return None
        vector = self._deserialise_vector(row[4])
        return EmbeddingRecord(
            item_type=row[0],
            item_id=row[1],
            text_hash=row[2],
            model=row[3],
            vector=vector,
            created_at=row[5],
        )

    def iter_by_type(self, item_type: str) -> Iterable[EmbeddingRecord]:
        """Iterate over embeddings of a given type."""

        with self._connect() as conn:
            for row in conn.execute(
                "SELECT item_type, item_id, text_hash, model, vector, created_at\n"
                "FROM embeddings\n"
                "WHERE item_type=?",
                (item_type,),
            ):
                yield EmbeddingRecord(
                    item_type=row[0],
                    item_id=row[1],
                    text_hash=row[2],
                    model=row[3],
                    vector=self._deserialise_vector(row[4]),
                    created_at=row[5],
                )

    def delete(self, item_type: str, item_id: str, model: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM embeddings WHERE item_type=? AND item_id=? AND model=?",
                (item_type, item_id, model),
            )
            conn.commit()
