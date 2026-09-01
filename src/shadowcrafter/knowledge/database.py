"""Build a portable SQLite FTS5 security knowledge database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  source_url TEXT,
  license TEXT NOT NULL,
  retrieved_at TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  metadata_json TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
  title, body, content='documents', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
  INSERT INTO documents_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
  INSERT INTO documents_fts(documents_fts, rowid, title, body)
  VALUES ('delete', old.rowid, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
  INSERT INTO documents_fts(documents_fts, rowid, title, body)
  VALUES ('delete', old.rowid, old.title, old.body);
  INSERT INTO documents_fts(rowid, title, body) VALUES (new.rowid, new.title, new.body);
END;
"""


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)


def upsert_document(path: Path, document: dict[str, Any]) -> None:
    required = {
        "id",
        "source_id",
        "license",
        "retrieved_at",
        "content_sha256",
        "title",
        "body",
    }
    missing = required - document.keys()
    if missing:
        raise ValueError(f"missing document fields: {sorted(missing)}")
    metadata = document.get("metadata", {})
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO documents (
              id, source_id, source_url, license, retrieved_at,
              content_sha256, title, body, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              source_id=excluded.source_id,
              source_url=excluded.source_url,
              license=excluded.license,
              retrieved_at=excluded.retrieved_at,
              content_sha256=excluded.content_sha256,
              title=excluded.title,
              body=excluded.body,
              metadata_json=excluded.metadata_json
            """,
            (
                document["id"],
                document["source_id"],
                document.get("source_url"),
                document["license"],
                document["retrieved_at"],
                document["content_sha256"],
                document["title"],
                document["body"],
                json.dumps(metadata, sort_keys=True, ensure_ascii=False),
            ),
        )


def search(path: Path, query: str, limit: int = 10) -> list[dict[str, Any]]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT d.*, bm25(documents_fts) AS rank
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    return [dict(row) for row in rows]
