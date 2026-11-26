"""Persistent Image Library and Job History Database for Aarohan.

Manages persistent SQLite storage for generated assets, edits, favorites, tags,
collections, and asynchronous job states inside data/memory/aarohan.sqlite.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/memory/aarohan.sqlite")


class ImageLibraryDB:
    """Manages image library and job execution persistence in SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Create image_library and image_jobs tables if they do not exist."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_library (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    enhanced_prompt TEXT,
                    category TEXT DEFAULT 'general',
                    style TEXT DEFAULT 'cinematic',
                    aspect_ratio TEXT DEFAULT '1:1',
                    dimensions TEXT DEFAULT '768x768',
                    seed INTEGER,
                    provider TEXT DEFAULT 'aarohan_local',
                    model_name TEXT DEFAULT 'standard',
                    is_favorite INTEGER DEFAULT 0,
                    tags_json TEXT DEFAULT '[]',
                    collection TEXT DEFAULT 'default',
                    original_image_id TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_jobs (
                    job_id TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    enhanced_prompt TEXT,
                    progress REAL DEFAULT 0.0,
                    error_message TEXT,
                    output_filename TEXT,
                    output_file_path TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    metadata_json TEXT DEFAULT '{}'
                )
            """)
            conn.commit()

    def add_image(
        self,
        filename: str,
        file_path: str,
        prompt: str,
        enhanced_prompt: str = "",
        category: str = "general",
        style: str = "cinematic",
        aspect_ratio: str = "1:1",
        dimensions: str = "768x768",
        seed: Optional[int] = None,
        provider: str = "aarohan_local",
        model_name: str = "standard",
        tags: Optional[list[str]] = None,
        collection: str = "default",
        original_image_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Insert newly generated or edited image into persistent library."""
        img_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        tags_json = json.dumps(tags or [])

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO image_library (
                    id, filename, file_path, prompt, enhanced_prompt, category,
                    style, aspect_ratio, dimensions, seed, provider, model_name,
                    is_favorite, tags_json, collection, original_image_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    img_id,
                    filename,
                    str(file_path),
                    prompt,
                    enhanced_prompt,
                    category,
                    style,
                    aspect_ratio,
                    dimensions,
                    seed,
                    provider,
                    model_name,
                    tags_json,
                    collection,
                    original_image_id,
                    created_at,
                ),
            )
            conn.commit()

        return self.get_image(img_id) or {"id": img_id, "filename": filename}

    def list_images(
        self,
        search: Optional[str] = None,
        category: Optional[str] = None,
        favorite_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List images with optional keyword search and filtering."""
        query = "SELECT * FROM image_library WHERE 1=1"
        params: list[Any] = []

        if favorite_only:
            query += " AND is_favorite = 1"
        if category and category != "all":
            query += " AND category = ?"
            params.append(category)
        if search and search.strip():
            query += " AND (prompt LIKE ? OR enhanced_prompt LIKE ? OR filename LIKE ?)"
            term = f"%{search.strip()}%"
            params.extend([term, term, term])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["tags"] = json.loads(item.get("tags_json") or "[]")
                item["url"] = f"/api/images/view/{item['filename']}"
                item["download_url"] = f"/api/images/download/{item['filename']}"
                results.append(item)
            return results

    def get_image(self, image_id: str) -> Optional[dict[str, Any]]:
        """Fetch image details by UUID or filename."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM image_library WHERE id = ? OR filename = ?", (image_id, image_id))
            row = cursor.fetchone()
            if not row:
                return None
            item = dict(row)
            item["tags"] = json.loads(item.get("tags_json") or "[]")
            item["url"] = f"/api/images/view/{item['filename']}"
            item["download_url"] = f"/api/images/download/{item['filename']}"
            return item

    def toggle_favorite(self, image_id: str) -> bool:
        """Toggle favorite bookmark state."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE image_library SET is_favorite = (1 - is_favorite) WHERE id = ? OR filename = ?", (image_id, image_id))
            conn.commit()
            cursor.execute("SELECT is_favorite FROM image_library WHERE id = ? OR filename = ?", (image_id, image_id))
            row = cursor.fetchone()
            return bool(row["is_favorite"]) if row else False

    def delete_image(self, image_id: str) -> bool:
        """Delete image from database and filesystem."""
        img = self.get_image(image_id)
        if not img:
            return False
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM image_library WHERE id = ? OR filename = ?", (image_id, image_id))
            conn.commit()

        # Delete physical file
        try:
            p = Path(img["file_path"])
            if p.is_file():
                p.unlink()
        except Exception as e:
            logger.warning("Could not delete file %s: %s", img["file_path"], e)
        return True

    def record_job(
        self,
        job_id: str,
        provider_id: str,
        status: str,
        prompt: str,
        enhanced_prompt: str = "",
        progress: float = 0.0,
        error_message: Optional[str] = None,
        output_filename: Optional[str] = None,
        output_file_path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save or update asynchronous generation job state."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO image_jobs (
                    job_id, provider_id, status, prompt, enhanced_prompt, progress,
                    error_message, output_filename, output_file_path, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    progress = excluded.progress,
                    error_message = excluded.error_message,
                    output_filename = excluded.output_filename,
                    output_file_path = excluded.output_file_path,
                    metadata_json = excluded.metadata_json,
                    completed_at = CASE WHEN excluded.status IN ('completed', 'failed', 'cancelled') THEN datetime('now') ELSE completed_at END
                """,
                (
                    job_id,
                    provider_id,
                    status,
                    prompt,
                    enhanced_prompt,
                    progress,
                    error_message,
                    output_filename,
                    output_file_path,
                    datetime.now().isoformat(),
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM image_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            item = dict(row)
            item["metadata"] = json.loads(item.get("metadata_json") or "{}")
            if item.get("output_filename"):
                item["output_url"] = f"/api/images/view/{item['output_filename']}"
            return item

    def list_jobs(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM image_jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            res = []
            for r in rows:
                item = dict(r)
                item["metadata"] = json.loads(item.get("metadata_json") or "{}")
                if item.get("output_filename"):
                    item["output_url"] = f"/api/images/view/{item['output_filename']}"
                res.append(item)
            return res
