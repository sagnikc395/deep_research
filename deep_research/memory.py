import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

_DB_PATH = Path(__file__).resolve().parent.parent / "memory.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT    NOT NULL,
            subtasks    TEXT    NOT NULL,
            report      TEXT    NOT NULL,
            created_at  TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_session(query: str, subtasks: list[dict], report: str) -> int:
    conn = _get_connection()
    cur = conn.execute(
        "INSERT INTO research_sessions (query, subtasks, report, created_at) VALUES (?, ?, ?, ?)",
        (query, json.dumps(subtasks, ensure_ascii=False), report, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return session_id


def recall_relevant(query: str, limit: int = 3) -> list[dict]:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT id, query, subtasks, report, created_at FROM research_sessions "
        "ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()

    if not rows:
        return []

    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "of", "in", "to",
        "for", "and", "or", "on", "with", "by", "at", "from", "as", "it",
        "that", "this", "what", "how", "why", "who", "when", "where", "do",
        "does", "did", "be", "been", "being", "have", "has", "had", "not",
        "but", "if", "about", "which", "their", "there", "can", "will",
    }
    query_words = {w.lower() for w in query.split() if len(w) > 2} - stop_words

    scored: list[tuple[float, dict]] = []
    for row in rows:
        past_words = {w.lower() for w in row["query"].split() if len(w) > 2} - stop_words
        if not query_words or not past_words:
            continue
        overlap = len(query_words & past_words)
        if overlap > 0:
            score = overlap / len(query_words | past_words)
            scored.append((score, {
                "id": row["id"],
                "query": row["query"],
                "plan": row["subtasks"],
                "report": row["report"],
                "created_at": row["created_at"],
            }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]
