"""Долговременная память бота: история сообщений и резюме пользователя в SQLite."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

# Путь по умолчанию совпадает с DATABASE_PATH=memory.db из .env.example.
# Боты передают путь из конфига явно через аргумент db_path.
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_context (
    user_id INTEGER PRIMARY KEY,
    summary TEXT DEFAULT '',
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Ускоряет выборку последних сообщений конкретного пользователя
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages (user_id, id);
"""


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Открывает соединение, фиксирует транзакцию при успехе и всегда закрывает соединение."""
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:  # commit при успехе, rollback при исключении
            yield connection


def init_db(*, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Создаёт таблицы, если их ещё нет."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as connection:
        connection.executescript(_SCHEMA)


def save_message(user_id: int, role: str, content: str, *, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Сохраняет одно сообщение пользователя ("user") или ассистента ("assistant")."""
    with _connect(db_path) as connection:
        # Значения передаются параметрами (?), а не вклеиваются в SQL: так SQL-инъекция невозможна
        connection.execute(
            "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )


def get_recent_messages(
    user_id: int, limit: int = 10, *, db_path: Path = DEFAULT_DB_PATH
) -> list[dict[str, str]]:
    """Возвращает последние `limit` сообщений пользователя в хронологическом порядке.

    Формат сразу подходит для OpenAI: [{"role": "user", "content": "..."}, ...].
    """
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    # Выбирали от новых к старым, чтобы взять именно последние; для модели разворачиваем обратно
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def count_user_messages(user_id: int, *, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Считает, сколько сообщений написал сам пользователь (нужно для обновления резюме)."""
    with _connect(db_path) as connection:
        (count,) = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ? AND role = 'user'",
            (user_id,),
        ).fetchone()
    return count


def get_user_summary(user_id: int, *, db_path: Path = DEFAULT_DB_PATH) -> str:
    """Возвращает краткое резюме пользователя или пустую строку, если его ещё нет."""
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT summary FROM user_context WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return row[0] if row and row[0] else ""


def update_user_summary(user_id: int, summary: str, *, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Сохраняет резюме пользователя или обновляет существующее."""
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_context (user_id, summary, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) DO UPDATE
                SET summary = excluded.summary, updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, summary),
        )


def clear_user_memory(user_id: int, *, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Удаляет все сообщения пользователя и его резюме. Других пользователей не трогает."""
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        connection.execute("DELETE FROM user_context WHERE user_id = ?", (user_id,))
