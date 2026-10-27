"""Конфигурация проекта: чтение переменных окружения из .env и настройка логов."""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Папка проекта: .env и относительный DATABASE_PATH ищутся в ней, а не в папке запуска
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_DATABASE_PATH = "memory.db"

# Значения-заглушки из .env.example не считаются настоящими ключами
_PLACEHOLDERS = {"your_telegram_bot_token", "your_openai_api_key"}
_REQUIRED_VARIABLES = ("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY")


@dataclass(frozen=True)
class Config:
    """Настройки бота. Объект неизменяемый, секреты скрыты из repr(), чтобы не попасть в логи."""

    telegram_bot_token: str = field(repr=False)
    openai_api_key: str = field(repr=False)
    openai_model: str
    database_path: Path


def load_config() -> Config:
    """Читает .env и возвращает настройки.

    Если обязательных переменных нет, печатает понятную ошибку и завершает программу.
    """
    load_dotenv(BASE_DIR / ".env")

    missing = [
        name
        for name in _REQUIRED_VARIABLES
        if os.getenv(name, "").strip() in ("", *_PLACEHOLDERS)
    ]
    if missing:
        print(
            f"Ошибка конфигурации: не заданы переменные {', '.join(missing)}.\n"
            "Скопируйте .env.example в .env и впишите настоящие значения.",
            file=sys.stderr,
        )
        sys.exit(1)

    database_path = Path(os.getenv("DATABASE_PATH", "").strip() or DEFAULT_DATABASE_PATH)
    if not database_path.is_absolute():
        database_path = BASE_DIR / database_path

    return Config(
        telegram_bot_token=os.environ["TELEGRAM_BOT_TOKEN"].strip(),
        openai_api_key=os.environ["OPENAI_API_KEY"].strip(),
        openai_model=os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL,
        database_path=database_path,
    )


class _RedactingFormatter(logging.Formatter):
    """Форматтер, который заменяет секреты на *** во всей строке лога, включая traceback."""

    def __init__(self, fmt: str, secrets: Iterable[str]) -> None:
        super().__init__(fmt)
        self._secrets = [secret for secret in secrets if secret]

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


def setup_logging(config: Config) -> None:
    """Включает вывод логов в консоль так, чтобы токен и ключ в них не попадали.

    - Логгер httpx на уровне INFO печатает адреса запросов к Telegram, а в адресе есть
      токен бота (https://api.telegram.org/bot<ТОКЕН>/...), поэтому поднимаем его до WARNING.
    - Сама python-telegram-bot при неверном токене пишет его в текст ошибки,
      поэтому секреты дополнительно вырезаются форматтером.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        _RedactingFormatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            secrets=(config.telegram_bot_token, config.openai_api_key),
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
