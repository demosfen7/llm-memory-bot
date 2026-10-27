"""Работа с OpenAI Chat Completions API."""
from __future__ import annotations

import logging

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Максимальная длина резюме пользователя в символах
SUMMARY_MAX_CHARS = 500

_SUMMARY_PROMPT = (
    "Ты ведёшь краткую память о пользователе для Telegram-ассистента. "
    "Из прежней памяти и нового фрагмента диалога выдели только важное о пользователе: "
    "имя, предпочтения, цели, важные факты. Объедини с прежней памятью, убери устаревшее, "
    "не пересказывай сам диалог. Пиши по-русски, коротко, в третьем лице, "
    f"не длиннее {SUMMARY_MAX_CHARS} символов. Если нового ничего нет, верни прежнюю память."
)


class LLMError(Exception):
    """Ошибка обращения к модели. Текст исключения можно показать пользователю."""


def create_client(api_key: str) -> AsyncOpenAI:
    """Создаёт асинхронный клиент OpenAI (один на всё время работы бота)."""
    return AsyncOpenAI(api_key=api_key, timeout=60.0, max_retries=2)


def _describe_error(exc: openai.OpenAIError) -> str:
    """Переводит ошибку OpenAI в понятное сообщение без технических деталей."""
    if isinstance(exc, openai.AuthenticationError):
        return "Неверный OPENAI_API_KEY. Проверьте ключ в .env."
    if isinstance(exc, openai.PermissionDeniedError):
        return "У ключа OpenAI нет доступа к этой модели или региону."
    if isinstance(exc, openai.RateLimitError):
        return "Превышен лимит запросов OpenAI или закончились средства на балансе. Попробуйте позже."
    if isinstance(exc, openai.NotFoundError):
        return "Модель не найдена. Проверьте OPENAI_MODEL в .env."
    if isinstance(exc, openai.BadRequestError):
        return "OpenAI не смог обработать запрос. Попробуйте сократить или переформулировать сообщение."
    if isinstance(exc, openai.APIConnectionError):  # сюда же относятся таймауты
        return "Нет связи с OpenAI. Проверьте интернет и попробуйте ещё раз."
    return "Сервис OpenAI вернул ошибку. Попробуйте ещё раз чуть позже."


async def ask_llm(client: AsyncOpenAI, model: str, messages: list[dict[str, str]]) -> str:
    """Отправляет сообщения в Chat Completions API и возвращает только текст ответа ассистента.

    messages — список в формате OpenAI: [{"role": "system" | "user" | "assistant", "content": "..."}].
    При ошибке API пишет подробности в лог и поднимает LLMError с понятным текстом.
    """
    try:
        response = await client.chat.completions.create(model=model, messages=messages)
    except openai.OpenAIError as exc:
        message = _describe_error(exc)
        logger.error("Ошибка OpenAI: %s (%s: %s)", message, type(exc).__name__, exc)
        raise LLMError(message) from exc

    answer = response.choices[0].message.content
    if not answer:
        raise LLMError("Модель вернула пустой ответ. Попробуйте переформулировать вопрос.")
    return answer.strip()


async def summarize_user(
    client: AsyncOpenAI,
    model: str,
    previous_summary: str,
    dialog: list[dict[str, str]],
) -> str:
    """Просит модель обновить краткое резюме пользователя по свежему фрагменту диалога."""
    speakers = {"user": "Пользователь", "assistant": "Ассистент"}
    dialog_text = "\n".join(f"{speakers.get(m['role'], m['role'])}: {m['content']}" for m in dialog)
    messages = [
        {"role": "system", "content": _SUMMARY_PROMPT},
        {
            "role": "user",
            "content": f"Прежняя память:\n{previous_summary or '(пусто)'}\n\nНовый фрагмент диалога:\n{dialog_text}",
        },
    ]
    summary = await ask_llm(client, model, messages)
    # Модель может не уложиться в лимит — обрезаем гарантированно
    return summary[:SUMMARY_MAX_CHARS]
