"""Telegram-бот С долговременной памятью: история диалога и резюме пользователя хранятся в SQLite."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from openai import AsyncOpenAI
from telegram import BotCommand, Message, Update
from telegram.constants import ChatAction
from telegram.error import InvalidToken, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import database
from config import load_config, setup_logging
from llm_client import LLMError, ask_llm, create_client, summarize_user

logger = logging.getLogger("bot_memory")

# Сколько последних сообщений из SQLite передаём в модель, чтобы не раздувать контекст
HISTORY_LIMIT = 10
# Резюме обновляется после каждых SUMMARY_EVERY сообщений пользователя
SUMMARY_EVERY = 10
# Для резюме берём сообщения за последний «отрезок»: 10 реплик пользователя + 10 ответов
SUMMARY_SOURCE_MESSAGES = 2 * SUMMARY_EVERY

SYSTEM_PROMPT_TEMPLATE = """Ты — полезный ассистент в Telegram.

Ниже находится краткая память о пользователе и последние сообщения диалога.
Используй эту информацию, чтобы отвечать персонализированно и учитывать контекст.

Краткая память о пользователе:
{user_summary}

Отвечай на русском языке, кратко, дружелюбно и по делу."""

START_TEXT = (
    "Привет! Я бот С памятью.\n\n"
    "Я сохраняю наш диалог в базе данных и помню его даже после перезапуска. "
    "Каждые 10 сообщений я обновляю краткую заметку о вас: имя, цели, предпочтения.\n\n"
    "Команды:\n"
    "/start — это сообщение\n"
    "/reset — забыть всё, что я о вас знаю"
)

RESET_TEXT = "Память очищена. Начинаем с чистого листа."

# Максимальная длина одного сообщения в Telegram
TELEGRAM_MESSAGE_LIMIT = 4096


def build_system_prompt(user_summary: str) -> str:
    """Подставляет резюме пользователя в системный промпт."""
    return SYSTEM_PROMPT_TEMPLATE.format(user_summary=user_summary or "Пока ничего не известно.")


async def send_answer(message: Message, text: str) -> None:
    """Отправляет ответ, разбивая его на части, если он длиннее лимита Telegram."""
    for start in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        await message.reply_text(text[start : start + TELEGRAM_MESSAGE_LIMIT])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /start: объясняет, что бот помнит предыдущий диалог."""
    await update.effective_message.reply_text(START_TEXT)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /reset: удаляет историю и резюме только этого пользователя."""
    user_id = update.effective_user.id
    database.clear_user_memory(user_id, db_path=context.bot_data["db_path"])
    logger.info("Память user=%s очищена", user_id)
    await update.effective_message.reply_text(RESET_TEXT)


async def update_summary_if_needed(user_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """После каждых SUMMARY_EVERY сообщений пользователя просит LLM обновить его резюме.

    Резюме нужно, чтобы бот помнил ключевые факты (имя, цели, предпочтения) даже тогда,
    когда они выпали из последних HISTORY_LIMIT сообщений.
    """
    db_path: Path = context.bot_data["db_path"]
    if database.count_user_messages(user_id, db_path=db_path) % SUMMARY_EVERY != 0:
        return

    dialog = database.get_recent_messages(user_id, SUMMARY_SOURCE_MESSAGES, db_path=db_path)
    previous_summary = database.get_user_summary(user_id, db_path=db_path)
    try:
        summary = await summarize_user(
            context.bot_data["llm"], context.bot_data["model"], previous_summary, dialog
        )
    except LLMError:
        # Ответ пользователь уже получил; резюме обновим на следующем десятке сообщений
        logger.warning("Не удалось обновить резюме user=%s", user_id)
        return

    database.update_user_summary(user_id, summary, db_path=db_path)
    logger.info("Резюме user=%s обновлено: %s", user_id, summary)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отвечает на сообщение с учётом истории диалога и резюме пользователя из SQLite.

    SQLite-запросы здесь занимают доли миллисекунды, поэтому вызываются напрямую.
    Для нагруженного бота их стоило бы вынести в поток через asyncio.to_thread.
    """
    message = update.effective_message
    user_id = update.effective_user.id
    db_path: Path = context.bot_data["db_path"]
    llm: AsyncOpenAI = context.bot_data["llm"]
    logger.info("Сообщение от user=%s: %s", user_id, message.text)

    # 1. Сохраняем сообщение пользователя
    database.save_message(user_id, "user", message.text, db_path=db_path)

    # 2. Загружаем последние сообщения; только что сохранённое — последнее в этом списке,
    #    поэтому отдельно новое сообщение в запрос не добавляем (иначе оно продублируется)
    history = database.get_recent_messages(user_id, HISTORY_LIMIT, db_path=db_path)

    # 3. Загружаем краткое резюме пользователя
    user_summary = database.get_user_summary(user_id, db_path=db_path)

    # 4–5. Системный промпт с резюме + история диалога, заканчивающаяся новым сообщением
    messages = [{"role": "system", "content": build_system_prompt(user_summary)}, *history]

    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
    try:
        answer = await ask_llm(llm, context.bot_data["model"], messages)
    except LLMError as exc:
        await message.reply_text(f"Не удалось получить ответ: {exc}")
        return

    # 6. Сохраняем ответ ассистента
    database.save_message(user_id, "assistant", answer, db_path=db_path)
    logger.info("Ответ для user=%s: %s", user_id, answer)

    # 7. Отправляем ответ пользователю
    await send_answer(message, answer)

    # Резюме обновляем уже после ответа, чтобы пользователь не ждал лишний запрос к модели
    await update_summary_if_needed(user_id, context)


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отвечает на команды, которых у бота нет."""
    await update.effective_message.reply_text("Не знаю такой команды. Доступны /start и /reset.")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Пишет ошибку в лог и, если возможно, коротко сообщает о ней пользователю."""
    logger.error("Ошибка при обработке обновления", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("Произошла ошибка. Попробуйте ещё раз чуть позже.")
        except TelegramError:
            logger.warning("Не удалось отправить пользователю сообщение об ошибке")


async def post_init(application: Application) -> None:
    """Регистрирует команды в меню Telegram после подключения к API."""
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Что умеет бот"),
            BotCommand("reset", "Очистить память обо мне"),
        ]
    )


def main() -> None:
    """Точка входа: создаёт таблицы, собирает приложение и запускает long polling."""
    config = load_config()
    setup_logging(config)

    database.init_db(db_path=config.database_path)
    logger.info("База данных памяти: %s", config.database_path)

    application = Application.builder().token(config.telegram_bot_token).post_init(post_init).build()
    # Зависимости обработчиков храним в bot_data, а не в глобальных переменных
    application.bot_data["llm"] = create_client(config.openai_api_key)
    application.bot_data["model"] = config.openai_model
    application.bot_data["db_path"] = config.database_path

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    application.add_error_handler(on_error)

    logger.info("Бот с памятью запускается (модель: %s). Остановка — Ctrl+C.", config.openai_model)
    try:
        # Только новые сообщения: отредактированные и прочие обновления боту не нужны
        application.run_polling(allowed_updates=[Update.MESSAGE])
    except InvalidToken:
        logger.error("Telegram отклонил TELEGRAM_BOT_TOKEN: токен неверный или отозван. Проверьте .env.")
        sys.exit(1)


if __name__ == "__main__":
    main()
