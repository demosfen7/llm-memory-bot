"""Telegram-бот БЕЗ памяти: каждый запрос уходит в LLM отдельно, история нигде не сохраняется."""
from __future__ import annotations

import logging
import sys

from telegram import BotCommand, Message, Update
from telegram.constants import ChatAction
from telegram.error import InvalidToken, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import load_config, setup_logging
from llm_client import LLMError, ask_llm, create_client

logger = logging.getLogger("bot_no_memory")

SYSTEM_PROMPT = "Ты — полезный ассистент в Telegram. Отвечай на русском языке, кратко и по делу."

START_TEXT = (
    "Привет! Я бот БЕЗ памяти.\n\n"
    "Каждое сообщение я обрабатываю отдельно и не сохраняю историю диалога. "
    "Если скажете, как вас зовут, а потом спросите об этом — я уже не вспомню.\n\n"
    "Просто напишите вопрос."
)

# Максимальная длина одного сообщения в Telegram
TELEGRAM_MESSAGE_LIMIT = 4096


async def send_answer(message: Message, text: str) -> None:
    """Отправляет ответ, разбивая его на части, если он длиннее лимита Telegram."""
    for start in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        await message.reply_text(text[start : start + TELEGRAM_MESSAGE_LIMIT])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /start: объясняет, что бот не помнит историю."""
    await update.effective_message.reply_text(START_TEXT)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет текст пользователя в LLM без какого-либо предыдущего контекста."""
    message = update.effective_message
    user_id = update.effective_user.id
    logger.info("Сообщение от user=%s: %s", user_id, message.text)

    # В модель уходят только системный промпт и текущее сообщение — истории нет
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message.text},
    ]

    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
    try:
        answer = await ask_llm(context.bot_data["llm"], context.bot_data["model"], messages)
    except LLMError as exc:
        await message.reply_text(f"Не удалось получить ответ: {exc}")
        return

    logger.info("Ответ для user=%s: %s", user_id, answer)
    await send_answer(message, answer)


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отвечает на команды, которых у бота нет."""
    await update.effective_message.reply_text("Не знаю такой команды. Доступна только /start.")


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
    await application.bot.set_my_commands([BotCommand("start", "Что умеет бот")])


def main() -> None:
    """Точка входа: читает конфиг, собирает приложение и запускает long polling."""
    config = load_config()
    setup_logging(config)

    application = Application.builder().token(config.telegram_bot_token).post_init(post_init).build()
    # Зависимости обработчиков храним в bot_data, а не в глобальных переменных
    application.bot_data["llm"] = create_client(config.openai_api_key)
    application.bot_data["model"] = config.openai_model

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    application.add_error_handler(on_error)

    logger.info("Бот без памяти запускается (модель: %s). Остановка — Ctrl+C.", config.openai_model)
    try:
        # Только новые сообщения: отредактированные и прочие обновления боту не нужны
        application.run_polling(allowed_updates=[Update.MESSAGE])
    except InvalidToken:
        logger.error("Telegram отклонил TELEGRAM_BOT_TOKEN: токен неверный или отозван. Проверьте .env.")
        sys.exit(1)


if __name__ == "__main__":
    main()
