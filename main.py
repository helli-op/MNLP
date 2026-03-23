import os
from assistant import DocumentAssistant
import json

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

TOKEN = os.getenv("TELEGRAM_TOKEN")  
assistant = DocumentAssistant(chunk_size=500, overlap=50, top_k=3)
with open('data.json', 'r', encoding='utf-8') as file:
    documents = json.load(file)

print("Indexing documents...")
assistant.index_documents(documents)
user_queries = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text
    answer, query_id = assistant.answer_query(query)
    await update.message.reply_text(answer)
    keyboard = [
        [
            InlineKeyboardButton("⭐1", callback_data="1"),
            InlineKeyboardButton("⭐2", callback_data="2"),
            InlineKeyboardButton("⭐3", callback_data="3"),
            InlineKeyboardButton("⭐4", callback_data="4"),
            InlineKeyboardButton("⭐5", callback_data="5"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Оцените ответ:",
        reply_markup=reply_markup
    )
    user_queries[user_id] = query_id

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Здравствуйте! Я - ассистент по онбордингу в компании, помогаю сотрудникам быстрее пройти адаптацию. Задайте ваш вопрос.")

async def handle_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    rating = int(query.data)

    query_id = user_queries.get(user_id)

    if not query_id:
        await query.edit_message_text("Не удалось найти запрос 😢")
        return

    assistant.log_feedback(query_id, user_id, rating)

    await query.edit_message_text(
        f"Спасибо за оценку!"
    )

if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_rating))

    app.run_polling()
