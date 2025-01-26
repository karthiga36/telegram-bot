from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN: Final = '8142624450:AAFz1XSfPcsDJw0mCwSljBlO7HA0TeMhFa8'
BOT_USERNAME: Final = '@Mykay29bot'

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Guten Tag, I am your AI assistant!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Please type something and let me help you!')

async def query_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('write your own queries here!')


def handle_response(text: str ) -> str:
    processed: str = text.lower()
    if 'hello' in processed:
        return 'Hello Leibe!'
    if 'how are you' in processed:
        return 'Ganz Gut! Wie gehts dir?!'
    if 'i need your assistance on my projects' in processed:
        return 'Sure, Go ahead!'
    
    return 'I dont understand your query, rephrase it...'

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "text"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)

        else:
            return
    else:
        response: str = handle_response(text)


    print('Bot:', response)
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if __name__ == '__main__':
    app= Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('quey', query_command))

    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    app.add_error_handler(error)


    print('Polling...')
    app.run_polling(poll_interval=3)