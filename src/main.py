from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler, CallbackQueryHandler
import requests
import re
import src.bot_states as bot_states
import src.bot_messages as bot_messages
import telegram
import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import os

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

custom_keyboard = [['/check']]
admin_keyboard = [['/check'], ['/admin_panel']]

client_keyboard = [['/check'], ['/makeorder'], ['/back']]


standart_markup = telegram.ReplyKeyboardMarkup(custom_keyboard, resize_keyboard=True)
admin_markup = telegram.ReplyKeyboardMarkup(admin_keyboard, resize_keyboard=True)
client_markup = telegram.ReplyKeyboardMarkup(client_keyboard, resize_keyboard=True)


def start(update, context):
    update.message.reply_text(bot_messages.start_command_response, reply_markup=standart_markup)


def check(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text=bot_messages.client_response)

    return bot_states.CHECK

def ml_part(update, context):
    text = update.message.text
    context.bot.send_message(chat_id=update.message.chat_id, text=text + "\n")


# FOR DEBUGING
def ping(update, context):
    print("context")
    print(type(context))
    print("update")
    print(update)


token = os.environ.get('TELEGRAMMARLY_BOT_TOKEN')
updater = Updater(token, use_context=True)
dp = updater.dispatcher


def main():
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('ping', ping))  # TODO
    check_handler = ConversationHandler(
        entry_points=[CommandHandler('check', check)],

        states={
            bot_states.CHECK: [MessageHandler(Filters.text, ml_part)]
        },

        fallbacks=[CommandHandler('start', start)]
    )
    dp.add_handler(check_handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
