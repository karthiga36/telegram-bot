import pytz
from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import Update
from telegram.ext import Application, MessageHandler, CallbackContext, filters, ContextTypes

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(input_text):
    
    inputs = tokenizer(input_text, return_tensors="pt")
    
    outputs = model.generate(inputs["input_ids"], max_length=100, do_sample=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! I’m your AI assistant. Ask me anything about your favorite animal!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
   
    user_message = update.message.text
    print(f"User message: {user_message}")
 
    ai_response = generate_response(user_message)
  
    await update.message.reply_text(ai_response)


async def handle_message(update: Update, context: CallbackContext):
   
    user_message = update.message.text
    print(f"User message: {user_message}")
 
    ai_response = generate_response(user_message)
  
    await update.message.reply_text(ai_response)



API_TOKEN = "8142624450:AAFz1XSfPcsDJw0mCwSljBlO7HA0TeMhFa8"
timezone = pytz.timezone('CET')
application = Application.builder().token(API_TOKEN).build()


application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    

if __name__ == "__main__":
    application.run_polling()
    print("Bot is running.")
