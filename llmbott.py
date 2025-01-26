import pytz
from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import Update
from telegram.ext import Application, MessageHandler, CallbackContext, filters

# ======== LLM Setup ======== #
# Load the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Replace with your chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Uncomment the line below to use GPU for faster inference
# model = model.to("cuda")

def generate_response(input_text):
    """
    Generate a response using the LLM for the given input text.
    """
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Uncomment if using GPU to move inputs to the same device
    # inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate the response with sampling
    outputs = model.generate(inputs["input_ids"], max_length=100, do_sample=True)
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ======== Telegram Bot Setup ======== #
# Replace this with your actual API token from BotFather
API_TOKEN = "8142624450:AAFz1XSfPcsDJw0mCwSljBlO7HA0TeMhFa8"
timezone = pytz.timezone('CET')

# Define a simple handler to process user input
async def handle_message(update: Update, context: CallbackContext):
    """
    Handle incoming messages from Telegram users.
    """
    # Retrieve the user's message
    user_message = update.message.text
    print(f"User message: {user_message}")
    
    # Generate a response using the LLM
    ai_response = generate_response(user_message)
    
    # Send the response back to the user
    await update.message.reply_text(ai_response)

def main():
    """
    Set up the Telegram bot and start polling for messages.
    """
    # Initialize the Application with the API token (no timezone required here)
    application = Application.builder().token(API_TOKEN).build()

    # Add a handler for text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot and let it run until you manually stop it
    application.run_polling()
    print("Bot is running... Press Ctrl+C to stop.")

if __name__ == "__main__":
    main()
