import ollama
import sqlite3
import datetime

# Configuration
OLLAMA_MODEL = 'llama3.2:1b'  # Replace with your desired Ollama model
DB_NAME = 'chatbot_memory.db'

def init_db():
    """Initializes the SQLite database for storing chat history."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_message(user_message, bot_response):
    """Saves a user message and bot response to the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO chat_history (user_message, bot_response, timestamp) VALUES (?, ?, ?)",
        (user_message, bot_response, timestamp)
    )
    conn.commit()
    conn.close()

def get_chat_history():
    """Retrieves recent chat history from the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_message, bot_response FROM chat_history ORDER BY timestamp DESC"
    )
    history = cursor.fetchall()
    conn.close()
    # Reverse the order to get chronological history
    return history[::-1]

def generate_response(prompt, chat_history):
    """Generates a response using Ollama, incorporating chat history."""
    messages = []
    # Add historical messages to provide context
    for user_msg, bot_resp in chat_history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': bot_resp})

    # Add the current user prompt
    messages.append({'role': 'user', 'content': prompt})

    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return response['message']['content']

def main():
    init_db()
    print(f"Chatbot initialized. Using Ollama model: {OLLAMA_MODEL}")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Get recent chat history for context
        history = get_chat_history() # Adjust limit as needed
        bot_response = generate_response(user_input, history)
        print(f"Bot: {bot_response}")

        save_message(user_input, bot_response)

if __name__ == "__main__":
    main()