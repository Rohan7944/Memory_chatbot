import streamlit as st
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
        "SELECT user_message, bot_response FROM chat_history ORDER BY timestamp ASC"
    )
    history = cursor.fetchall()
    conn.close()
    return history

def generate_response(prompt, chat_history):
    """Generates a response using Ollama, incorporating chat history."""
    messages = []
    for user_msg, bot_resp in chat_history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': bot_resp})

    messages.append({'role': 'user', 'content': prompt})

    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return response['message']['content']

# Initialize DB
init_db()

# Streamlit App UI
st.set_page_config(page_title="Chat with Ollama", layout="centered")
st.title("Chatbot with Ollama")
st.markdown(f"Model: `{OLLAMA_MODEL}`")

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = get_chat_history()

# Clear session chat button
if st.button("Clear Displayed Chat"):
    st.session_state.chat_history = []

# Display chat messages from session
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

# User input
if prompt := st.chat_input("Say something..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        response = generate_response(prompt, get_chat_history())
        save_message(prompt, response)
        st.session_state.chat_history.append((prompt, response))

    with st.chat_message("assistant"):
        st.markdown(response)
