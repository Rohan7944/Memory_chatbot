import os
import sqlite3
import datetime

def getconnobject():
    # Configuration
    DB_NAME = 'chatbot_memory.db'
    db_path = f"data/{DB_NAME}"
    return sqlite3.connect(db_path)

def get_chat_summary_record():
    """Get chat summaries"""
    conn = getconnobject()
    cursor = conn.cursor()
    #Execute query
    cursor.execute(
        "SELECT summary_text FROM chat_summary ORDER BY timestamp ASC"
    )
    record = cursor.fetchall()
    conn.close()
    print(f"VERBOSE: Fetched all chat summary record")
    
    return record

def save_chat_summary_record(chat_summary: str):
    """Saves a chat summary to the database."""
    conn = getconnobject()
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO chat_summary (summary_text, timestamp) VALUES (?, ?)",
        (chat_summary, timestamp)
    )
    
    conn.commit()
    print(f"VERBOSE: Saved chat summary")
    
    conn.close()
    
def get_chat_history():
    """Retrieves recent chat history from the database."""
    conn = getconnobject()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT user_message, bot_response FROM chat_history ORDER BY timestamp ASC"
    )
    history = cursor.fetchall()
    print(f"VERBOSE: Fetched all chat history if there was any")
    
    conn.close()
    
    return history

def save_chat_responses(user_message: str, bot_response: str):
    """Saves a user message and bot response to the database."""
    conn = getconnobject()
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO chat_history (user_message, bot_response, timestamp) VALUES (?, ?, ?)",
        (user_message, bot_response, timestamp)
    )
    
    conn.commit()
    print(f"VERBOSE: Saved chat responses")
    
    conn.close()

def init_db():  
    
    # Connect to SQL db and create DB file if it does not exist
    if not os.path.exists("data"):
        os.makedirs("data")
    conn = getconnobject()
    cursor = conn.cursor()
    
    #Create tables in SQL DB
    cursor.executescript("""
                         CREATE TABLE IF NOT EXISTS chat_history (
                             id INTEGER PRIMARY KEY AUTOINCREMENT,
                             user_message TEXT NOT NULL,
                             bot_response TEXT NOT NULL,
                             timestamp TEXT NOT NULL
                        );
                             
                        CREATE TABLE IF NOT EXISTS chat_summary (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            summary_text TEXT NOT NULL,
                            timestamp TEXT NOT NULL
                        );
                         """)
    
    # Commit changes and close connection
    conn.commit()
    print(f"VERBOSE: Created tables if they did not exist")
    
    conn.close()