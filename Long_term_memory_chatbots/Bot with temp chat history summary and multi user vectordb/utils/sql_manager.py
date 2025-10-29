import os
import sqlite3
import datetime

from dotenv import load_dotenv
load_dotenv()

def getconnobject():
    DB_NAME = os.getenv("TEMP_MEMORY_DB_NAME")
    db_path = f"data/{DB_NAME}"
    return sqlite3.connect(db_path)

def get_chat_summary_record(user_id: str):
    """Get chat summaries"""
    conn = getconnobject()
    cursor = conn.cursor()
    #Execute query
    cursor.execute(
        "SELECT summary_text FROM chat_summary WHERE user_id = ? ORDER BY timestamp ASC",
        (user_id,)
    )
    record = cursor.fetchall()
    conn.close()
    print(f"VERBOSE: Fetched all chat summary record")
    
    return record

def save_chat_summary_record(chat_summary: str, user_id: str):
    """Saves a chat summary to the database."""
    conn = getconnobject()
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO chat_summary (summary_text, timestamp, user_id) VALUES (?, ?, ?)",
        (chat_summary, timestamp, user_id)
    )
    
    conn.commit()
    print(f"VERBOSE: Saved chat summary")
    
    conn.close()
    
def get_chat_history(user_id: str):
    """Retrieves recent chat history from the database."""
    conn = getconnobject()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT user_message, bot_response FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC",
        (user_id,)
    )
    history = cursor.fetchall()
    print(f"VERBOSE: Fetched all chat history if there was any")
    
    conn.close()
    
    return history

def save_chat_responses(user_message: str, bot_response: str, user_id: str):
    """Saves a user message and bot response to the database."""
    conn = getconnobject()
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO chat_history (user_message, bot_response, timestamp, user_id) VALUES (?, ?, ?, ?)",
        (user_message, bot_response, timestamp, user_id)
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
                             timestamp TEXT NOT NULL,
                             user_id TEXT NOT NULL
                        );
                             
                        CREATE TABLE IF NOT EXISTS chat_summary (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            summary_text TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            user_id TEXT NOT NULL
                        );
                         """)
    
    # Commit changes and close connection
    conn.commit()
    print(f"VERBOSE: Created tables if they did not exist")
    
    conn.close()