import os
import sqlite3
import datetime

from utils.vectorstore_manager import update_vector_store

from dotenv import load_dotenv
load_dotenv()

# Configuration
chats_tobesaved = os.getenv("SAVED_CHAT_CONVO")
summaries_tobesaved = os.getenv("SAVED_CHAT_SUMMARIES")

def delete_row(user_id: str, timestamp: str, table_name: str):
    """Deletes a row from table"""
    conn = getconnobject()
    cursor = conn.cursor()
    
    query = f"DELETE FROM {table_name} WHERE user_id = ? AND timestamp = ?"
    cursor.execute(query, (user_id, timestamp))

    conn.commit()
    print(f"VERBOSE: Deleted row from {table_name} table")
    
    conn.close()

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
        "SELECT summary_text, timestamp FROM chat_summary WHERE user_id = ? ORDER BY timestamp ASC",
        (user_id,)
    )
    
    record = cursor.fetchall()
    conn.close()
    print(f"VERBOSE: Fetched all chat summary record")
    
    # Need to return only the summary text
    result = [row[0] for row in record]
    
    if len(record) > int(summaries_tobesaved): # Check if records exceed threshold
        update_vector_store(new_summary = record[0][0], # Update the user vector store before deleting the oldest summary record
                            user_id     = user_id) 
        delete_row(user_id    = user_id, 
                   timestamp  = record[0][1], 
                   table_name = "chat_summary") # delete the oldest record from table
        
        return result[1:] # Return the rest
    else:
        return result # Return all the records

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
        "SELECT user_message, bot_response, timestamp FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC",
        (user_id,)
    )
    
    history = cursor.fetchall()
    conn.close()
    print(f"VERBOSE: Fetched all chat history if there was any")
    
    # Need to return only the user_message and bot_response
    result = [(row[0], row[1]) for row in history]
    
    if len(history) > int(chats_tobesaved): # Check if records exceed threshold
        delete_row(user_id    = user_id, 
                   timestamp  = history[0][2], 
                   table_name = "chat_history") # delete the oldest record from table
        return result[1:] # Return the rest
    else:
        return result # Return all the records

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